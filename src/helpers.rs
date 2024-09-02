use crate::{create_context_var, prelude::DEBUG};
use anyhow::{anyhow, Ok};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use inferno::flamegraph;
use lazy_static::lazy_static;
use memoize::memoize;
use pprof::{
    protos::{Message, Profile},
    ProfilerGuard, ProfilerGuardBuilder,
};
use regex::Regex;
use reqwest::{get, header::CONTENT_LENGTH};
use rusqlite::{params, params_from_iter, Connection};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, to_string, Value};
use serde_pickle::{DeOptions, SerOptions};
use std::{
    any::Any,
    array::IntoIter,
    cmp::Ordering,
    collections::{HashMap, HashSet},
    default, env,
    ffi::CString,
    fmt::Display,
    fs::{self, File},
    hash::Hash,
    io::{BufReader, BufWriter, Cursor, Read, Write},
    iter::{once, Product},
    ops::{Add, Deref, Div, Mul, Sub},
    os::raw::{c_char, c_void},
    path::PathBuf,
    process::Command,
    ptr, slice,
    str::{Bytes, FromStr},
    sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock},
    time::{self, Duration, Instant, UNIX_EPOCH},
};
use tempfile::{tempfile, NamedTempFile};
lazy_static! {
    pub static ref OSX: bool = cfg!(macos);
    pub static ref CI: bool = env::var("CI").unwrap_or_default() != "";
    pub static ref ContextStack: Arc<Mutex<Vec<HashMap<String, isize>>>> =
        Arc::new(Mutex::new(vec![HashMap::new()]));
    pub static ref ContextVarCache: Arc<Mutex<HashMap<String, ContextVar>>> =
        Arc::new(Mutex::new(HashMap::new()));
}
// Define the argfix macro
#[macro_export]
macro_rules! argfix {
    // Match when the argument is a single tuple or list
    ( ( $( $x:expr ),* ) ) => {
        {
            let args = ( $( $x ),* );
            if args.len() != 1 {
                compile_error!(concat!("bad arg ", stringify!($( $x ),*)));
            } // Call the helper function to check
            args  // Return the tuple as is
        }
    };

    // Match when the argument is not a tuple or list
    ( $( $x:expr ),* ) => {
        ( $( $x ),* )  // Return the arguments as a tuple
    };
}
#[macro_export]
macro_rules! make_pair {
    (($($x:expr), *)) => {
        ($($x), *)
    };
    (($($x:expr), *), cnt=$cnt:expr) => {
        ($($x), *)
    };
    // Case: single expression
    ($x:expr) => {
        ($x, $x)
    };

    // Case: single expression with count
    ($x:expr, cnt=$cnt:expr) => {
        {
            let mut result = Vec::new();
            for _ in 0..$cnt {
                result.push($x);
            }
            result
        }
    };
}

#[macro_export]
macro_rules! argsort {
    ($($x:expr), *) => {{
        let args = ($($x:expr), *);
        let mut indices: Vec<usize> = (0..args.len()).collect();

        indices.sort_by(|&a, &b| $x[a].cmp(&$x[b]));

        indices
    }};
}

pub fn prod<T>(x: impl IntoIterator<Item = T>) -> Option<T>
where
    T: Product,
{
    let iter = x.into_iter();
    if iter.size_hint().0 == 0 {
        None
    } else {
        Some(iter.product())
    }
}

pub fn dedup<T: Eq + Hash + Clone>(x: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for item in x {
        if seen.insert(item.clone()) {
            result.push(item.clone());
        }
    }

    result
}

pub fn all_same<T>(items: &[T]) -> bool
where
    T: PartialEq,
{
    items.iter().all(|x| x == &items[0])
}

pub fn colored<T>(st: T, color: Option<&str>, background: Option<bool>) -> String
where
    T: Display,
{
    match color {
        Some(c) => format!(
            "\u{001b}[{}]m{}\u{001b}[0m",
            10 * {
                if background.unwrap_or(false) {
                    1
                } else {
                    0
                }
            } + 60
                * ({
                    if c.to_uppercase() == c {
                        1
                    } else {
                        0
                    }
                })
                + 30
                + {
                    vec![
                        "black", "red", "green", "yellow", "blue", "magenta", "cyan", "white",
                    ]
                    .into_iter()
                    .position(|x| x == c.to_lowercase())
                    .unwrap()
                },
            st
        ),
        None => st.to_string(),
    }
}

pub fn ansistrip(s: &str) -> String {
    Regex::new(r"\x1b\[([KM]|.*?m)")
        .unwrap()
        .replace_all(s, "")
        .to_string()
}

pub fn ansilen(s: &str) -> usize {
    ansistrip(s).len()
}
// use in-built flatten
// pub fn flatten<T>(l: impl IntoIterator<Item = impl IntoIterator<Item = T>>) -> Vec<T>{
//     let mut ret = Vec::new();

//     for sublist in l{
//         for item in sublist{
//             ret.push(item);
//         }
//     }

//     ret
// }

pub fn strip_parens<'a>(fst: &'a str) -> &'a str {
    if fst.chars().nth(0) == Some('(')
        && fst.chars().nth(fst.len() - 1) == Some(')')
        && fst[1..fst.len() - 1]
            .chars()
            .into_iter()
            .position(|x| x == '(')
            <= fst[1..fst.len() - 1]
                .chars()
                .into_iter()
                .position(|x| x == ')')
    {
        &fst[1..fst.len() - 1]
    } else {
        fst
    }
}

pub trait FloorDiv<RHS = Self> {
    type Output;
    fn floor_div(self, rhs: RHS) -> Self::Output;
}

#[macro_export]
// Implement FloorDiv for signed integers
macro_rules! impl_floor_div_signed {
    ($t:ty) => {
        impl FloorDiv<$t> for $t {
            type Output = $t;

            fn floor_div(self, rhs: $t) -> $t {
                self / rhs
            }
        }
    };
}

#[macro_export]
// Implement FloorDiv for unsigned integers
macro_rules! impl_floor_div_unsigned {
    ($t:ty) => {
        impl FloorDiv<$t> for $t {
            type Output = $t;

            fn floor_div(self, rhs: $t) -> $t {
                self / rhs
            }
        }
    };
}

#[macro_export]
// Implement FloorDiv for f32 and f64
macro_rules! impl_floor_div_float {
    ($t:ty) => {
        impl FloorDiv<$t> for $t {
            type Output = $t;

            fn floor_div(self, rhs: $t) -> $t {
                (self / rhs).floor()
            }
        }
    };
}

// Implement FloorDiv for f32 and f64 specifically
impl_floor_div_float!(f32);
impl_floor_div_float!(f64);
// Implement FloorDiv for isize and usize
impl_floor_div_signed!(isize);
impl_floor_div_signed!(i64);
impl_floor_div_signed!(i32);
impl_floor_div_signed!(i16);
impl_floor_div_signed!(i8);
impl_floor_div_unsigned!(usize);
impl_floor_div_signed!(u128);
impl_floor_div_signed!(u64);
impl_floor_div_signed!(u32);
impl_floor_div_signed!(u16);
impl_floor_div_signed!(u8);
pub trait ToInteger {
    fn to_isize(self) -> isize;
}
impl ToInteger for f64 {
    fn to_isize(self) -> isize {
        if self >= (isize::MIN as f64) && self <= (isize::MAX as f64) {
            self as isize
        } else {
            panic!("Conversion from f64 to isize would overflow");
        }
    }
}

impl ToInteger for f32 {
    fn to_isize(self) -> isize {
        if self >= (isize::MIN as f32) && self <= (isize::MAX as f32) {
            self as isize
        } else {
            panic!("Conversion from f32 to isize would overflow");
        }
    }
}
// Generic function `round_up` that takes any numeric type that supports the operations we need
pub fn round_up<T>(num: T, amt: T) -> isize
where
    T: Copy
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + From<u8>
        + PartialOrd
        + FloorDiv<Output = T>
        + ToInteger,
{
    // Calculate the ceiling of num divided by amt
    let result = ((num + amt - T::from(1u8)).floor_div(amt).to_isize()) * amt.to_isize();
    result
}

pub fn merge_hms<T, U>(ds: impl IntoIterator<Item = HashMap<T, U>>) -> HashMap<T, U>
where
    T: Eq + Hash,
    U: Eq + Hash,
{
    let kvs: HashSet<(T, U)> = ds.into_iter().flat_map(|d| d.into_iter()).collect();

    let keys: HashSet<_> = kvs.iter().map(|(k, _)| k).collect();
    assert!(kvs.len() == keys.len());

    kvs.into_iter().collect()
}

pub fn partition<T>(lst: Vec<T>, fxn: impl Fn(&T) -> bool) -> (Vec<T>, Vec<T>) {
    let mut a = Vec::new();
    let mut b = Vec::new();

    for s in lst.into_iter() {
        if fxn(&s) {
            a.push(s);
        } else {
            b.push(s);
        }
    }
    return (a, b);
}

pub fn get_nested_value<'a>(data: &'a Value, key: &str) -> Option<&'a Value> {
    let mut current_obj = data;

    for k in key.split('.') {
        current_obj = match current_obj {
            Value::Object(map) => map.get(k)?,
            Value::Array(arr) => {
                let idx = k.parse::<usize>().ok()?;
                arr.get(idx)?
            }
            _ => return None,
        };
    }

    Some(current_obj)
}

// IMPLEMENT GET_CONTRACTION

#[memoize]
pub fn to_function_name(s: String) -> String {
    s.chars()
        .into_iter()
        .filter(|c| c.is_ascii_alphanumeric() || c == &'_')
        .collect()
}

#[memoize]
pub fn getenv(key: String, default: Option<String>) -> String {
    match env::var(key) {
        std::result::Result::Ok(v) => v,
        Err(_) => default.unwrap_or("1".to_string()),
    }
}

pub fn temp(x: &str) -> String {
    let mut temp_dir = env::temp_dir();
    temp_dir.push(PathBuf::from(x));
    temp_dir.to_string_lossy().into_owned()
}

pub struct Context {
    pub stack: Arc<Mutex<Vec<HashMap<String, isize>>>>,
    pub kwargs: HashMap<String, isize>,
}

impl Context {
    pub fn init(&mut self) {
        let mut stack = self.stack.lock().unwrap();
        let stack_l = stack.len();
        let mut cache: std::sync::MutexGuard<HashMap<String, ContextVar>> =
            ContextVarCache.lock().unwrap();
        stack[stack_l - 1] = {
            let mut ret = HashMap::new();
            cache.iter().for_each(|(k, o)| {
                ret.insert(k.clone(), o.value.clone());
            });
            ret
        };
        for (k, v) in self.kwargs.iter() {
            cache.get_mut(k).unwrap().value = v.clone();
        }
        stack.push(self.kwargs.clone());
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let mut stack = self.stack.lock().unwrap();
        let mut cache: std::sync::MutexGuard<HashMap<String, ContextVar>> =
            ContextVarCache.lock().unwrap();
        match stack.pop() {
            Some(kk) => {
                for (k, _) in kk {
                    cache.get_mut(&k).unwrap().value = stack[stack.len() - 1]
                        .get(&k)
                        .unwrap_or(&cache.get(&k).unwrap().value)
                        .clone()
                }
            }
            None => {
                // do nothin
            }
        }
    }
}
#[derive(Clone)]
pub struct ContextVar {
    pub cache: Arc<Mutex<HashMap<String, ContextVar>>>,
    pub value: isize,
    pub key: String,
}

impl Into<bool> for ContextVar {
    fn into(self) -> bool {
        self.value == 1
    }
}
impl<T> PartialEq<T> for ContextVar
where
    T: PartialEq<isize>,
{
    fn eq(&self, other: &T) -> bool {
        other == &self.value
    }
}
impl<T> PartialOrd<T> for ContextVar
where
    T: PartialOrd<isize>,
{
    fn ge(&self, other: &T) -> bool {
        match other.partial_cmp(&self.value) {
            Some(Ordering::Less) | Some(Ordering::Equal) => true,
            _ => false,
        }
    }
    fn gt(&self, other: &T) -> bool {
        match other.partial_cmp(&self.value) {
            Some(Ordering::Less) => true,
            _ => false,
        }
    }
    fn lt(&self, other: &T) -> bool {
        match other.partial_cmp(&self.value) {
            Some(Ordering::Greater) => true,
            _ => false,
        }
    }
    fn le(&self, other: &T) -> bool {
        match other.partial_cmp(&self.value) {
            Some(Ordering::Greater) | Some(Ordering::Equal) => true,
            _ => false,
        }
    }

    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        Some(other.partial_cmp(&self.value)?)
    }
}
#[macro_export]
macro_rules! create_context_var {
    ($key: expr, $default_value: expr) => {{
        use crate::helpers::{getenv, ContextVarCache};
        let c = ContextVarCache.clone();
        let mut cache = c.lock().unwrap();
        match cache.get(&($key.to_string())) {
            Some(k) => k.clone(),
            None => {
                let instance = ContextVar {
                    cache: c.clone(),
                    value: getenv($key.to_string(), Some($default_value.to_string()))
                        .parse()
                        .unwrap(),
                    key: $key.to_string(),
                };
                cache.insert($key.to_string(), instance.clone());
                instance
            }
        }
    }};
}
#[macro_export]
macro_rules! create_new_context {
    ($($kwargs:expr),*) => {
        {
            use std::collections::HashMap;
            // use crate::rustgrad_2::helpers::{ContextStack, Context};
            use rustgrad_2::helpers::{Context, ContextStack};
            let v_kwargs = vec![$($kwargs),*];
            let k_v: Vec<Vec<String>> = v_kwargs
                .into_iter()
                .map(|x| x.split('=').map(|s| s.to_string()).collect::<Vec<_>>())
                .collect();
            let mut ret = HashMap::new();
            k_v.into_iter().for_each(|x| {
                assert_eq!(x.len(), 2);
                let lhs = x[0].trim().to_string();
                let rhs = x[1].trim().to_string().parse::<i32>().unwrap();
                ret.insert(lhs, rhs as isize);
            });

            Context {
                stack: ContextStack.clone(),
                kwargs: ret,
            }
        }
    };
}

#[macro_export]
macro_rules! parse_context_kwargs {
    ($var:ident) => {
        format!("{} = {:?}", stringify!($var), $var);
    };
}

lazy_static! {
    pub static ref GlobalCounter: Arc<Mutex<GlobalCounters>> =
        Arc::new(Mutex::new(GlobalCounters::default()));
}

//do not init
pub struct GlobalCounters {
    pub global_ops: isize,
    pub global_mem: usize,
    pub time_sum_s: f64,
    pub kernel_count: usize,
    pub mem_used: usize,
}
impl Default for GlobalCounters {
    fn default() -> Self {
        Self {
            global_ops: 0,
            global_mem: 0,
            time_sum_s: 0.0,
            kernel_count: 0,
            mem_used: 0,
        }
    }
}

impl GlobalCounters {
    pub fn reset(&mut self) {
        self.global_ops = 0;
        self.global_mem = 0;
        self.time_sum_s = 0.0;
        self.kernel_count = 0;
        self.mem_used = 0;
    }
}

pub struct Timing<T>
where
    T: Fn(Duration) -> String,
{
    prefix: String,
    on_exit: Option<T>,
    enabled: bool,
    st: Option<Instant>,
    et: Option<Duration>,
}

impl<T> Timing<T>
where
    T: Fn(Duration) -> String,
{
    pub fn new(prefix: Option<&str>, on_exit: Option<T>, enabled: Option<bool>) -> Self {
        Self {
            prefix: prefix.unwrap_or("").to_string(),
            on_exit,
            enabled: enabled.unwrap_or(true),
            st: None,
            et: None,
        }
    }

    pub fn init(&mut self) {
        self.st = Some(Instant::now());
    }
}

impl<T> Drop for Timing<T>
where
    T: Fn(Duration) -> String,
{
    fn drop(&mut self) {
        self.et = Some(Instant::now() - self.st.unwrap_or_else(|| panic!("Not inited")));
        if self.enabled {
            print!(
                "{}{}",
                format!(
                    "{}{:6.2} ms",
                    self.prefix,
                    self.et.unwrap().as_secs_f64() * 1e-6
                ),
                {
                    match &self.on_exit {
                        Some(func) => func(self.et.unwrap()),
                        None => String::new(),
                    }
                }
            )
        }
    }
}

pub struct Profiling<'a> {
    enabled: bool,
    sort: String,
    frac: f64,
    func: Option<String>,
    time_scale: f64,
    pr: Option<ProfilerGuard<'a>>,
}

impl Profiling<'_> {
    pub fn new(
        enabled: Option<bool>,
        sort: Option<&str>,
        frac: Option<f64>,
        func: Option<String>,
        ts: Option<f64>,
    ) -> Self {
        Profiling {
            enabled: enabled.unwrap_or(true),
            sort: sort.unwrap_or("cumtime").to_string(),
            frac: frac.unwrap_or(0.2),
            func,
            time_scale: 1e3 / ts.unwrap_or(1.0),
            pr: None,
        }
    }
    pub fn init(&mut self) -> Result<(), anyhow::Error> {
        if self.enabled {
            self.pr = Some(
                ProfilerGuardBuilder::default()
                    .build()
                    .expect("failed to start profing"),
            );
        }
        Ok(())
    }
}

impl Drop for Profiling<'_> {
    fn drop(&mut self) {
        if self.enabled {
            let stats = self.pr.as_mut().unwrap().report().build().unwrap();

            let profile = stats.pprof().expect("Failed to parse profile");
            if let Some(path) = &self.func {
                let mut file = File::create(path).unwrap();
                let mut content = Vec::new();
                profile.write_to_vec(&mut content).unwrap();

                file.write_all(&content).unwrap();
            }
            let sample_info = analyze_samples(&profile);
            let caller_map = extract_callers(&profile);

            let mut samples: Vec<(&String, i64, i64, i64)> = sample_info
                .iter()
                .map(|(k, v)| (k, v.0, v.1, v.2))
                .collect();
            if self.sort == "cumtime" {
                samples.sort_by(|a, b| b.3.cmp(&a.3));
            }
            for (func_name, num_calls, tottime, cumtime) in
                samples[0..(samples.len() as f64 * self.frac) as usize].into_iter()
            {
                if let Some(caller_map) = caller_map.get(*func_name) {
                    let callers = caller_map.keys().collect::<Vec<&String>>();
                    let mut scallers = callers.clone();
                    scallers.sort_by(|a, b| {
                        sample_info
                            .get(*a)
                            .unwrap()
                            .1
                            .cmp(&sample_info.get(*b).unwrap().1)
                    });

                    print!(
                        "n:{:8} tm:{:7.2}ms tot:{:7.2}ms {} {}",
                        num_calls,
                        *tottime as f64 * self.time_scale,
                        *cumtime as f64 * self.time_scale,
                        format!(
                            "{}{}",
                            colored(format_fcn(&profile, func_name), Some("yellow"), None),
                            vec![" "; 50 - format_fcn(&profile, func_name).len()].join("")
                        ),
                        {
                            if !scallers.is_empty() {
                                colored(
                                    format!(
                                        "<- {:3.0}% {}",
                                        sample_info.get(scallers[0]).unwrap().1 * 100,
                                        format_fcn(&profile, scallers[0])
                                    ),
                                    Some("BLACK"),
                                    None,
                                )
                            } else {
                                "".to_string()
                            }
                        }
                    );
                }
            }
        }
    }
}

fn format_fcn(profile: &Profile, func_name: &str) -> String {
    let (filename, line) = get_filename_line(profile, func_name).unwrap();
    format!("{}:{}:{}", filename, line, func_name)
}
fn get_filename_line(profile: &Profile, func_name: &str) -> Option<(String, usize)> {
    for sample in &profile.sample {
        for location in &sample.location_id {
            if let Some(loc) = profile.location.iter().find(|loc| &loc.id == location) {
                if let Some(line) = loc.line.get(0) {
                    if let Some(function) = profile
                        .function
                        .iter()
                        .find(|func| func.id == line.function_id)
                    {
                        if profile.string_table[function.name as usize] == func_name {
                            return Some((
                                profile.string_table[function.filename as usize].clone(),
                                line.line as usize,
                            ));
                        }
                    }
                }
            }
        }
    }
    None
}
// funmc_name, num_calls, tottime, cumtime
pub fn analyze_samples(profile: &Profile) -> HashMap<String, (i64, i64, i64)> {
    let (num_calls_idx, tottime_idx, cumtime_idx) = get_sample_type_indices(profile);
    let mut sample_info: HashMap<String, (i64, i64, i64)> = HashMap::new();

    for sample in &profile.sample {
        let num_calls = num_calls_idx.map_or(0, |idx| sample.value.get(idx).cloned().unwrap_or(0));
        let tottime = tottime_idx.map_or(0, |idx| sample.value.get(idx).cloned().unwrap_or(0));
        let cumtime = cumtime_idx.map_or(0, |idx| sample.value.get(idx).cloned().unwrap_or(0));

        for location in &sample.location_id {
            if let Some(loc) = profile.location.iter().find(|loc| &loc.id == location) {
                if let Some(line) = loc.line.get(0) {
                    if let Some(function) = profile
                        .function
                        .iter()
                        .find(|func| func.id == line.function_id)
                    {
                        let function_name = profile.string_table[function.name as usize].clone();
                        let entry = sample_info.entry(function_name).or_insert((0, 0, 0));
                        entry.0 = entry
                            .0
                            .checked_add(num_calls)
                            .expect("Overflow detected in num_calls");
                        entry.1 = entry
                            .1
                            .checked_add(tottime)
                            .expect("Overflow detected in tottime");
                        entry.2 = entry
                            .2
                            .checked_add(cumtime)
                            .expect("Overflow detected in cumtime");
                    }
                }
            }
        }
    }
    sample_info
}

// {    func name,  {   func_name,  count   }   }
pub fn extract_callers(profile: &Profile) -> HashMap<String, HashMap<String, i64>> {
    let mut callers: HashMap<String, HashMap<String, i64>> = HashMap::new();

    for sample in &profile.sample {
        for (i, location_id) in sample.location_id.iter().enumerate() {
            if let Some(location) = profile.location.iter().find(|loc| &loc.id == location_id) {
                if let Some(line) = location.line.get(0) {
                    if let Some(function) = profile
                        .function
                        .iter()
                        .find(|func| func.id == line.function_id)
                    {
                        let function_name = profile.string_table[function.name as usize].clone();

                        if i + 1 < sample.location_id.len() {
                            let caller_location_id = &sample.location_id[i + 1];
                            if let Some(caller_location) = profile
                                .location
                                .iter()
                                .find(|loc| &loc.id == caller_location_id)
                            {
                                if let Some(caller_line) = caller_location.line.get(0) {
                                    if let Some(caller_function) = profile
                                        .function
                                        .iter()
                                        .find(|func| func.id == caller_line.function_id)
                                    {
                                        let caller_function_name = profile.string_table
                                            [caller_function.name as usize]
                                            .clone();

                                        let function_entry = callers
                                            .entry(function_name.clone())
                                            .or_insert_with(HashMap::new);
                                        let count = function_entry
                                            .entry(caller_function_name.clone())
                                            .or_insert(0);
                                        *count += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    callers
}

// num_call_idx, tottime_idx, cumtime_idx
pub fn get_sample_type_indices(profile: &Profile) -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut num_calls_idx = None;
    let mut tottime_idx = None;
    let mut cumtime_idx = None;

    for (index, sample_type) in profile.sample_type.iter().enumerate() {
        let type_name = profile.string_table[sample_type.ty as usize].as_str();
        let unit_name = profile.string_table[sample_type.unit as usize].as_str();

        if type_name == "num_calls" && unit_name == "count" {
            num_calls_idx = Some(index);
        } else if type_name == "cpu" && unit_name == "nanoseconds" {
            tottime_idx = Some(index);
        } else if type_name == "cumulative_time" && unit_name == "nanoseconds" {
            cumtime_idx = Some(index);
        }
    }

    (num_calls_idx, tottime_idx, cumtime_idx)
}

lazy_static! {
    static ref PROFILE_LOGGER: Arc<Mutex<ProfileLogger>> =
        Arc::new(Mutex::new(ProfileLogger::build_ref()));
}
pub struct ProfileLogger {
    writers: usize,
    mjson: Vec<serde_json::Value>,
    actors: HashMap<String, isize>,
    subactors: HashMap<(String, String), isize>,
    path: String,
    events: Vec<(String, f64, f64, String, Option<String>)>,
}

impl ProfileLogger {
    fn build_ref() -> Self {
        Self {
            writers: 1,
            mjson: vec![],
            actors: HashMap::new(),
            subactors: HashMap::new(),
            path: getenv(
                "PROFILE_OUTPUT_FILE".to_string(),
                Some(temp("rustgrad_profile.json")),
            ),
            events: vec![],
        }
    }
    pub fn new() -> Arc<Mutex<Self>> {
        let mut p_l = PROFILE_LOGGER.lock().unwrap();
        p_l.events = vec![];
        p_l.writers += 1;

        PROFILE_LOGGER.clone()
    }
    pub fn add_event(
        &mut self,
        ev_name: String,
        ev_start: f64,
        ev_end: f64,
        actor: String,
        subactor: Option<String>,
    ) {
        self.events
            .push((ev_name, ev_start, ev_end, actor, subactor))
    }

    pub fn delete(&mut self) {
        for (name, st, et, actor_name, subactor_name) in self.events.iter() {
            if !self.actors.contains_key(actor_name) {
                let pid = self.actors.len();
                *self.actors.get_mut(actor_name).unwrap() = pid as isize;
                let payload = json!({
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "args": {
                        "name": actor_name
                    }
                });
                self.mjson.push(payload);
            }

            if let Some(subactors) = subactor_name {
                let sub_actors_key = (actor_name.clone(), subactors.clone());
                if !self.subactors.contains_key(&sub_actors_key) {
                    let tid = self.subactors.len();
                    self.subactors.insert(sub_actors_key, tid.clone() as isize);
                    let payload = json!({
                        "name": "thread_name",
                        "ph": "M",
                        "pid": self.actors.get(actor_name).unwrap(),
                        "tid": tid,
                        "args": {
                            "name": subactors
                        }
                    });
                    self.mjson.push(payload);
                }
            }

            let tid: isize = {
                if let Some(subactor) = subactor_name {
                    self.subactors
                        .get(&(actor_name.clone(), subactor.clone()))
                        .map(|x| x.clone())
                        .unwrap_or(-1)
                } else {
                    -1
                }
            };
            let payload = json!({
                "name": name,
                "ph": "X",
                "pid": self.actors.get(actor_name).unwrap(),
                "tid": tid,
                "ts": st,
                "dur": et-st
            });
            self.mjson.push(payload);
        }

        self.writers -= 1;

        if self.writers == 0 {
            let mut file = File::create(&self.path).unwrap();
            let json_string = json!({"traceEvents": self.mjson});
            file.write_all(to_string(&json_string).unwrap().as_bytes())
                .unwrap();
            print!(
                "Saved profile to {}. Use Use https://ui.perfetto.dev/ to open it.",
                self.path
            )
        }
    }
}

lazy_static! {
    pub static ref _CACHE_DIR: String = getenv("XDG_CACHE_HOME".to_string(), {
        if *OSX {
            let mut path = dirs::home_dir().unwrap_or_default();
            path.push("Library");
            path.push("Caches");
            Some(path.to_str().unwrap().to_owned())
        } else {
            let mut path = dirs::home_dir().unwrap_or_default();
            path.push(".cache");
            Some(path.to_str().unwrap().to_owned())
        }
    });
    pub static ref CACHEDB: String = getenv("CACHEDB".to_string(), {
        let mut path = PathBuf::from(_CACHE_DIR.clone());

        path.push("rustgrad");
        path.push("cache.db");

        let abs_path = path.canonicalize().unwrap_or(path);

        let absolute_path_str = abs_path.to_str().unwrap_or_else(|| {
            panic!("Failed to convert path {:?} to string", abs_path);
        });
        Some(absolute_path_str.to_string())
    });
    pub static ref CACHELEVEL: usize = getenv("CACHELEVEL".to_owned(), Some("2".to_owned()))
        .parse()
        .unwrap();
    pub static ref VERSION: usize = 16;
    static ref DB_CONNECTION: Arc<Mutex<Option<Connection>>> = Arc::new(Mutex::new(None));
}
pub fn db_connection() -> Arc<Mutex<Option<Connection>>> {
    let conn_ref_result = DB_CONNECTION.lock();
    let mut conn_ref: MutexGuard<Option<Connection>> = match conn_ref_result {
        Result::Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };

    if let None = *conn_ref {
        let parent_dir = match CACHEDB.rsplit_once(std::path::MAIN_SEPARATOR) {
            Some((dir, _)) => dir,
            None => {
                panic!("Invalid path: {}", CACHEDB.clone());
            }
        };

        if let Err(err) = fs::create_dir_all(parent_dir) {
            panic!("Failed to create directory: {}", err);
        }

        let new_connection = conn_ref.insert(Connection::open(CACHEDB.clone()).unwrap());
        if DEBUG.clone() >= 7 {
            new_connection.trace(Some(|s| print!("{}", s)));
        }
    }
    return DB_CONNECTION.clone();
}

pub fn diskcache_clear() -> Result<(), anyhow::Error> {
    let conn = db_connection();

    let mut cur_ = conn.lock().map_err(|_| anyhow!("Mutex poisioned"))?;

    let drop_tables_query = "SELECT 'DROP TABLE IF EXISTS ' || quote(name) || ';' FROM sqlite_master WHERE type = 'table';";
    let cur = cur_.as_mut().expect("Connection not inited");
    let drop_tables: Vec<String> = cur
        .prepare(drop_tables_query)?
        .query_map([], |row| row.get(0))?
        .filter_map(Result::ok)
        .collect();

    // Execute the drop statements in a transaction
    let drop_script = drop_tables.join("\n");
    cur.execute_batch(&drop_script)?;

    Ok(())
}

pub fn diskcache_get<'a, T>(table: &str, key: Box<dyn Any>) -> Option<T>
where
    T: serde::Deserialize<'a>,
{
    if CACHELEVEL.clone() == 0 {
        return None;
    }

    let mut key_downcasted: Option<HashMap<String, String>> = None;

    if let Some(str_k) = key.downcast_ref::<String>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), str_k.clone());
        key_downcasted = Some(hm);
    } else if let Some(int_k) = key.downcast_ref::<isize>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), int_k.to_string());
        key_downcasted = Some(hm);
    } else if let Some(u8_k) = key.downcast_ref::<u8>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), u8_k.to_string());
        key_downcasted = Some(hm);
    } else if let Some(usize_k) = key.downcast_ref::<usize>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), usize_k.to_string());
        key_downcasted = Some(hm);
    } else if let Some(f64_k) = key.downcast_ref::<f64>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), f64_k.to_string());
        key_downcasted = Some(hm);
    } else if let Some(str_ref_k) = key.downcast_ref::<&str>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), str_ref_k.to_string());
        key_downcasted = Some(hm);
    } else if let Some(hash_map) = key.downcast_ref::<HashMap<String, String>>() {
        key_downcasted = Some(hash_map.clone());
    }
    let bind_result = db_connection();

    let conn_ref_result = bind_result.lock();
    let conn_ref = match conn_ref_result {
        Result::Ok(guard) => guard,
        Err(poisoned) => PoisonError::into_inner(poisoned),
    };

    let conn = conn_ref
        .as_ref()
        .expect("Failed to get the database connection.");

    let put_query = format!(
        "SELECT val FROM '{}_{}' WHERE {}",
        table,
        VERSION.clone(),
        key_downcasted
            .as_ref()
            .expect("")
            .into_iter()
            .map(|(k, _)| { format!("{}=?", k) })
            .collect::<Vec<String>>()
            .join(" AND ")
    );

    let mut stmt = conn.prepare(&put_query).ok()?;

    let query_params = params_from_iter(
        key_downcasted
            .as_ref()
            .expect("")
            .into_iter()
            .map(|(_, v)| v),
    );

    // Execute the query
    let mut rows = stmt.query(query_params).ok()?;

    let mut buffer = Cursor::new(Vec::new());
    // Iterate over the results
    while let Some(row) = rows.next().expect("Failed to fetch row") {
        let val: String = row.get(0).expect("Failed to get column");
        serde_pickle::to_writer(&mut buffer, &val, SerOptions::default()).unwrap();
    }

    Some(serde_pickle::from_reader(buffer, DeOptions::default()).unwrap())
}

lazy_static! {
    static ref DB_TABLES: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
}

pub fn diskcache_put<'a, T>(table: &str, key: Box<dyn Any>, val: T) -> Result<T, anyhow::Error>
where
    T: serde::Deserialize<'a> + Serialize,
{
    if CACHELEVEL.clone() == 0 {
        return Ok(val);
    }

    let mut key_downcasted: Option<HashMap<String, String>> = None;
    let mut key_type = None;
    if let Some(str_k) = key.downcast_ref::<String>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), str_k.clone());
        key_downcasted = Some(hm);
        key_type = Some("str".to_string());
    } else if let Some(int_k) = key.downcast_ref::<isize>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), int_k.to_string());
        key_downcasted = Some(hm);
        key_type = Some("int".to_owned());
    } else if let Some(u8_k) = key.downcast_ref::<u8>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), u8_k.to_string());
        key_downcasted = Some(hm);
        key_type = Some("int".to_string());
    } else if let Some(usize_k) = key.downcast_ref::<usize>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), usize_k.to_string());
        key_downcasted = Some(hm);
        key_type = Some("int".to_string());
    } else if let Some(f64_k) = key.downcast_ref::<f64>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), f64_k.to_string());
        key_downcasted = Some(hm);
        key_type = Some("float".to_string());
    } else if let Some(str_ref_k) = key.downcast_ref::<&str>() {
        let mut hm = HashMap::new();
        hm.insert("key".to_string(), str_ref_k.to_string());
        key_downcasted = Some(hm);
        key_type = Some("str".to_string());
    } else if let Some(hash_map) = key.downcast_ref::<HashMap<String, String>>() {
        key_downcasted = Some(hash_map.clone());
    }
    let bind = db_connection();
    let conn_ = bind.lock().unwrap();
    let conn = conn_.as_ref().expect("");
    if DB_TABLES.clone().lock().unwrap().contains(table) {
        let types = {
            let mut hm = HashMap::new();
            hm.insert("str".to_owned(), "text".to_owned());
            hm.insert("bool".to_owned(), "integer".to_owned());
            hm.insert("int".to_owned(), "integer".to_owned());
            hm.insert("float".to_owned(), "numeric".to_owned());
            hm.insert("bytes".to_owned(), "blob".to_owned());
            hm
        };

        let ltypes = key_downcasted
            .as_ref()
            .expect("unreachable")
            .iter()
            .map(|(k, _)| format!("{} {}", k, types.get(key_type.as_ref().expect("")).unwrap()))
            .collect::<Vec<String>>()
            .join(", ");

        let put_query = format!(
            "CREATE TABLE IF NOT EXISTS '{}_{}' ({}, val blob, PRIMARY KEY ({}))",
            table,
            VERSION.clone(),
            ltypes,
            key_downcasted
                .as_ref()
                .expect("")
                .iter()
                .map(|(k, _)| k.clone())
                .collect::<Vec<String>>()
                .join(", ")
        );

        let query_params = params![];

        let mut stmt = conn.prepare(&put_query).expect("failed to prepare query");

        // Execute the query
        let _rows = stmt.query(query_params).expect("Failed to execute query");
        DB_TABLES.lock().unwrap().insert(table.to_string());
    }
    let put_query = format!(
        "REPLACE INTO '{}_{}' ({}, val) VALUES ({}, ?)",
        table,
        VERSION.clone(),
        key_downcasted
            .as_ref()
            .expect("")
            .iter()
            .map(|(k, _)| k.clone())
            .collect::<Vec<String>>()
            .join(", "),
        vec!["?"; key_downcasted.as_ref().expect("").keys().len()].join(", ")
    );
    //let mut buffer = Cursor::new(Vec::new());
    let v = serde_pickle::to_value(&val).unwrap();
    let it_v = v.to_string();
    //let by = buffer.into_inner();
    // Create query parameters by chaining the values of `key_downcasted` with the serialized value
    let query_params = params_from_iter(
        key_downcasted
            .as_ref()
            .expect("")
            .values()
            .chain(std::iter::once(&it_v)),
    );

    let mut stmt = conn
        .prepare(&put_query)
        .map_err(|e| anyhow!(format!("failed to prepare query {}", e)))?;

    // Execute the query
    let _ = stmt
        .query(query_params)
        .map_err(|e| anyhow!(format!("failed to prepare query {}", e)))?;
    return Ok(val);
}

pub fn parse_fetch_paths(url: &str, name: Option<String>, subdir: Option<String>) -> PathBuf {
    if let Some(n) = &name {
        if n.contains('/') {
            return PathBuf::from(n);
        }
    }
    PathBuf::from(_CACHE_DIR.clone())
        .join("rustgrad")
        .join("downloads")
        .join(subdir.unwrap_or("".to_string()))
        .join(name.unwrap_or(format!("{:x}", md5::compute(url))))
}
pub async fn fetch(
    url: &str,
    fp: PathBuf,
    allow_cache: Option<bool>,
) -> Result<PathBuf, anyhow::Error> {
    if url.starts_with("/") || url.starts_with(".") {
        return PathBuf::from_str(url)
            .map_err(|e| anyhow!("couldnt parse str into path buf {}", e));
    }

    if !fp.exists()
        || allow_cache.unwrap_or(
            !getenv("DISABLE_HTTP_CACHE".to_string(), None)
                .parse::<isize>()
                .unwrap()
                == 1,
        )
    {
        let response = get(url).await?;

        if response.status() != 200 {
            return Err(anyhow!(
                "HTTP request failed with status: {}",
                response.status()
            ));
        }

        let total_length = response
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|len| len.to_str().ok())
            .and_then(|len| len.parse().ok())
            .unwrap_or(0);

        let progress_bar = if CI.clone() {
            ProgressBar::hidden()
        } else {
            let pb = ProgressBar::new(total_length);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                    .progress_chars("##-"),
            );
            pb.set_message(format!("{}: ", url));
            pb
        };

        let path = fp.parent().unwrap();
        fs::create_dir_all(path)?;

        let mut tempfile = NamedTempFile::new_in(path)?;
        let mut downloaded = 0;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            tempfile.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            progress_bar.set_position(downloaded);
        }
        progress_bar.finish();

        let file_size = tempfile.as_file().metadata()?.len();
        if file_size < total_length {
            return Err(anyhow!(
                "fetch size incomplete, {} < {}",
                file_size,
                total_length
            ));
        }
        tempfile.persist(fp.clone())?;
    }
    return Ok(fp);
}

pub fn cpu_time_execution<T>(cb: T, enable: bool) -> Option<Duration>
where
    T: Fn(),
{
    let st = {
        if enable {
            Some(Instant::now())
        } else {
            None
        }
    };

    cb();

    if enable {
        return Some(Instant::now() - st.unwrap());
    } else {
        return None;
    }
}

pub fn cpu_objdump(lib: &[u8]) -> Result<(), anyhow::Error> {
    let mut tempfile =
        tempfile::NamedTempFile::new().map_err(|e| anyhow!("failed to create a tempfile {}", e))?;
    tempfile
        .write_all(lib)
        .map_err(|e| anyhow!("failed to write bytes {}", e))?;
    let command_out = Command::new("objdump")
        .arg("-d")
        .arg(tempfile.path().to_str().unwrap())
        .output()?;

    if command_out.status.success() {
        let output = String::from_utf8_lossy(&command_out.stdout);
        print!("{}", output);
    } else {
        let error_str = String::from_utf8_lossy(&command_out.stderr);
        eprintln!("Error: {}", error_str);
    }

    Ok(())
}

pub fn from_mv<T>(mv: &[u8]) -> *const T {
    let ptr = mv.as_ptr() as *const c_void;
    ptr as *const T
}

// Function to convert a pointer to a memory view (like Python's memoryview)
pub fn to_mv(ptr: *const c_void, sz: usize) -> Option<&'static [u8]> {
    // Safety: Creating a slice from a raw pointer requires unsafe code
    unsafe {
        if ptr.is_null() {
            return None; // Handle null pointers gracefully
        }

        // Cast the pointer to a pointer to an array of c_uint8 with size `sz`
        let ptr_arr = ptr as *const [u8; 0]; // Using [u8; 0] to match the size dynamically
        let slice = std::slice::from_raw_parts(ptr_arr as *const u8, sz);

        Some(slice)
    }
}

pub fn mv_address(mv: &[u8]) -> *const c_void {
    mv.as_ptr() as *const c_void
}

pub fn to_char_p_p<T>(options: &[&[u8]]) -> Vec<*const T> {
    let mut pointers: Vec<*const T> = Vec::with_capacity(options.len());

    for option in options {
        let c_string = CString::new(*option).expect("Failed to create CString");
        pointers.push(c_string.as_ptr() as *const T);

        // Note: Ensure the CStrings are not dropped until after the pointers are used.
        // For simplicity, we'll use a vector to store them and prevent them from being dropped.
    }

    pointers
}

pub fn flat_mv(mv: &[u8]) -> &[u8] {
    if mv.is_empty() {
        mv
    } else {
        // Create a new slice with the same byte size
        let len = mv.len();
        // Safety: We are creating a slice with the same underlying buffer
        unsafe { std::slice::from_raw_parts(mv.as_ptr(), len) }
    }
}

