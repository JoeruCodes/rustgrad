use cached::proc_macro::cached;
use colored::*;
use dirs::home_dir;
use erased_serde::Serialize;
use itertools::Itertools;
use lazy_static::lazy_static;
use libloading::{Library, Symbol};
use lru::LruCache;
use num::integer;
use num_traits::{Num, One};
use regex::Regex;
use rusqlite::types::FromSql;
use rusqlite::{params, Connection, ToSql};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::any::Any;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::env::temp_dir;
use std::error::Error;
use std::fmt::Debug;
use std::fmt::{self};
use std::fs;
use std::fs::File;
use std::hash::Hash;
use std::hash::Hasher;
use std::io::{self, Write};
use std::iter::Flatten;
use std::iter::Product;
use std::num::NonZeroUsize;
use std::ops::{Add, Div, Mul, Sub};
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::time::Instant;
use std::usize::MAX;
use std::{env, result, vec};

use super::shape::sym::BTypes;
lazy_static! {
    static ref OSX: Arc<bool> = Arc::new(cfg!(target_os = "macos"));
    static ref CI: Arc<bool> = Arc::new(env::var("CI").is_ok());
    //adjust the CACHE_LRU size dynamically
    pub static ref CACHE_LRU: Arc<Mutex<LruCache<Arc<String>, Arc<String>>>> = Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(MAX).unwrap())));

    pub static ref DEBUG: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("DEBUG", 0)));
    pub static ref IMAGE: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("IMAGE", 0)));
    pub static ref WINO: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("WINO", 0)));
    pub static ref BEAM: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("BEAM", 0)));
    pub static ref NOOPT: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("NOOPT", 0)));
    pub static ref GRAPH: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("GRAPH", 0)));
    pub static ref MERGE_VIEW: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("MERGE_VIEW", 0)));
    pub static ref REPR: Arc<Mutex<ContextVar>> = Arc::new(Mutex::new(ContextVar::new("REPR", 0)));
    pub static ref GRAPHPATH: Arc<String> =  Arc::new(getenv("GRAPHPATH".to_string(), "/temp/net".to_string()));

    pub static ref CACHE_DIR: Arc<String> = Arc::new(getenv("XDG_CACHE_HOME".to_string(), home_dir().map_or("".to_string(), |home| home.join(if *Arc::clone(&OSX) { "Library/Caches" } else { ".cache" }).to_string_lossy().into_owned())));
    pub static ref CACHEDB: Arc<String> = Arc::new(getenv("CACHEDB".to_string(), Path::new(Arc::clone(&CACHE_DIR).as_ref()).join("rustgrad").join("cache_db").canonicalize().expect("Failed to get abs path").to_string_lossy().to_owned().to_string()));
    pub static ref CACHELEVEL: Arc<String> = Arc::new(getenv("CACHELEVEL".to_string(), 2.to_string()));


    pub static ref VERSION:Arc<usize> = Arc::new(1);
    pub static ref DB_CONNECTION: Arc<Mutex<Option<Connection>>> = Arc::new(Mutex::new(None));
    static ref _DB_TABLES: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
}

pub fn prod<T>(x: impl IntoIterator<Item = T>) -> T
where
    T: Mul<Output = T> + Product + One,
{
    x.into_iter().fold(T::one(), |acc, val| acc * val)
}

pub fn dedup<T>(x: Vec<T>) -> Vec<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    let mut set = HashSet::with_capacity(x.len());
    let mut result = Vec::with_capacity(x.len());

    for item in x.into_iter() {
        if set.insert(item.clone()) {
            result.push(item);
        }
    }

    result
}

// pub fn argfix<T>(x: Option<&Vec<T,>>) -> Option<Vec<T>>
// where
//     T: Clone
// {
//     match x {
//         Some(values) if values.is_empty() => None,
//         Some(values) => Some(values.to_vec()),
//         None => None,
//     }
// }

// pub fn argfix<T, C>(x:(T, C)) -> Vec<_>
// where
//     C: AsRef<[T]>,
// {
//     let mut temp = Vec::new();
//     temp.push(x);
//     temp
// }

pub fn argsort<T: Ord>(seq: &[T]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..seq.len()).collect();
    indices.sort_by_key(|&i| &seq[i]);
    indices.into_iter().map(|x| x).collect_vec()
}

pub fn all_same<T: PartialEq>(items: &Vec<T>) -> bool {
    if let Some(first) = items.first() {
        items.iter().all(|x| x == first)
    } else {
        true
    }
}

pub fn all_int<T>(t: &[T]) -> bool {
    t.iter().all(|_| true)
}

pub fn colored(st: &str, color: Option<&str>, background: bool) -> String {
    if let Some(color_str) = color {
        let color_enum = match color_str.to_lowercase().as_str() {
            "black" => Color::Black,
            "red" => Color::Red,
            "green" => Color::Green,
            "yellow" => Color::Yellow,
            "blue" => Color::Blue,
            "magenta" => Color::Magenta,
            "cyan" => Color::Cyan,
            "white" => Color::White,
            _ => Color::White, // Default to white if the color is not recognized
        };

        let colored_st = if background {
            st.on_color(color_enum)
        } else {
            st.color(color_enum)
        };

        return colored_st.to_string();
    }

    st.to_string()
}

pub fn ansistrip(s: &str) -> String {
    let re = Regex::new(r"\x1b\[([Km]|.*?m)").unwrap();
    re.replace_all(s, "").to_string()
}

pub fn ansilen(s: &str) -> usize {
    ansistrip(s).len()
}

pub fn make_pair(x: isize, cnt: usize) -> Vec<isize> {
    vec![x; cnt]
}

pub(crate) trait DeepFlattenIteratorOf<Depth, T> {
    type DeepFlattenStruct: Iterator<Item = T>;
    fn deep_flatten(this: Self) -> Self::DeepFlattenStruct;
}

impl<I: Iterator> DeepFlattenIteratorOf<(), I::Item> for I {
    type DeepFlattenStruct = Self;
    fn deep_flatten(this: Self) -> Self::DeepFlattenStruct {
        this
    }
}

impl<Depth, I: Iterator, T> DeepFlattenIteratorOf<(Depth,), T> for I
where
    Flatten<I>: DeepFlattenIteratorOf<Depth, T>,
    I: Iterator,
    <I as Iterator>::Item: IntoIterator,
{
    type DeepFlattenStruct = <Flatten<I> as DeepFlattenIteratorOf<Depth, T>>::DeepFlattenStruct;
    fn deep_flatten(this: Self) -> Self::DeepFlattenStruct {
        DeepFlattenIteratorOf::deep_flatten(this.flatten())
    }
}

// wrapper type to help out type inference
pub(crate) struct DeepFlattenStruct<Depth, I, T>
where
    I: DeepFlattenIteratorOf<Depth, T>,
{
    inner: I::DeepFlattenStruct,
}

pub trait DeepFlatten: Iterator + Sized {
    fn deep_flatten<Depth, T>(self) -> DeepFlattenStruct<Depth, Self, T>
    where
        Self: DeepFlattenIteratorOf<Depth, T>,
    {
        DeepFlattenStruct {
            inner: DeepFlattenIteratorOf::deep_flatten(self),
        }
    }
}
impl<I: Iterator> DeepFlatten for I {}
impl<Depth, I, T> Iterator for DeepFlattenStruct<Depth, I, T>
where
    I: DeepFlattenIteratorOf<Depth, T>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

pub fn flatten<T, I>(l: impl IntoIterator<Item = I>) -> Vec<T>
where
    I: IntoIterator<Item = T>,
{
    l.into_iter().flat_map(|item| item.into_iter()).collect()
}

// pub fn flatten<T, F>(l: Vec<Vec<T>>) -> Vec<T>{
//     l.into_iter().flatten().collect()
// }

pub fn from_import(mod_name: &str, frm_name: &str) -> Result<extern "C" fn(), String> {
    let lib = Library::new(mod_name).map_err(|e| format!("Failed to load module: {:?}", e))?;

    unsafe {
        let symbol: Result<Symbol<extern "C" fn()>, _> = lib.get(frm_name.as_bytes());
        match symbol {
            Ok(func) => Ok(*func.into_raw()),
            Err(_) => Err(format!("Failed to find function in module: {}", frm_name)),
        }
    }
}

pub fn strip_parens(fst: &str) -> &str {
    if fst.starts_with('(') && fst.ends_with(')') {
        let inner_str = &fst[1..fst.len() - 1];
        if inner_str.find('(') <= inner_str.find(')') {
            return inner_str;
        }
    }
    fst
}

pub fn round_up(num: BTypes, amt: isize) -> BTypes{
    let b_amt = BTypes::Int(amt);
    (&(&num + &b_amt) - &BTypes::Int(1)).floordiv(&b_amt, true)
}

pub fn merge_maps<T, U>(maps: impl IntoIterator<Item = HashMap<T, U>>) -> HashMap<T, U>
where
    T: Eq + std::hash::Hash + Clone,
{
    let mut merged_map = HashMap::new();
    let mut seen_keys = HashSet::new();

    for map in maps {
        for (key, value) in map.into_iter() {
            // Check if the key is already seen
            if !seen_keys.insert(key.clone()) {
                panic!("Cannot merge, multiple values for the same key");
            }

            // Insert the key-value pair into the merged map
            merged_map.insert(key, value);
        }
    }

    merged_map
}

pub fn partition<T, F>(lst: Vec<T>, fxn: F) -> (Vec<T>, Vec<T>)
where
    T: Sized,
    F: Fn(&T) -> bool,
{
    let (a, b): (Vec<T>, Vec<T>) = lst.into_iter().partition(|s| fxn(s));
    (a, b)
}

fn get_child<'a>(
    obj: &'a Box<dyn std::any::Any>,
    key: &'a str,
) -> Option<&'a Box<dyn std::any::Any>> {
    let mut current_obj = obj;

    for k in key.split('.') {
        if k.parse::<usize>().is_ok() {
            if let Some(inner_obj) =
                current_obj.downcast_ref::<HashMap<String, Box<dyn std::any::Any>>>()
            {
                if let Some(inner_box) = inner_obj.get(&k.to_string()) {
                    current_obj = inner_box;
                } else {
                    return None;
                }
            } else {
                return None;
            }
        } else if let Some(inner_obj) =
            current_obj.downcast_ref::<HashMap<String, Box<dyn std::any::Any>>>()
        {
            if let Some(inner_box) = inner_obj.get(k) {
                current_obj = inner_box;
            } else {
                return None;
            }
        } else {
            return None;
        }
    }

    Some(current_obj)
}

pub fn get_contraction(old_shape: &[BTypes], new_shape: &[BTypes]) -> Option<Vec<Vec<usize>>> {
    let acc_old: Vec<BTypes> = old_shape
        .iter()
        .scan(BTypes::Int(1), |acc, x| {
            *acc = &*acc * x;
            Some(*acc)
        })
        .collect();
    let acc_new: Vec<BTypes> = new_shape
        .iter()
        .scan(BTypes::Int(1), |acc, x| {
            *acc = &*acc * x;
            Some(*acc)
        })
        .collect();

    let split: Vec<usize> = acc_new
        .iter()
        .map(|&acc| {
            acc_old
                .iter()
                .position(|&x| x == acc )
                .unwrap_or(0)
                + 1
        })
        .collect();

    let contraction: Vec<Vec<usize>> = (0..split.len())
        .map(|i| (if i == 0 { 0 } else { split[i - 1] })..split[i])
        .map(|range| range.map(|idx| idx).collect())
        .collect();

    Some(contraction)
}

#[cached]
pub fn to_function_name(s: String) -> String {
    // let cache_clone = Arc::clone(&CACHE_LRU);
    // let mut cache = cache_clone.lock().unwrap();

    // if let Some(result) = cache.get(&Arc::new(s.to_string())) {
    //     return Arc::clone(result).as_str().to_string();
    // }

    let result: String = ansistrip(&s)
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c.to_string()
            } else {
                format!("{:02X}", c as u32)
            }
        })
        .collect();

    // cache.put(Arc::new(s.to_string()), Arc::new(result.clone()));

    result
}

#[cached]
pub fn getenv(k: String, default: String) -> String {
    // let cache_clone = Arc::clone(&CACHE_LRU);
    // let mut cache = cache_clone.lock().unwrap();

    // if let Some(result) = cache.get(&Arc::new(key.to_string())) {
    //     return Ok(Arc::clone(result).as_str().to_string());
    // }

    match env::var(k) {
        Ok(value) => {
            let result = value.parse().unwrap_or_else(|_| default.clone());
            // cache.put(Arc::new(key.to_string()), Arc::new(result.clone()));
            result
        }
        Err(_) => {
            // cache.put(Arc::new(key.to_string()), Arc::new(default.clone()));
            default
        }
    }
}

pub fn temp(x: String) -> String {
    let temp_dir = temp_dir();
    let temp_path = Path::new(&temp_dir).join(x);
    temp_path.to_string_lossy().into_owned()
}

//xxxxxxxxxxxxxxxx TO BE DEPRECATED AND REPLACED BY panic! OR Result enum xxxxxxxxxxxxxxxxxxxx
#[derive(Debug)]
pub struct GraphError {
    message: Option<String>,
}

impl GraphError {
    fn new(message: Option<String>) -> Self {
        GraphError { message }
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref message) = self.message {
            write!(f, "Graph error: {}", message)
        } else {
            write!(f, "Graph error")
        }
    }
}
impl Error for GraphError {}

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

pub struct Context {
    pub stack: Arc<RefCell<Vec<HashMap<String, isize>>>>,
}

impl Context {
    fn new() -> Self {
        Context {
            stack: Arc::new(RefCell::new(vec![HashMap::new()])),
        }
    }

    fn enter(&self, kwargs: HashMap<String, isize>) {
        let mut stack = self.stack.borrow_mut();
        stack.last_mut().unwrap().extend(kwargs.clone());

        for (k, v) in kwargs.iter() {
            stack.last_mut().unwrap().insert(k.clone(), *v);
        }

        stack.push(kwargs);
    }

    fn exit(&self) {
        let mut stack = self.stack.borrow_mut();
        if let Some(undo_state) = stack.pop() {
            for (k, v) in undo_state.iter() {
                stack.last_mut().unwrap().insert(k.clone(), *v);
            }
        }
    }
}

//automatic exit when dropped can be overriden
impl<'a> Drop for Context {
    fn drop(&mut self) {
        let mut stack = self.stack.borrow_mut();
        if let Some(undo_state) = stack.pop() {
            for (k, v) in undo_state.iter() {
                stack.last_mut().unwrap().insert(k.clone(), *v);
            }
        }
    }
}
//xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Context Usage xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// let context = Context::new();

// // Example of using the context
// {
//     let mut stack = context.stack.lock().unwrap();
//     stack.last_mut().unwrap().insert(String::from("example"), 42);
// }

// context.enter(HashMap::new()); // Enter the context with empty temporary state
// context.enter(HashMap::from([("key1", 10), ("key2", 20)])); // Enter the context with new temporary state

// // ... Do some work with the modified state ...

// context.exit();

#[derive(Debug)]
pub struct ContextVar {
    key: Arc<String>,
    pub value: isize,
}

impl ContextVar {
    pub fn new(key: &str, default_value: isize) -> ContextVar {
        let cache_clone = Arc::clone(&CACHE_LRU);
        let mut cache = cache_clone.lock().unwrap();

        if let Some(result) = cache.get(&Arc::new(key.to_string())) {
            return ContextVar {
                key: Arc::new(key.to_string()),
                value: result.parse().unwrap_or(default_value),
            };
        }

        let result: isize =  getenv(key.to_string(), default_value.to_string()).parse().unwrap();

        cache.put(Arc::new(key.to_string()), Arc::new(result.to_string()));

        ContextVar {
            key: Arc::new(key.to_string()),
            value: result,
        }
    }
}

impl std::cmp::PartialEq for ContextVar {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl std::cmp::PartialOrd for ContextVar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.value.cmp(&other.value))
    }
}

pub struct Timing<'a> {
    prefix: &'a str,
    on_exit: Option<Box<dyn Fn(u128) -> String>>,
    enabled: bool,
    st: Option<Instant>,
}

impl<'a> Timing<'a> {
    pub fn new(
        prefix: &'a str,
        on_exit: Option<Box<dyn Fn(u128) -> String>>,
        enabled: bool,
    ) -> Self {
        Timing {
            prefix,
            on_exit,
            enabled,
            st: None,
        }
    }

    pub fn enter(&mut self) {
        self.st = Some(Instant::now());
    }

    pub fn exit(&mut self) {
        if let Some(start_time) = self.st {
            let elapsed_time = start_time.elapsed().as_nanos();
            if self.enabled {
                let message = format!("{}{:.2} ms", self.prefix, elapsed_time as f64 * 1e-6);
                if let Some(ref on_exit_fn) = self.on_exit {
                    print!("{}{}", message, on_exit_fn(elapsed_time));
                } else {
                    print!("{}", message);
                }
            }
        }
    }
}

// automatic when dropped can be overridden later
impl<'a> Drop for Timing<'a> {
    fn drop(&mut self) {
        if let Some(start_time) = self.st {
            let elapsed_time = start_time.elapsed().as_nanos();
            if self.enabled {
                let message = format!("{}{:.2} ms", self.prefix, elapsed_time as f64 * 1e-6);
                if let Some(ref on_exit_fn) = self.on_exit {
                    print!("{}{}", message, on_exit_fn(elapsed_time));
                } else {
                    print!("{}", message);
                }
            }
        }
    }
}

fn format_fcn(fcn: (u32, u32, u32)) -> String {
    format!("{}:{}:{}", fcn.0, fcn.1, fcn.2)
}

pub fn db_connection() -> Result<String, String> {
    {
        if Arc::clone(&DB_CONNECTION)
            .lock()
            .unwrap()
            .as_ref()
            .is_some()
        {
            return Err("DB already running".to_string());
        }
    }

    let cache_db = Arc::clone(&CACHEDB);
    fs::create_dir_all(Path::new(cache_db.as_str()).parent().unwrap()).ok();
    let mut connection = Connection::open(cache_db.as_str()).expect("Failed to connect to db");

    if Arc::clone(&DEBUG)
        .lock()
        .unwrap()
        .key
        .parse::<i32>()
        .unwrap()
        >= 7
    {
        connection.trace(Some(|i: &str| print!("{}", i)));
    }

    // let db = Arc::clone(&DB_CONNECTION);
    // let mut db_guard = db.lock().unwrap();
    // *db_guard = Some(connection);

    *(Arc::clone(&DB_CONNECTION).lock().unwrap()) = Some(connection);

    Ok("DB started".to_string())
}

// pub fn discache_get<T>(tabel: String, key: HashMap<String, String>, val: T) -> Option<serde_pickle>{
//     if Arc::clone(&CACHELEVEL).parse::<isize>().unwrap() == 0{
//         return None;
//     }
// }

// pub fn diskcache_get<T>(table: &str, key: Map<String, Value>) -> Option<Vec<u8>>{
//     if Arc::clone(&CACHELEVEL).parse::<i32>().unwrap() == 0{
//         return Option::None;
//     }

//     let connStatus = Arc::clone(&DB_CONNECTION);
//     let result = match connStatus.lock().unwrap()
//                                         .as_ref()?
//                                         .execute(
//                                             format!(
//                                                 "SELECT val FROM {}_{} WHERE {}",
//                                                 table,
//                                                 Arc::clone(&VERSION),
//                                                 key.keys()
//                                                     .map(|k| format!("{}=?", k))
//                                                     .collect::<Vec<String>>()
//                                                     .join(" AND ")).as_str(),
//                                                     params![
//                                                         key.values()
//                                                             .map(
//                                                                 |v| serde_json::to_vec(&v.to_string()).unwrap())
//                                                                     .collect::<Vec<Vec<u8>>>()
//                                                                     .concat()]){
//                                                                         Ok(r) => return
//                                                                     };

//     return None;
// }

// pub fn diskcache_get(table: &str, key: Map<String, Value>) -> Option<Vec<u8>> {
//     if Arc::clone(&CACHELEVEL).parse::<i32>().unwrap() == 0 {
//         return None;
//     }
//     let mut temp = vec![];
//     let conn_status = Arc::clone(&DB_CONNECTION);
//     let mut conn = conn_status.lock().unwrap();
//     let mut query = conn
//         .as_mut()?
//         .prepare(
//             &format!(
//                 "SELECT val FROM {}_{} WHERE {}",
//                 table,
//                 Arc::clone(&VERSION),
//                 key.keys()
//                     .map(|k| format!("{}=?", k))
//                     .collect::<Vec<String>>()
//                     .join(" AND ")
//             ),
//         )
//         .unwrap();

//         let _ = query.query(params!(
//             key.values()
//                 .map(|v| serde_json::to_vec(&v.to_string()).unwrap())
//                 .collect::<Vec<Vec<u8>>>()
//                 .concat()
//         ))
//         .unwrap().map(|r| Ok(temp.push(r.get::<usize, Vec<u8>>(0)?)));

//     temp.into_iter().map(|v| serde_pickle::from_slice(&v, Default::default()).unwrap()).collect()

// }

// overly complicated... TODO: simplify...
pub fn diskcache_get(table: &str, key: Map<String, Value>) -> Option<Vec<u8>> {
    if Arc::clone(&CACHELEVEL).parse::<i32>().unwrap() == 0 {
        return None;
    }

    let mut temp = Vec::new();
    let conn_status = Arc::clone(&DB_CONNECTION);
    let mut conn = conn_status.lock().unwrap();

    let mut query = conn
        .as_mut()?
        .prepare(&format!(
            "SELECT val FROM {}_{} WHERE {}",
            table,
            Arc::clone(&VERSION),
            key.keys()
                .map(|k| format!("{}=?1", k))
                .collect::<Vec<String>>()
                .join(" AND ")
        ))
        .unwrap();

    let _ = query
        .query(params!(key
            .values()
            .map(|v| String::from_utf8_lossy(
                serde_json::to_vec(&v.to_string()).unwrap().as_slice()
            )
            .to_string())
            .collect::<Vec<String>>()
            .concat()))
        .unwrap()
        .map(|r| Ok(temp.push(r.get::<usize, Vec<u8>>(0)?)));

    temp.into_iter()
        .map(|v| serde_pickle::from_slice(&v, Default::default()).unwrap())
        .collect()
}

pub fn diskcache_put(table: &str, key: Map<String, Value>, val: Value) -> Value {
    if Arc::clone(&CACHELEVEL).parse::<isize>().unwrap() == 0 {
        return val;
    }

    let conn_status = Arc::clone(&DB_CONNECTION);
    let mut conn = conn_status.lock().unwrap();

    let tables = Arc::clone(&_DB_TABLES);
    let mut t = tables.lock().unwrap();

    if t.contains(&table.to_string()) {
        let _ = conn
            .as_mut()
            .unwrap()
            .prepare(
                format!(
                    "CREATE TABLE IF NOT EXISTS {}_{} (?1, val blob, PRIMARY KEY (?2))",
                    table,
                    Arc::clone(&VERSION)
                )
                .as_str(),
            )
            .unwrap()
            .execute(params![
                key.keys()
                    .map(|k| format!("{} TYPES text", k.to_string()))
                    .collect::<Vec<String>>()
                    .join(", "),
                key.keys()
                    .map(|k| k.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ])
            .unwrap();
        t.push(table.to_string());
    }
    let _ = conn
        .as_mut()
        .unwrap()
        .prepare(
            format!(
                "REPLACE INTO {}_{} ({}) VALUES (?2)",
                table,
                Arc::clone(&VERSION),
                vec!["?"; key.keys().len()].join(", ")
            )
            .as_str(),
        )
        .unwrap()
        .execute(params![
            key.values()
                .map(|v| String::from_utf8_lossy(
                    serde_json::to_vec(&v.to_string()).unwrap().as_slice()
                )
                .to_string())
                .collect::<Vec<String>>()
                .concat(),
            serde_json::from_value::<Vec<_>>(val.clone()).unwrap()
        ])
        .unwrap();
    let _ = conn.take().unwrap().close().unwrap();
    return val;
}

fn diskcache_wrapper<F, R>(
    func: F,
) -> impl Fn(
    &[Box<dyn erased_serde::Serialize>],
    &HashMap<String, Box<dyn erased_serde::Serialize>>,
) -> Vec<u8>
where
    F: Fn(
        &[Box<dyn erased_serde::Serialize>],
        &HashMap<String, Box<dyn erased_serde::Serialize>>,
    ) -> R,
    R: erased_serde::Serialize + serde::Serialize,
{
    move |args: &[Box<dyn erased_serde::Serialize>],
          kwargs: &HashMap<String, Box<dyn erased_serde::Serialize>>|
          -> Vec<u8> {
        let table = format!("cache_{}", std::any::type_name::<F>());
        let key = hash_key(args, kwargs);

        if let Some(ret) = diskcache_get(
            &table,
            serde_json::from_str::<Map<String, Value>>(&key).unwrap(),
        ) {
            return ret;
        }

        let result = func(args, kwargs);
        let serialized_result = serde_json::to_value(&result).unwrap();

        diskcache_put(
            &table,
            serde_json::from_str::<Map<String, Value>>(&key).unwrap(),
            serialized_result.clone(),
        );

        serde_json::to_vec(&result).unwrap()
    }
}

fn hash_key(
    args: &[Box<dyn erased_serde::Serialize>],
    kwargs: &HashMap<String, Box<dyn erased_serde::Serialize>>,
) -> String {
    let serialized_args =
        serde_json::to_vec(&(args.iter().map(|a| a.as_ref()).collect::<Vec<_>>(), kwargs)).unwrap();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    hasher.write(&serialized_args);
    format!("{:x}", hasher.finish())
}
