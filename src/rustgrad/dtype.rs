use cached::proc_macro::{cached, io_cached};
use itertools::Itertools;
use lazy_static::lazy_static;
use num::integer;
use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::sync::{Arc};
use std::fmt;

use super::shape::sym::BTypes;
lazy_static!{
    pub static ref DTYPES_DICT: Arc<HashMap<TYPES, DTypes>> = {
        let mut hm = HashMap::new();
        hm.insert(TYPES::bool, DTypes::DType(DType::new(0, 1, String::from("bool"), Some('?'), 1)));
        hm.insert(TYPES::int8, DTypes::DType(DType::new(1, 1, String::from("char"), Some('b'), 1)));
        hm.insert(TYPES::uint8, DTypes::DType(DType::new(2, 1, String::from("unsigned char"), Some('B'), 1)));
        hm.insert(TYPES::int16, DTypes::DType(DType::new(3, 2, String::from("short"), Some('h'), 1)));
        hm.insert(TYPES::uint16, DTypes::DType(DType::new(4, 2, String::from("unsigned short"), Some('H'), 1)));
        hm.insert(TYPES::int32, DTypes::DType(DType::new(5, 4, String::from("int"), Some('i'), 1)));
        hm.insert(TYPES::uint32, DTypes::DType(DType::new(6, 4, String::from("unsigned int"), Some('I'), 1)));
        hm.insert(TYPES::int64, DTypes::DType(DType::new(7, 8, String::from("long"), Some('l'), 1)));
        hm.insert(TYPES::uint64, DTypes::DType(DType::new(8, 8, String::from("unsigned long"), Some('L'), 1)));
        hm.insert(TYPES::float16, DTypes::DType(DType::new(9, 2, String::from("half"), Some('e'), 1)));
        hm.insert(TYPES::bfloat16, DTypes::DType(DType::new(10, 2, String::from("__bf16"), None, 1)));
        hm.insert(TYPES::float32, DTypes::DType(DType::new(11, 4, String::from("float"), Some('f'), 1)));
        hm.insert(TYPES::float64, DTypes::DType(DType::new(12, 8, String::from("double"), Some('d'), 1)));
        hm.insert(TYPES::half, hm.get(&TYPES::float16).unwrap().clone());
        hm.insert(TYPES::float,hm.get(&TYPES::float32).unwrap().clone());
        hm.insert(TYPES::double,hm.get(&TYPES::float64).unwrap().clone());
        hm.insert(TYPES::uchar,hm.get(&TYPES::uint8).unwrap().clone());
        hm.insert(TYPES::ushort,hm.get(&TYPES::uint16).unwrap().clone());
        hm.insert(TYPES::uint,hm.get(&TYPES::uint32).unwrap().clone());
        hm.insert(TYPES::ulong,hm.get(&TYPES::uint64).unwrap().clone());
        hm.insert(TYPES::char,hm.get(&TYPES::int8).unwrap().clone());
        hm.insert(TYPES::short,hm.get(&TYPES::int16).unwrap().clone());
        hm.insert(TYPES::int,hm.get(&TYPES::int32).unwrap().clone());
        hm.insert(TYPES::long,hm.get(&TYPES::int64).unwrap().clone());
        hm.insert(TYPES::default_float,hm.get(&TYPES::float32).unwrap().clone());
        hm.insert(TYPES::default_int,hm.get(&TYPES::int32).unwrap().clone());
        Arc::new(hm)        
    };
    pub static ref INVERSE_DTYPES_DICT: Arc<HashMap<String, TYPES>> = {
        let mut hm = HashMap::new();
        // Cloning the Arc and then iterating over the cloned items
        for (k, v) in DTYPES_DICT.clone().iter() {
            if let DTypes::DType(d) = v{
                hm.insert(d.name.clone(), k.clone());
            }
        }
        Arc::new(hm)
    };

    pub static ref promo_lattice: Arc<HashMap<DTypes, Vec<DTypes>>> = {
        let mut hm = HashMap::new();
        hm.insert(DTYPES_DICT.clone().get(&TYPES::bool).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::int8).unwrap().clone(),
            DTYPES_DICT.clone().get(&TYPES::uint8).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::int8).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::int16).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::int16).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::int32).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::int32).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::int64).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::int64).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::float16).unwrap().clone(),
            DTYPES_DICT.clone().get(&TYPES::bfloat16).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::uint8).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::int16).unwrap().clone(),
            DTYPES_DICT.clone().get(&TYPES::uint16).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::uint16).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::int32).unwrap().clone(),
            DTYPES_DICT.clone().get(&TYPES::uint32).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::int32).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::int64).unwrap().clone(),
            DTYPES_DICT.clone().get(&TYPES::uint64).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::int64).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::float16).unwrap().clone(),
            DTYPES_DICT.clone().get(&TYPES::bfloat16).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::float16).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::float32).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::bfloat16).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::float32).unwrap().clone()
        ]);
        hm.insert(DTYPES_DICT.clone().get(&TYPES::float32).unwrap().clone(), vec![
            DTYPES_DICT.clone().get(&TYPES::float64).unwrap().clone()
        ]);
        Arc::new(hm)
    };
}
#[derive(Debug, Clone, Eq, Hash, PartialOrd, Ord)]
pub struct DType{
    priority: isize,
    pub itemsize: isize,
    pub name: String,
    fmt: Option<char>,
    count: isize
}
impl DType{
    fn new(priority: isize, itemsize: isize, name: String, fmt: Option<char>, count: isize) -> DType{
        DType { priority, itemsize, name, fmt, count }
    }
    fn vec(self, sz: isize) -> DType{
        assert!(sz > 1 && &self.count == &1, "{}", format!("cant vectorize {} with size {}", self, sz));
        return DType::new(self.priority, self.itemsize*sz.clone(), format!("{}{}", INVERSE_DTYPES_DICT[&self.name], sz.clone()), None, sz)
    }

    fn scalar(self) -> DTypes{
        if &self.count > &1{
            DTYPES_DICT[&self.name[..self.name.len() - self.count.to_string().len()].parse().unwrap()].clone()
        }else{
            DTypes::DType(self)
        }
    }
}

#[derive(Eq, PartialEq, Hash, Clone)]
pub enum TYPES{
    bool,
    int8,
    uint8,
    int16,
    uint16,
    int32,
    uint32,
    int64,
    uint64,
    float16,
    bfloat16,
    float32,
    float64,
    half,
    float,
    double,
    uchar,
    ushort,
    uint,
    ulong,
    char,
    short,
    int,
    long,
    default_float,
    default_int
}

impl fmt::Display for TYPES {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TYPES::bool => write!(f, "bool"),
            TYPES::int8 => write!(f, "int8"),
            TYPES::uint8 => write!(f, "uint8"),
            TYPES::int16 => write!(f, "int16"),
            TYPES::uint16 => write!(f, "uint16"),
            TYPES::int32 => write!(f, "int32"),
            TYPES::uint32 => write!(f, "uint32"),
            TYPES::int64 => write!(f, "int64"),
            TYPES::uint64 => write!(f, "uint64"),
            TYPES::float16 => write!(f, "float16"),
            TYPES::bfloat16 => write!(f, "bfloat16"),
            TYPES::float32 => write!(f, "float32"),
            TYPES::float64 => write!(f, "float64"),
            TYPES::half => write!(f, "half"),
            TYPES::float => write!(f, "float"),
            TYPES::double => write!(f, "double"),
            TYPES::uchar => write!(f, "uchar"),
            TYPES::ushort => write!(f, "ushort"),
            TYPES::uint => write!(f, "uint"),
            TYPES::ulong => write!(f, "ulong"),
            TYPES::char => write!(f, "char"),
            TYPES::short => write!(f, "short"),
            TYPES::int => write!(f, "int"),
            TYPES::long => write!(f, "long"),
            TYPES::default_float => write!(f, "default_float"),
            TYPES::default_int => write!(f, "default_int"),
        }
    }
}

impl std::str::FromStr for TYPES {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "bool" => Ok(TYPES::bool),
            "int8" => Ok(TYPES::int8),
            "uint8" => Ok(TYPES::uint8),
            "int16" => Ok(TYPES::int16),
            "uint16" => Ok(TYPES::uint16),
            "int32" => Ok(TYPES::int32),
            "uint32" => Ok(TYPES::uint32),
            "int64" => Ok(TYPES::int64),
            "uint64" => Ok(TYPES::uint64),
            "float16" => Ok(TYPES::float16),
            "bfloat16" => Ok(TYPES::bfloat16),
            "float32" => Ok(TYPES::float32),
            "float64" => Ok(TYPES::float64),
            "half" => Ok(TYPES::half),
            "float" => Ok(TYPES::float),
            "double" => Ok(TYPES::double),
            "uchar" => Ok(TYPES::uchar),
            "ushort" => Ok(TYPES::ushort),
            "uint" => Ok(TYPES::uint),
            "ulong" => Ok(TYPES::ulong),
            "char" => Ok(TYPES::char),
            "short" => Ok(TYPES::short),
            "int" => Ok(TYPES::int),
            "long" => Ok(TYPES::long),
            "default_float" => Ok(TYPES::default_float),
            "default_int" => Ok(TYPES::default_int),
            _ => Err(()),
        }
    }
}
impl std::fmt::Display for DType{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = if self.count != 1 { "_" } else { "" };
        write!(f, "dtypes{}{}{}", count, {
            if self.count != 1{
                match INVERSE_DTYPES_DICT.clone().get(&self.name){
                    Some(x) => format!("{}", x.clone()),
                    None => self.clone().scalar().name()
                }
            }else{
                self.name.clone()
            }
        }, self.count)
    }
}

#[derive(Hash, Clone, PartialOrd, Eq, Ord)]
pub struct PtrDType{
    priority: isize,
    pub itemsize: isize,
    pub name: String,
    fmt: Option<char>,
    count: isize
}

impl PtrDType{
    fn new(dt: DTypes) -> PtrDType{
        PtrDType { priority: dt.priority(), itemsize: dt.itemsize(), name: dt.name(), fmt: dt.fmt(), count: dt.count() }
    }
    fn vec(self, sz: isize) -> PtrDType{
        assert!(sz > 1 && &self.count == &1, "{}", format!("cant vectorize {} with size {}", self, sz));
        return PtrDType::new(DTypes::DType(DType::new(self.priority, self.itemsize*sz.clone(), format!("{}{}", INVERSE_DTYPES_DICT[&self.name], sz.clone()), None, sz)))
    }

    fn scalar(self) -> PtrDType{
        if &self.count > &1{
            PtrDType::new(DTYPES_DICT[&self.name[..self.name.len() - self.count.to_string().len()].parse().unwrap()].clone())
        }else{
            self
        }
    }
}

impl PartialEq for PtrDType{
    fn eq(&self, other: &Self) -> bool {
        self.priority==other.priority && self.itemsize==other.itemsize && self.name==other.name && self.count==other.count
    }

    fn ne(&self, other: &Self) -> bool {
        !(self == other)
    }
}

impl PartialEq for DType{
    fn eq(&self, other: &Self) -> bool {
        self.priority==other.priority && self.itemsize==other.itemsize && self.name==other.name && self.count==other.count
    }

    fn ne(&self, other: &Self) -> bool {
        !(self == other)
    }
}
impl Display for PtrDType{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = if self.count != 1 { "_" } else { "" };
        write!(f, "ptr.{}{}{}", count, {
            if self.count != 1{
                match INVERSE_DTYPES_DICT.clone().get(&self.name){
                    Some(x) => format!("{}", x.clone()),
                    None => self.clone().scalar().name
                }
            }else{
                self.name.clone()
            }
        }, self.count)
    }
}

impl Debug for PtrDType{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = if self.count != 1 { "_" } else { "" };
        write!(f, "ptr.{}{}{}", count, {
            if self.count != 1{
                match INVERSE_DTYPES_DICT.clone().get(&self.name){
                    Some(x) => format!("{}", x.clone()),
                    None => self.clone().scalar().name
                }
            }else{
                self.name.clone()
            }
        }, self.count)
    }
}
#[derive(Hash, Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct ImageDType{
    priority: isize,
    itemsize: isize,
    pub name: String,
    fmt: Option<char>,
    count: isize,
    pub shape: Vec<isize>,
    pub base: DType
}

impl ImageDType{
    fn vec(self, sz: isize) -> DType{
        self.base.vec(sz)
    }

    fn scalar(self) -> DType{
        self.base
    }
}

impl Display for ImageDType{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("dtypes.{}({:?})", self.name.clone(), self.shape))
    }
}

impl Debug for ImageDType{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", format!("dtypes.{}({:?})", self.name.clone(), self.shape))
    }
}

fn is_float(x: &DTypes) -> bool {
    let p_dd = DTYPES_DICT.clone();
    match p_dd.get(&TYPES::float16) {
        Some(y) if x == y => true,
        _ => match p_dd.get(&TYPES::float32) {
            Some(y) if x == y => true,
            _ => match p_dd.get(&TYPES::float64) {
                Some(y) if x == y => true,
                _ => false,
            },
        },
    }
}

fn is_int(x: &DTypes) -> bool {
    let p_dd = DTYPES_DICT.clone();
    match p_dd.get(&TYPES::int8) {
        Some(y) if x == y => true,
        _ => match p_dd.get(&TYPES::int16) {
            Some(y) if x == y => true,
            _ => match p_dd.get(&TYPES::int32) {
                Some(y) if x == y => true,
                _ => match p_dd.get(&TYPES::int64){
                    Some(y) if x == y => true,
                    _ => is_unsigned(x)
                },
            },
        },
    }
}

fn is_unsigned(x: &DTypes) -> bool {
    let p_dd = DTYPES_DICT.clone();
    match p_dd.get(&TYPES::uint8) {
        Some(y) if x == y => true,
        _ => match p_dd.get(&TYPES::uint16) {
            Some(y) if x == y => true,
            _ => match p_dd.get(&TYPES::uint32) {
                Some(y) if x == y => true,
                _ => match p_dd.get(&TYPES::uint64){
                    Some(y) if x ==y => true,
                    _ => false
                },
            },
        },
    }
}

fn dtype_fields() -> Arc<HashMap<TYPES, DTypes>>{
    return DTYPES_DICT.clone()
}


fn _get_recursive_parents(dtype: &DTypes) -> HashSet<DTypes>{
    if dtype != &DTYPES_DICT.clone().get(&TYPES::float64).unwrap().clone(){
        let mut vec = promo_lattice
        .clone()
        .get(&dtype)
        .unwrap()
        .clone()
        .iter()
        .map(|d| _get_recursive_parents(d))
        .collect_vec();

    vec.push({
        let mut x = HashSet::new();
        x.insert(dtype.clone());
        x
    });

    let mut union_set = HashSet::new();
    for inner_vec in &vec {
        for elem in inner_vec {
            union_set.insert(elem.clone());
        }
    }

    union_set

    }else{
        let mut x = HashSet::new();
        x.insert(DTYPES_DICT.clone().get(&TYPES::float64).unwrap().clone());
        x
    }
}

fn least_upper_dtype(ds: &[DTypes]) -> DTypes{
    let images: Vec<&DTypes> = ds.iter().filter_map(|d|{
        if let DTypes::ImageDType(i) = d{
            Some(d)
        }else{
            None
        }
    }).collect_vec();

    if !images.is_empty(){
        let x: HashSet<DTypes> = ds
        .iter()
        .map(|d| _get_recursive_parents(d))
        .fold(None, |acc: Option<HashSet<DTypes>>, set| {
            Some(match acc {
                Some(acc) => acc.intersection(&set).cloned().collect(),
                None => set,
            })
        })
        .unwrap_or_else(|| HashSet::new());
        x.into_iter().min().unwrap()
    }else{
        images[0].clone()
    }
}

fn least_upper_float(dt: DTypes) -> DTypes{
    if is_float(&dt){
        dt
    }else{
        least_upper_dtype(&vec![dt, DTYPES_DICT.clone().get(&TYPES::float32).unwrap().clone()])
    }
}

fn imageh(shp: Vec<isize>) -> ImageDType{
    ImageDType { priority: 100, itemsize: 2, name: String::from("imageh"), fmt: Some('e'), count: 1, shape: shp, base: {
        if let DTypes::DType(d) = DTYPES_DICT.clone().get(&TYPES::float32).unwrap().clone(){
            d
        }else{
            panic!()
        }
    } }
}

fn imagef(shp: Vec<isize>) -> ImageDType{
    ImageDType { priority: 100, itemsize: 4, name: String::from("imageh"), fmt: Some('f'), count: 1, shape: shp, base: {
        if let DTypes::DType(d) = DTYPES_DICT.clone().get(&TYPES::float32).unwrap().clone(){
            d
        }else{
            panic!()
        }
    } }
}
#[derive(Hash, Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub enum DTypes{
    DType(DType),
    ImageDType(ImageDType),
    PtrDType(PtrDType)
}

impl DTypes{
    fn priority(&self) -> isize{
        match self{
            DTypes::DType(d) => d.priority.clone(),
            DTypes::ImageDType(d) => d.priority.clone(),
            DTypes::PtrDType(d) => d.priority.clone()
        }
    }
    pub fn itemsize(&self) -> isize{
        match self{
            DTypes::DType(d) => d.itemsize.clone(),
            DTypes::ImageDType(d) => d.itemsize.clone(),
            DTypes::PtrDType(d) => d.itemsize.clone()
        }
    }

    pub fn name(&self) -> String{
        match self{
            DTypes::DType(d) => d.name.clone(),
            DTypes::ImageDType(d) => d.name.clone(),
            DTypes::PtrDType(d) => d.name.clone()
        }
    }

    fn fmt(&self) -> Option<char>{
        match self{
            DTypes::DType(d) => d.fmt.clone(),
            DTypes::ImageDType(d) => d.fmt.clone(),
            DTypes::PtrDType(d) => d.fmt.clone()
        }
    }

    fn count(&self) -> isize{
        match self{
            DTypes::DType(d) => d.count.clone(),
            DTypes::ImageDType(d) => d.count.clone(),
            DTypes::PtrDType(d) => d.count.clone()
        }
    }
}