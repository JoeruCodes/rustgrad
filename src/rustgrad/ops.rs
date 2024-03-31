use std::{any::Any, borrow::BorrowMut, cell::RefCell, collections::{hash_map::DefaultHasher, HashMap, HashSet}, fmt::Debug, hash::{Hash, Hasher}, ops::Deref, rc::{Rc, Weak}};

use itertools::Itertools;
use lazy_static::lazy_static;

use super::{codegen::kernel::LocalBuffer, dtype::{DType, DTypes, DTYPES_DICT}, helpers::Context, shape::{shapetracker::ShapeTracker, sym::{BTypes, NodeTypes}}};
lazy_static!{
    pub static ref InterpretedFlopCounter: HashMap<Op, >
}

#[derive(PartialEq, Clone, Debug, Hash)]
enum UnaryOps{
    EXP2,
    LOG2,
    CAST,
    SIN,
    SQRT,
    NEG
}
#[derive(PartialEq, Clone, Debug, Hash)]
enum BinaryOps{
    ADD,
    SUB,
    MUL,
    DIV,
    MAX,
    MOD,
    CMPLT,
    CMPEQ,
    XOR
}
#[derive(PartialEq, Clone, Debug, Hash)]
enum TernaryOps{
    WHERE
}
#[derive(PartialEq, Clone, Debug, Hash)]
enum ReduceOps{
    SUM,
    MAX,
}
#[derive(PartialEq, Clone, Debug, Hash)]
pub enum BufferOps{
    LOAD,
    CONST,
    STORE
}
#[derive(PartialEq, Clone, Debug, Hash)]
enum LoadOps{
    EMPTY,
    CONST,
    COPY,
    CONTIGUOUS,
    CUSTOM,
    SYNC,
    WAIT,
    ASSIGN
}
#[derive(PartialEq, Clone, Debug, Hash)]
pub enum Op{
    UnaryOps(UnaryOps),
    BinaryOps(BinaryOps),
    ReduceOps(ReduceOps),
    LoadOps(LoadOps),
    TernaryOps(TernaryOps),
    BufferOps(BufferOps)
}
#[derive(Debug, PartialEq, Hash, Clone)]
pub struct MemBuffer{
    idx: usize,
    pub dtype: DTypes,
    pub st: Rc<ShapeTracker>
}

#[derive(Debug, PartialEq, Clone)]
pub struct ConstBuffer{
    pub val: f64,
    pub dtype: DTypes,
    pub st: Rc<ShapeTracker>
}
impl Hash for ConstBuffer{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (format!("{}", self.val), self.dtype, self.st).hash(state)
    }
}
struct ScheduleItem{
    ast: Vec<LazyOp>,
    outputs: Vec<LazyBuffer>,
    inputs: Vec<LazyBuffer>,
    var_vals: HashMap<Rc<NodeTypes>, isize>
}

#[derive(Clone)]
pub struct LazyOp{
    pub op: Op,
    pub src: Vec<Rc<LazyOp>>,
    pub arg: Option<Items>,
    ptr: RefCell<Option<Weak<Self>>>
}

impl LazyOp
{
    fn new(op: Op, src: Vec<Rc<LazyOp>>, arg: Option<Items>) -> Rc<LazyOp>{
        let l = Rc::new(LazyOp{
            op, src, arg, ptr: RefCell::new(None)
        });

        l.clone().ptr.borrow_mut().replace(Rc::downgrade(&l));
        l
    }

    fn ptr(&self) -> Rc<LazyOp>{
        self.ptr.borrow().as_ref().unwrap().upgrade().unwrap()
    }
    fn cached_compare(&self, x: Rc<LazyOp>, context: &mut HashMap<(usize, usize), bool>) -> bool{
        if self as *const _ == x.clone().deref() as *const _ {
            return true;
        }
        if self.op != x.op || self.arg != x.arg || self.src.len() != x.src.len() {
            return false;
        }
        let key = (self as *const _) as usize;
        let x_key = (x.deref() as *const _) as usize;
        if let Some(&ret) = context.get(&(key, x_key)) {
            return ret;
        }
        let ret = self.src.iter().zip(x.src.iter()).all(|(a, b)| a.cached_compare(b.clone(), context));
        context.insert((key, x_key), ret);
        ret
    }

    fn key(&self) -> [u8; 256] {
        let mut hasher = DefaultHasher::new();
        let key_str = self
            .src
            .iter()
            .map(|s| &s.clone().deref().key())
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        
        let mut combined_key: Vec<u8> = Vec::new();
        combined_key.extend_from_slice(&key_str);
    
        combined_key.extend_from_slice(&(format!("{:?}", (self.op, self.arg)).as_bytes()));
    
        combined_key.into_iter().reduce(|x, y| x+y).unwrap().hash(&mut hasher);
        let hash_value = hasher.finish();
    
        let mut result: [u8; 256] = [0; 256];
        for i in 0..32 {
            result[i] = ((hash_value >> (i * 8)) & 0xff) as u8;
        }
        result
    }

    fn hash(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (self.op, self.src, self.arg).hash(&mut hasher);
        hasher.finish()
    }
    
    pub fn lazyops(&self) -> Vec<Rc<LazyOp>>{
        let mut x: Vec<Rc<LazyOp>> = self.src.iter().map(|x| x.clone().lazyops().into_iter().map(|item| item.clone()).collect::<Vec<Rc<LazyOp>>>()).flatten().collect_vec();
        x.push(self.ptr());
        x
    }

    pub fn vars(&self) -> Vec<Rc<NodeTypes>>{
        let mut union_set: HashSet<Rc<NodeTypes>> = HashSet::new();
        for x in &self.lazyops(){
            if let Op::BufferOps(_) = &x.op{
                union_set.extend(x.vars());
            }
        }

        let mut sorted_union: Vec<Rc<NodeTypes>> = union_set.into_iter().collect_vec();
        sorted_union.sort_by_key(|x| {
            match x.clone().deref(){
                NodeTypes::Variable(v) => v.expr.clone(),
                _ => unreachable!()
            }
        });

        sorted_union

    }
}

impl PartialEq for LazyOp{
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.src == other.src && self.arg == other.arg
    }
}
impl Hash for LazyOp{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.op, self.src, self.arg).hash(state)
    }
}

#[derive(Clone)]
pub struct FlopCounter{
    shape: Vec<isize>,
    pub dtype: DTypes,
    flops: BTypes,
    mem: HashMap<usize, usize>
}
impl FlopCounter{

    fn new(shape: Vec<isize>, dtype: DTypes, flops: BTypes, mem: HashMap<usize, usize>) -> FlopCounter{
        FlopCounter { shape, dtype, flops, mem }
    }
    fn mem_estimate(&self) -> usize{
        self.mem.values().sum()
    }

    fn consume_flops(&mut self) -> BTypes{
        let ret = self.flops;
        self.flops = BTypes::Int(0);
        ret
    }
}

pub fn get_lazyop_info(ast: Rc<LazyOp>) -> FlopCounter{
    let self_n_frnds = ast.clone().src.iter().map(|x| run_ast(x.clone())).collect_vec();
    fn run_ast(ast: Rc<LazyOp>) -> FlopCounter{
        let self_n_frnds = ast.clone().deref().src.into_iter().map(|x|run_ast(x)).collect_vec();
        let args = ast.clone().deref().arg;
        match ast.clone().deref().op{
            Op::BinaryOps(_) => {
                return BinaryOps::interpreted_flop_counter(ast.clone().deref().op, self_n_frnds, args)
            },
            Op::BufferOps(_) => {
                return  BufferTypes::interpreted_flop_counter(ast.clone().deref().op, self_n_frnds, args);
            },
            Op::UnaryOps(_) => {
                return UnaryOps::interpreted_flop_counter(ast.clone().deref().op, self_n_frnds, args);
            },
            Op::ReduceOps(_) => {
                return ReduceOps::interpreted_flop_counter(ast.clone().deref().op, self_n_frnds, args);
            },
            Op::TernaryOps(_) => {
                return TernaryOps::interpreted_flop_counter(ast.clone().deref().op, self_n_frnds)
            }
            _ => unreachable!()
        }
    }
    run_ast(ast)
}

#[derive(Debug, PartialEq, Hash, Clone)]
pub enum BufferTypes{
    ConstBuffer(ConstBuffer),
    MemBuffer(MemBuffer),
    LocalBuffer(LocalBuffer)
}

impl BufferTypes{
    fn interpreted_flop_counter(op: Op, self_n_frnds: Vec<FlopCounter>, arg: Option<Items>) -> FlopCounter{
        match op{
            Op::BufferOps(b) => {
                match b{
                    BufferOps::LOAD => {
                        assert!(self_n_frnds.is_empty() && arg.is_some());
                        match &arg.unwrap(){
                            Items::Buffer(bff) => {
                                match bff{
                                    BufferTypes::MemBuffer(c) => FlopCounter::new({
                                        c.st.shape().into_iter().map(|x|{
                                            match x{
                                                BTypes::Int(i) => i,
                                                BTypes::Node(_) => panic!()
                                            }
                                        }).collect_vec()
                                    }, c.dtype, BTypes::Int(0), HashMap::new()),
                                    
                                    _ => panic!()
                                }
                            }
                            _ => panic!()
                        }
                    },
                    BufferOps::CONST =>{
                        assert!(self_n_frnds.is_empty() && arg.is_some());
                        match &arg.unwrap(){
                            Items::Buffer(bff) => {
                                match bff{
                                    BufferTypes::MemBuffer(c) => FlopCounter::new({
                                        c.st.shape().into_iter().map(|x|{
                                            match x{
                                                BTypes::Int(i) => i,
                                                BTypes::Node(_) => panic!()
                                            }
                                        }).collect_vec()
                                    }, c.dtype, BTypes::Int(0), HashMap::new()),
                                    BufferTypes::ConstBuffer(c) => FlopCounter::new({
                                        c.st.shape().into_iter().map(|x|{
                                            match x{
                                                BTypes::Int(i) => i,
                                                BTypes::Node(_) => panic!()
                                            }
                                        }).collect_vec()
                                    }, c.dtype, BTypes::Int(0), HashMap::new()),
                                    _ => panic!()
                                }
                            }
                            _ => panic!()
                        }
                    },
                    BufferOps::STORE => {
                        assert!(self_n_frnds.len() == 1 && arg.is_some());
                        let mut self_n = self_n_frnds[0].clone();
                        match &arg.unwrap(){
                            Items::Buffer(bff) => {
                                match bff{
                                    BufferTypes::MemBuffer(s) => FlopCounter::new({
                                        s.st.shape().into_iter().map(|x|{
                                            match x{
                                                BTypes::Int(i) => i,
                                                BTypes::Node(_) => panic!()
                                            }
                                        }).collect_vec()
                                    }, s.dtype, self_n.consume_flops(), {
                                        let mut x = HashMap::new();
                                        self_n.mem.into_iter().for_each(|(k, v)| {x.insert(k, v);});
                                        x.insert(s.idx, s.dtype.itemsize() as usize*s.st.real_size() as usize);
                                        x
                                    }),
                                    _ => panic!()
                                }
                            }
                            _ => panic!()
                        }
                    }
                }
            }
            _ => panic!()
        }
    }
}

impl UnaryOps{
    fn interpreted_flop_counter(op: Op, self_n_frnds: Vec<FlopCounter>, arg: Option<Items>) -> FlopCounter{
        let mut self_n = self_n_frnds;
        match op{
            Op::UnaryOps(u) => {
                match u{
                    UnaryOps::CAST => {
                        assert!(self_n_frnds.len() == 1 && arg.is_some());
                        if let Items::Dtype(d) = arg.unwrap(){
                            FlopCounter::new(self_n[0].shape, d, self_n[0].consume_flops(), self_n[0].mem)
                        }else{
                            panic!()
                        }
                    },
                    _ =>{
                        if self_n.len() == 1 && arg.is_none(){
                            FlopCounter::new(self_n[0].shape.clone(), DTYPES_DICT.clone().get(&super::dtype::TYPES::bool).unwrap().clone(), &self_n[0].consume_flops() + &BTypes::Int(self_n[0].shape.iter().product()), self_n[0].mem)
                        }else {
                            panic!()
                        }
                    }
                }
            },
            _ => panic!()
        }
    }
}

impl BinaryOps{
    fn interpreted_flop_counter(op: Op, self_n_frnds: Vec<FlopCounter>, arg: Option<Items>) -> FlopCounter{
        assert!(self_n_frnds.len() == 2);
        let mut self_n = self_n_frnds;
        FlopCounter::new(self_n[0].shape.clone(), {
            match arg{
                Some(a) => {
                    match a{
                        Items::Op(o) => {
                            match o{
                                Op::BinaryOps(b) => {
                                    match b{
                                        BinaryOps::CMPEQ | BinaryOps::CMPLT => {
                                            DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::bool).unwrap().clone()
                                        },
                                        _ => self_n[0].dtype
                                    }
                                }
                                _ => self_n[0].dtype
                            }
                        }
                        _ => panic!()
                    }
                },
                None => {
                    match op{
                        Op::BinaryOps(b) => {
                            match b{
                                BinaryOps::CMPEQ | BinaryOps::CMPLT => {
                                    DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::bool).unwrap().clone()
                                },
                                _ => self_n[0].dtype
                            }
                        }
                        _ => self_n[0].dtype
                    }
                }
            }
        }, &(&self_n[0].consume_flops() + &self_n[1].consume_flops()) + &BTypes::Int(self_n[0].shape.into_iter().product()), {
            let mut x = HashMap::new();
            x.extend(self_n[0].mem);
            x.extend(self_n[1].mem);
            x
        })
    }
}

impl ReduceOps{
    fn interpreted_flop_counter(op: Op, self_n_frnds: Vec<FlopCounter>, axis: Option<Items>) -> FlopCounter{
        assert!(self_n_frnds.len() == 1);
        let mut self_n = self_n_frnds;
        FlopCounter::new({
            self_n[0].shape.into_iter().enumerate().map(|(i, s)|{
                if let Items::Axis(a) = axis.unwrap(){
                    if a.contains(&(i as isize)){
                        1
                    }else{
                        s
                    }
                }else{
                    panic!()
                }
            }).collect_vec()
        }, self_n[0].dtype, &self_n[0].consume_flops() + &BTypes::Int(self_n[0].shape.iter().product()), self_n[0].mem)
    }
}

impl TernaryOps{
    fn interpreted_flop_counter(op: Op, self_n_frnds: Vec<FlopCounter>) -> FlopCounter{
        assert!(self_n_frnds.len() == 3);
        let mut self_n = self_n_frnds;
        FlopCounter::new(self_n[0].shape.clone(), self_n[1].dtype, &(&(&self_n[0].consume_flops() + &self_n[1].consume_flops()) + &self_n[2].consume_flops()) + &BTypes::Int(self_n[0].shape.iter().product()), {
            let mut x = HashMap::new();
            x.extend(self_n[0].mem.iter());
            x.extend(self_n[1].mem.iter());
            x.extend(self_n[2].mem.iter());
            x
        })
    }
}

#[derive(Debug, PartialEq, Hash, Clone)]
pub enum Items {
    Axis(Vec<isize>),
    Op(Op),
    Dtype(DTypes),
    Buffer(BufferTypes)
}

struct GlobalCounters{
    global_ops: usize,
    global_mem: usize,
    time_sum_s: f64,
    kernel_count: usize,
    mem_used: usize
}


impl GlobalCounters {
    fn reset(&mut self) {
        self.global_ops = 0;
        self.global_mem = 0;
        self.time_sum_s = 0.0;
        self.kernel_count = 0;
    }
}

impl Default for GlobalCounters {
    fn default() -> Self {
        GlobalCounters {
            global_ops: 0,
            global_mem: 0,
            time_sum_s: 0.0,
            kernel_count: 0,
            mem_used: 0, // NOTE: this is not reset
        }
    }
}