use std::{any::Any, borrow::{Borrow, BorrowMut}, collections::{HashMap, HashSet}, fmt::{Debug, Display}, i32::MAX, isize::{self, MAX}, mem::swap, ops::Deref, rc::Rc, sync::Arc};

use itertools::{iproduct, Itertools};
use lazy_static::lazy_static;
use num::{Integer, Num};
use num_traits::NumAssignOps;

use crate::rustgrad::{device::Device, dtype::{self, DTypes, ImageDType, DTYPES_DICT}, helpers::{self, all_int, ansilen, colored, dedup, flatten, get_contraction, getenv, prod, round_up, DEBUG}, ops::{get_lazyop_info, BufferOps, BufferTypes, FlopCounter, Items, LazyOp, MemBuffer, Op}, shape::{shapetracker::ShapeTracker, sym::{BTypes, NodeTypes}, view::{strides_for_shape, View}}};

lazy_static!{
    pub static ref TENSOR_CORES: Arc<HashMap<String, Vec<TensorCore>>> = {
        let mut hm = HashMap::new();
        hm.insert(String::from("METAL"), vec![
            TensorCore{dims: vec![8, 8, 8], dtype_out:DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::float).unwrap().clone(), dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::float).unwrap().clone(), wmma_func: String::from("__metal_wmma<float2,simdgroup_float8x8,float2>"), threads: vec![(0,2), (1,4), (0,2),(1,2)], thread_local_sizes: vec![vec![2], vec![2], vec![2]],         thread_local_aliases: vec![
                vec![vec![4], vec![0], vec![2], vec![0], vec![-1, 1, 3], vec![0]],
                vec![vec![0], vec![3], vec![0], vec![1], vec![2, 4], vec![-1]],
                vec![vec![4], vec![3], vec![2], vec![1], vec![0], vec![-1]],
            ]},
            TensorCore {
                dims: vec![8, 8, 8],
                dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                dtype_out: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::float).unwrap().clone(),
                threads: vec![(0, 2), (1, 4), (0, 2), (1, 2)],
                thread_local_sizes: vec![vec![2], vec![2], vec![2]],
                thread_local_aliases: vec![
                    vec![vec![4], vec![0], vec![2], vec![0], vec![-1, 1, 3], vec![0]],
                    vec![vec![0], vec![3], vec![0], vec![1], vec![2, 4], vec![-1]],
                    vec![vec![4], vec![3], vec![2], vec![1], vec![0], vec![-1]],
                ],
                wmma_func: "__metal_wmma<half2,simdgroup_float8x8,float2>".to_string(),
            },
            TensorCore {
                dims: vec![8, 8, 8],
                dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                dtype_out: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                threads: vec![(0, 2), (1, 4), (0, 2), (1, 2)],
                thread_local_sizes: vec![vec![2], vec![2], vec![2]],
                thread_local_aliases: vec![
                    vec![vec![4], vec![0], vec![2], vec![0], vec![-1, 1, 3], vec![0]],
                    vec![vec![0], vec![3], vec![0], vec![1], vec![2, 4], vec![-1]],
                    vec![vec![4], vec![3], vec![2], vec![1], vec![0], vec![-1]],
                ],
                wmma_func: "__metal_wmma<half2,simdgroup_half8x8,half2>".to_string(),
            }
        ]);
        hm.insert(String::from("HSA"), vec![
            TensorCore {
                dims: vec![16, 16, 16],
                dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                dtype_out: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::float).unwrap().clone(),
                threads: vec![(0, 16), (1, 2)],
                thread_local_sizes: vec![vec![16], vec![16], vec![8]],
                thread_local_aliases: vec![
                    vec![vec![0], vec![0], vec![-1], vec![1]],
                    vec![vec![0], vec![1], vec![-1], vec![0]],
                    vec![vec![0], vec![1], vec![0], vec![2, -1]],
                ],
                wmma_func: "__builtin_amdgcn_wmma_f32_16x16x16_f16_w32".to_string(),
            },
            TensorCore {
                dims: vec![16, 16, 16],
                dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                dtype_out: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                threads: vec![(0, 16), (1, 2)],
                thread_local_sizes: vec![vec![16], vec![16], vec![8]],
                thread_local_aliases: vec![
                    vec![vec![0], vec![0], vec![-1], vec![1]],
                    vec![vec![0], vec![1], vec![-1], vec![0]],
                    vec![vec![0], vec![1], vec![0], vec![2, -1]],
                ],
                wmma_func: "__hip_wmma_f16_f16".to_string(),
            }
        ]);

        hm.insert(String::from("CUDA"), vec![
            TensorCore {
                dims: vec![8, 16, 16],
                dtype_in: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::half).unwrap().clone(),
                dtype_out: DTYPES_DICT.clone().get(&crate::rustgrad::dtype::TYPES::float).unwrap().clone(),
                threads: vec![(0, 2), (0, 2), (1, 2), (1, 2), (0, 2)],
                thread_local_sizes: vec![vec![2, 2, 2], vec![2, 2], vec![2, 2]],
                thread_local_aliases: vec![
                    vec![
                        vec![0],
                        vec![-2],
                        vec![5],
                        vec![0],
                        vec![0],
                        vec![-1, 1, 2, -3],
                        vec![3, 4],
                    ],
                    vec![
                        vec![5],
                        vec![0],
                        vec![0],
                        vec![4],
                        vec![3],
                        vec![-1, 1, 2, -2],
                        vec![0],
                    ],
                    vec![
                        vec![2],
                        vec![-2],
                        vec![5],
                        vec![1],
                        vec![-1],
                        vec![0],
                        vec![3, 4],
                    ],
                ],
                wmma_func: "__cuda_mma_m16n8k16_f16_f32".to_string(),
            }
        ]);

        Arc::new(hm)
    };
}
#[derive(PartialEq, PartialOrd, Debug)]
enum OptOps{
    TC,
    UPCAST,
    UPCASTMID,
    UNROLL,
    LOCAL,
    GROUP,
    GROUPTOP,
    NOLOCALS,
    PADTO
}

#[derive(PartialEq, PartialOrd, Clone)]
struct Opt{
    op: OptOps,
    axis: Option<usize>,
    amt: Option<isize>
}

impl Debug for Opt{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Opt(op={:?}, axis={:?}, amt={:?})", self.op, self.axis, self.amt)
    }
}

impl Display for Opt{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Opt(op={:?}, axis={:?}, amt={:?})", self.op, self.axis, self.amt)
    }
}

#[derive(Clone, Debug)]
struct TensorCore{
    dims: Vec<isize>,
    dtype_in: DTypes,
    dtype_out: DTypes,
    threads: Vec<(isize, isize)>,
    thread_local_aliases: Vec<Vec<Vec<isize>>>,
    thread_local_sizes: Vec<Vec<isize>>,
    wmma_func: String
}

impl ToString for TensorCore{
    fn to_string(&self) -> String {
        format!("tensor_core<{:?}, {:?}, {:?}>", self.dims, self.dtype_in, self.dtype_out)
    }
}

impl TensorCore{
    fn num_threads(&self) -> usize{
        self.threads.len()
    }
    fn num_upcasts(&self) -> usize{
        self.thread_local_aliases[0].len() - self.num_threads()
    }
}

#[derive(Clone)]
struct TensorCoreOptions{
    bufs: (usize, usize),
    axes: Vec<usize>,
    axes_exist: Vec<bool>
}

impl TensorCoreOptions{
    fn fix_axes(&mut self, removed_axis: usize){
        for (to_dim, &exist) in self.axes_exist.iter().enumerate(){
            if exist{
                if removed_axis < self.axes[to_dim]{
                    self.axes[to_dim] -= 1;
                }else if removed_axis == self.axes[to_dim]{
                    self.axes_exist[to_dim] = false;
                }
            }
        }
    }
}

#[derive(Debug, PartialEq, Hash, Clone)]
pub struct LocalBuffer{
    name: String,
    size: usize,
    dtype: DTypes,
    realized: Option<bool>
}
impl LocalBuffer{
    fn new(name: String, size: usize, dtype:DTypes) -> LocalBuffer{
        LocalBuffer { name, size, dtype, realized: None }
    }
}

impl ToString for LocalBuffer{
    fn to_string(&self) -> String {
        return format!("localbuffer<{}[{}]>", self.name, self.size)
    }
}

#[derive(Clone)]
struct LinearizerOptions{
    device: String,
    suffix: String,
    supports_float4: bool,
    has_local: bool,
    has_shared: bool,
    hash_tensor_cores: bool,
    global_max: Option<Vec<usize>>,
    local_max: Option<Vec<usize>>,
    shared_max: usize
}

impl Default for LinearizerOptions{
    fn default() -> Self {
        LinearizerOptions { device: String::new(), suffix: String::new(), supports_float4: true, has_local: true, has_shared: true, hash_tensor_cores: false, global_max: None, local_max: None, shared_max: 32768 }
    }
}

#[derive(Clone)]
struct Kernel{
    opts: Option<LinearizerOptions>,
    ast: Vec<Rc<LazyOp>>,
    lazyops: Vec<Rc<LazyOp>>,
    info: FlopCounter,
    reduceop: Option<Rc<LazyOp>>,
    outbufs: Vec<Option<Items>>,
    vars: Vec<Rc<NodeTypes>>,
    bufs: Vec<BufferTypes>,
    earlybufs: Vec<Option<Items>>,
    full_buf_index: usize,
    sts: Vec<Rc<ShapeTracker>>,
    applied_opts: Vec<Opt>,
    group_for_reduces: usize,
    upcasted: usize,
    local_dims: usize,
    local_alias: HashMap<isize, BufferTypes>,
    tensor_core: Option<TensorCore>,
    tensor_core_opts: Option<TensorCoreOptions>,
    dont_use_locals: bool,
    applied_opts_cache: Option<Vec<Opt>>
}

impl Kernel{
    fn new(ast: Vec<Rc<LazyOp>>, opts: Option<LinearizerOptions>) -> Kernel{
        let opts_n = {
            opts.or({
                let device = Device.index(&Device.default());
                if let Some(d) = device.compiler{
                    Some(d.linearizer_opts)
                }else{
                    None
                }
            })
        };

        assert!(ast.iter().all(|op|{
            Op::BufferOps(BufferOps::STORE) == op.op
        }));

        assert!(ast.iter().map(|op|{
            if let Some(Items::Buffer(b)) = &op.arg{
                match b{
                    BufferTypes::ConstBuffer(c) => c.st.size(),
                    BufferTypes::MemBuffer(m) => m.st.size(),
                    BufferTypes::LocalBuffer(_) => panic!()
                }
            }else{
                panic!()
            }
        }).collect::<HashSet<usize>>().len() == 1);

        let ast_n = ast;
        let lazyops = ast_n.iter().map(|op|op.lazyops()).flatten().collect_vec();
        let info = get_lazyop_info(ast_n[0].clone());
        let reduceops = {
            let redops = lazyops.iter().filter_map(|x|{
                if let Op::ReduceOps(_) = x.op{
                    Some(x.clone())
                }else{
                    None
                }
            }).collect_vec();
            assert!(dedup(redops).len() <= 1);
            if !redops.is_empty(){
                Some(redops[0])
            }else{
                None
            }
        };

        let outbufs = ast_n.iter().map(|x| x.clone().arg).collect_vec();
        let vars = ast_n.iter().map(|x|x.vars()).flatten().collect_vec();

        let mut bufs: Vec<BufferTypes>  = outbufs.clone().into_iter().filter_map(|x|{
            if let Some(Items::Buffer(b)) = x{
                Some(b)
            }else{
                None
            }
        }).collect_vec();

        bufs.extend(lazyops.iter().filter_map(|x|{
            if let Op::BufferOps(BufferOps::CONST) = &x.op{
                if let Some(Items::Buffer(b)) = &x.arg{
                    return Some(b.clone())
                }
            }else if let Op::BufferOps(BufferOps::LOAD) = &x.op{
                if let Some(Items::Buffer(b)) = &x.arg{
                    return Some(b.clone())
                }
            }
            None
        }));

        let earlybufs = {
            if let Some(r) = &reduceops{
                r.lazyops().iter().filter_map(|x|{
                    if let Op::BufferOps(_) = &x.op{
                        Some(x.arg.clone())
                    }else{
                        None
                    }
                }).collect_vec()
            }else{
                vec![]
            }
        };

        let full_buf_index = {
            if !earlybufs.is_empty(){
                bufs.iter().position(|x| &Some(Items::Buffer(x.clone()))==&earlybufs[0])
            }else{
                Some(0)
            }
        };

        let sts = {
           bufs.iter().filter_map(|x|{
            if let BufferTypes::MemBuffer(m) = x{
                Some(m.st.clone())
            } else if let BufferTypes::ConstBuffer(c) = x{
                Some(c.st.clone())
            }else{
                None
            }
           }).collect_vec()
        };

        let reduce = full_shape(&sts, full_buf_index.unwrap_or(0)).iter().zip(output_shape(&sts).iter()).enumerate().collect_vec();
        let mut premute = reduce.iter().filter_map(|(i, (s, n))|{
            if s == n{
                Some(i.clone())
            }else{
                None
            }
        }).collect_vec();

        premute.extend(reduce.into_iter().filter_map(|(i, (s, n))|{
            if s!=n{
                Some(i)
            }else{
                None
            }
        }));

        reshape_and_permute(&mut sts,None, Some(premute));



        let appiled_opts = vec![];
        let group_for_reduces:usize = 0;
        let upcasted: usize = 0;
        let local_dims: usize = 0;
        let local_alias: HashMap<isize, BufferTypes> = HashMap::new();
        let tensor_core = None;
        let tensor_core_opts = None;
        let dont_use_locals = false;

        let mut k = Kernel{
            opts: opts_n,
            ast: ast_n,
            lazyops,
            info,
            reduceop: reduceops,
            outbufs,
            vars,
            bufs,
            earlybufs,
            full_buf_index: full_buf_index.unwrap(),
            sts,
            applied_opts: appiled_opts,
            group_for_reduces,
            upcasted,
            local_dims,
            local_alias,
            tensor_core,
            tensor_core_opts,
            dont_use_locals,
            applied_opts_cache:None
        };
        k.simplify_ones();
        k.simplify_merge_adjacent();

        k
    }


    fn full_shape(&self) -> Vec<BTypes>{
        self.sts[self.full_buf_index].clone().shape()
    }

    fn output_shape(&self) -> Vec<BTypes>{
        self.sts[0].clone().shape()
    }

    fn reshape_and_permute<T>(&mut self, new_shape_fxn: Option<T>, axis: Option<Vec<usize>>)
        where
            T: Fn(Vec<BTypes>) -> Vec<BTypes>
    {
        let mut new_sts = vec![];
        self.sts.into_iter().for_each(|mut st|{
            if let Some(x) = new_shape_fxn{
                st = *st.reshape(&x(st.shape())).borrow_mut()
            }
            if let Some(x) = axis{
                st = st.permute(&x)
            }
            new_sts.push(st)
        });
        self.sts = new_sts;
    }

    fn shape_len(&self) -> usize{
        self.sts[0].shape().len()
    }

    fn simplify_ones(&mut self) -> bool{
        if self.shape_len() == 0{
            return false
        }
        let all_ones = self.full_shape().into_iter().map(|s| s==BTypes::Int(1)).collect_vec();
        self.local_dims -= all_ones[self.first_reduce() - self.local_dims .. self.first_reduce()].iter().map(|x|{
            if x== &true{
                1
            }else{
                0
            }
        }).fold(0, |acc, x| acc + x);
        self.upcasted -= all_ones[self.shape_len() -self.upcasted.clone() ..].iter().map(|x|{
            if x == &true{
                1
            }else{
                0
            }
        }).fold(0, |acc, x| acc + x);
        self.reshape_and_permute(Some(|shape: Vec<BTypes>|{
            shape.into_iter().enumerate().filter_map(|(i, x)|{
                if !all_ones[i]{
                    Some(x)
                }else{
                    None
                }
            }).collect_vec()
        }), None);

        all_ones.into_iter().any(|x| x)
    }

    fn simplify_merge_adjacent(&mut self){
        if self.shape_len() == 0{
            return 
        }
        let mut shapes = self.sts.iter().map(|x|x.shape()).collect_vec();
        let mut strides = self.sts.iter().map(|x| x.real_strides(false)).collect_vec();
        match &self.bufs[0]{
            BufferTypes::ConstBuffer(c) => {
                if let DTypes::ImageDType(d) = &c.dtype{
                    let base_shape = d.shape.clone();
                    let shape_idx_groups = get_contraction(&self.output_shape(), &base_shape.iter().map(|x|BTypes::Int(x.clone())).collect_vec());
                    let mut special_strides: Vec<BTypes> = vec![];
                    if let Some(s_i_g) =  shape_idx_groups{
                        s_i_g.into_iter().enumerate().for_each(|(i, g)|{
                            let shape_piece = g.iter().map(|x|self.output_shape()[*x]).collect_vec();
                            assert!(shape_piece.iter().fold(BTypes::Int(1), |acc, i|{
                                &acc*i
                            }) == base_shape[i]);
                            special_strides.extend(strides_for_shape(&shape_piece));
                        })
                    }
                    shapes.push(self.output_shape());
                    strides.push(special_strides.into_iter().map(|x|Some(x)).collect_vec())
                }
            },
            BufferTypes::MemBuffer(c) => {
                if let DTypes::ImageDType(d) = &c.dtype{
                    let base_shape = d.shape.clone();
                    let shape_idx_groups = get_contraction(&self.output_shape(), &base_shape.iter().map(|x|BTypes::Int(x.clone())).collect_vec());
                    let mut special_strides: Vec<BTypes> = vec![];
                    if let Some(s_i_g) =  shape_idx_groups{
                        s_i_g.into_iter().enumerate().for_each(|(i, g)|{
                            let shape_piece = g.iter().map(|x|self.output_shape()[*x]).collect_vec();
                            assert!(shape_piece.iter().fold(BTypes::Int(1), |acc, i|{
                                &acc*i
                            }) == base_shape[i]);
                            special_strides.extend(strides_for_shape(&shape_piece));
                        })
                    }
                    shapes.push(self.output_shape());
                    strides.push(special_strides.into_iter().map(|x|Some(x)).collect_vec())
                }
            },
            BufferTypes::LocalBuffer(c) => {
                if let DTypes::ImageDType(d) = &c.dtype{
                    let base_shape = d.shape.clone();
                    let shape_idx_groups = get_contraction(&self.output_shape(), &base_shape.iter().map(|x|BTypes::Int(x.clone())).collect_vec());
                    let mut special_strides: Vec<BTypes> = vec![];
                    if let Some(s_i_g) =  shape_idx_groups{
                        s_i_g.into_iter().enumerate().for_each(|(i, g)|{
                            let shape_piece = g.iter().map(|x|self.output_shape()[*x]).collect_vec();
                            assert!(shape_piece.iter().fold(BTypes::Int(1), |acc, i|{
                                &acc*i
                            }) == base_shape[i]);
                            special_strides.extend(strides_for_shape(&shape_piece));
                        })
                    }
                    shapes.push(self.output_shape());
                    strides.push(special_strides.into_iter().map(|x|Some(x)).collect_vec())
                }
            }
        }
        let mut rets = (0..shapes.len()).map(|j|vec![(shapes[j][0], strides[j][0])]).collect_vec();
        (1..shapes[0].len()).for_each(|i|{
            let mut can_merge = vec![];
            (0..shapes.len()).for_each(|j|{
                can_merge.push(
                    if strides[j][i].is_some() &&
                        ((strides[j][i].unwrap() != 0 && rets[j].last().unwrap().1 == Some(&shapes[j][i] * &strides[j][i].unwrap()))
                            || (strides[j][i].unwrap() == 0 && rets[j].last().unwrap().1 == Some(BTypes::Int(0))))
                {
                    true
                } else {
                    false
                });
            });
            let mergable = can_merge.into_iter().all(|x|x) && i != self.first_reduce();

            (0..shapes.len()).for_each(|j|{
                if mergable{
                    rets[j][rets[j].len()-1] = (&rets[j][rets[j].len()-1].0 * &shapes[j][i], strides[j][i])
                }else{
                    rets[j].push((shapes[j][i], strides[j][i]))
                }
            });
        });
        rets[..self.sts.len()].into_iter().enumerate().for_each(|(i, x)|{
            self.sts[i] = self.sts[i].reshape(&x.into_iter().map(|y|y.0).collect_vec())
        })
    }

    fn first_reduce(&self) -> usize{
        let mut result = vec![];
        for (x, y) in self.sts[0].shape()[..self.shape_len() - self.upcasted.clone()].iter().take(self.shape_len()-self.upcasted.clone()).zip(self.full_shape().iter().take(self.shape_len() - self.upcasted.clone())){
            result.push(x != y)
        };
        result.into_iter().position(|x| x==true).unwrap()
    }

    fn copy(&self) -> Self{
        let mut ret = self.clone();
        ret.bufs = ret.bufs.into_iter().filter_map(|x|{
            if let BufferTypes::LocalBuffer(l) = &x{
                Some(x)
            }else{
                None
            }
        }).collect_vec();
        ret.sts = ret.sts[..ret.bufs.len()].to_vec();
        ret.applied_opts_cache = None;
        ret
    }

    fn membufs(&self) -> Vec<MemBuffer>{
        self.bufs.iter().filter_map(|x|{
            if let BufferTypes::MemBuffer(m) = x{
                Some(m.clone())
            }else{
                None
            }
        }).collect_vec()
    }
    fn shape_offsets(&self, i: usize) -> Vec<BTypes>{
        if self.upcasted > 0{
            iproduct!(self.sts[i].shape()[self.shape_len() - self.upcasted.clone()..].into_iter().step_by(self.sts[i].shape()[self.shape_len() - self.upcasted.clone()..].len() - 1).map(|x| *x)).collect_vec()
        }else{
            vec![]
        }
    }

    fn float4_axis(&self, i:usize) -> Vec<usize>{
        self.sts[i].unit_stride_axes(false).iter().filter_map(|x|{
            if x >= &(self.shape_len()-self.upcasted.clone()) && &self.sts[i].shape()[x.clone()]%&BTypes::Int(4) == 0{
                Some(x - self.shape_len() - self.upcasted.clone())
            }else{
                None
            }
        }).collect_vec()
    }

    fn upcasted_axis(&self, i: usize) -> Vec<(isize, std::option::Option<BTypes>, bool)>{
        // ret x.0 has type sint rather than int and is being used as an index in a range
        self.sts[i].shape()[self.shape_len() - self.upcasted.clone()..].into_iter()
        .zip(self.sts[i].real_strides(false)[self.shape_len() - self.upcasted.clone() ..].into_iter())
        .zip({
            self.sts[0].shape()[self.shape_len() - self.upcasted.clone()..].into_iter()
            .zip(self.full_shape()[self.shape_len() - self.upcasted.clone() ..].into_iter())
            .map(|(x, y)| x!=y)
        }).map(|(((x, y), z))|{
            
            ({
                if let BTypes::Int(i) = *x{
                    i
                }else{
                    panic!()
                }
            }, *y, z)
        }).collect_vec()
    }

    // fn acc_offsets(&self, i: usize) -> Vec<isize>{
    //     if &self.upcasted == &0{
    //         return vec![0]
    //     }
    //     let upcased_i = self.upcasted_axis(i.clone());
    //     //potential bug here
    //     let acc_strides = strides_for_shape(&upcased_i.iter().step_by(upcased_i.len() -1).map(|(s, _, r)|{
    //         if r.clone(){
    //             BTypes::Int(1)
    //         }else{
    //             s.clone()
    //         }
    //     }).collect_vec()).into_iter().enumerate().map(|(i, x)|{
    //         x*(1-upcased_i.iter().step_by(upcased_i.len() - 1).cloned().collect_vec()[i].2)
    //     }).collect_vec();
    // }

    fn acc_offsets(&self, i: usize) -> Vec<BTypes> {
        if self.upcasted == 0 {
            return vec![BTypes::Int(0)];
        }
        
        let upcasted_i = self.upcasted_axis(i);
        // let acc_strides: Vec<BTypes> = strides_for_shape(&upcasted_i[..upcasted_i.len()-1].iter().cloned().map(|(s, _, r)|{
        //     if r{
        //         BTypes::Int(1)
        //     }else{
        //         s
        //     }
        // }).collect_vec()).into_iter().enumerate().map(|(i, x)|{
        //     x*()
        // });

        let acc_strides = strides_for_shape(&upcasted_i
        .iter()
        .rev()
        .map(|&(s, _, r)| if r { BTypes::Int(1)} else { BTypes::Int(s) })
        .collect_vec()).into_iter().enumerate().map(|(i,x)|{
            &x * &BTypes::Int(1 - {
                if upcasted_i[upcasted_i.len() - 1 - i].2{
                    1
                }else{
                    0
                }
            })
        }).collect_vec();
        
        // let shape = upcasted_i.iter().rev().map(|x| if x.2 { BTypes::Int(1) } else { x.0 }).collect::<Vec<_>>();
        // let strides_for_shape: Vec<BTypes> = strides_for_shape(&shape);
        
        // let product_iter = iproduct!(upcasted_i.iter().rev(), 0..);
        // // let offsets: Vec<isize> = product_iter
        // //     .map(|(x, _)| {
        // //         let inner_offsets: Vec<Vec<BTypes>> = upcasted_i.iter().enumerate()
        // //             .map(|(i, x)| (0..{
        // //                 match &x.0{
        // //                     BTypes::Int(i) => *i,
        // //                     BTypes::Node(_) => panic!()
        // //                 }
        // //             }).map(|y| &BTypes::Int(y) * &acc_strides[i]).collect::<Vec<BTypes>>())

        // //             .collect_vec();
        // //         inner_offsets
        // //     })
        // //     .flatten()
        // //     .map(|x| x.into_iter().fold(BTypes::Int(0), |acc, z| &acc+&z))
        // //     .collect();
        iproduct!(upcasted_i[..upcasted_i.len()-1].into_iter().enumerate().map(|(i, x)|{
            //x.0 should be an int but its marked as a sint
            (0..x.0).into_iter().map(|y|{
                &BTypes::Int(y)*&acc_strides[i]
            }).collect_vec()
        })).into_iter().map(|t|{
            t.iter().fold(BTypes::Int(0), |acc, x| &acc + x)
        })
        // .map(|x|{

        //     //Tingrad bug: this is inferred to be a sint
        //     if let BTypes::Int(i) = x {
        //         i
        //     }else{
        //         panic!()
        //     }
        // })
        .collect_vec()
    }
    fn get_float4_upcast_dim(&self, i: usize) -> Vec<usize>{
        let should_upcast = {
            if let Some(x) = &self.opts{
                x.supports_float4()
            }else{
                panic!()
            }
        } && (match &self.bufs[i]{
            BufferTypes::ConstBuffer(c) => DTYPES_DICT.borrow().get(&dtype::TYPES::float) == Some(&c.dtype) || DTYPES_DICT.borrow().get(&dtype::TYPES::half) == Some(&c.dtype) || match &c.dtype{
                DTypes::ImageDType(_) => true,
                _ => false
            },
            BufferTypes::LocalBuffer(l) => DTYPES_DICT.borrow().get(&dtype::TYPES::float) == Some(&l.dtype) || DTYPES_DICT.borrow().get(&dtype::TYPES::half) == Some(&l.dtype) || match &l.dtype{
                DTypes::ImageDType(_) => true,
                _ => false
            },
            BufferTypes::MemBuffer(m) => DTYPES_DICT.borrow().get(&dtype::TYPES::float) == Some(&m.dtype) || DTYPES_DICT.borrow().get(&dtype::TYPES::half) == Some(&m.dtype) || match &m.dtype{
                DTypes::ImageDType(_) => true,
                _ => false
            }
        });

        if should_upcast{
            return self.sts[i].unit_stride_axes(false).into_iter().filter_map(|x|{
                if x>= (self.shape_len() - self.upcasted.clone()) && self.sts[i].shape()[x] > BTypes::Int(1){
                    Some(x.clone())
                } else{
                    None
                }
            }).collect_vec()
        }else{
            vec![]
        }
    }



    fn full_unupcasted(&self) -> Vec<BTypes>{
        self.full_shape()[..&self.shape_len()-self.upcasted].to_vec()
    }



    fn upcast_in_mid_reduce_axes(&self) -> Vec<usize>{
        (self.first_reduce()..self.first_reduce() + self.group_for_reduces).filter_map(|j|{
            if &self.full_shape()[j] == &self.sts[0].shape()[j]{
                Some(j)
            }else{
                None
            }
        }).collect_vec()
    }

    fn global_dims(&self) -> usize{
        return self.first_reduce() -self.local_dims.clone()
    }

    fn colors(&self) -> Vec<String> {
        // first non local non reduce dims are global (blue)
        let mut colors = {vec!["blue".to_string(); self.global_dims()];
        if !self.dont_use_locals {
            vec!["blue".to_string(); self.global_dims()]
        }else{
            vec!["BLUE".to_string(); self.global_dims()]
        }};
        // after global are local_dims; warp ones used in tensor cores must be closest to first_reduce (cyan)
        for _ in 0..self.local_dims {
            colors.push("cyan".to_string());
        }
        // between first_reduce and first_reduce + group_for_reduces, they are either upcast mid reduce (white), or late upcasted (green)
        for i in self.first_reduce()..self.first_reduce() + self.group_for_reduces.clone() {
            if self.upcast_in_mid_reduce_axes().contains(&i) {
                colors.push("white".to_string());
            } else {
                colors.push("green".to_string());
            }
        }
        // between first_reduce + group_for_reduces and upcasted, they are reduce (red)
        let reduce_count = (self.shape_len() - self.upcasted.clone()) - (self.first_reduce() + self.group_for_reduces);
        for _ in 0..reduce_count {
            colors.push("red".to_string());
        }
        // upcasted dimensions are reduce (magenta) or normal (yellow)
        for i in (self.shape_len() - self.upcasted.clone())..self.shape_len() {
            if self.full_shape()[i] != self.sts[0].shape()[i] {
                colors.push("magenta".to_string());
            } else {
                colors.push("yellow".to_string());
            }
        }
        assert_eq!(colors.len(), self.shape_len(), "colors size mismatch");
        colors
    }
    
    fn colored_shape(&self, pad: Option<isize>, dense: bool) -> String{
        //pad: None, dense: False
        let mut ret = {
            self.full_shape().into_iter().map(|s|{
                if !dense{
                    match &s{
                        BTypes::Int(i) => format!("{:<4}", i),
                        BTypes::Node(n) => format!("{:?}", n.deref())
                    }
                }else{
                    format!("{:?}", s)
                }
            }).zip(self.colors().iter()).map(|(s, color)|{
                colored(s.as_str(), Some(&s.as_str()), false)
            }).join("_")
        };

        if let Some(p) = pad{
            ret = ret + " " +  &format!("{}", p - ansilen(&ret) as isize)
        }
        ret
    }


    // ********** base sims ****************

    fn upcast(&mut self){
        assert!(self.full_shape()[self.full_shape().len()-1] != BTypes::Int(1));
        self.upcasted += 1
    }

    fn shift_to(&mut self, axis: usize, amount: isize, top: bool, inset_before: Option<usize>){
        let mut ins_bf;

        if let None = inset_before{
            ins_bf = self.shape_len();
        }else{
            ins_bf = inset_before.unwrap();
        }
        let move_axis = {
            if top{
                axis
            }else{
                axis + 1
            }
        };
        if move_axis < ins_bf{
            ins_bf += 1;
        }
        self.reshape_and_permute(Some(|x: Vec<BTypes>|{
            let mut result = vec![];
            result.extend_from_slice(&x[0..axis.clone()]);
            if x[axis] > 1{
                let amt = BTypes::Int(amount);
                if top{
                    result.push(amt);
                    result.push(x[axis].floordiv(&amt, true));
                }else{
                    result.push(x[axis].floordiv(&amt, true));
                    result.push(amt)
                }
            }else{
                result.push(BTypes::Int(1));
                result.push(BTypes::Int(1));
            }
            result
        }), {
            let mut x: Vec<usize> = (0..ins_bf).filter_map(|i|{
                if i != move_axis{
                    Some(i)
                }else{
                    None
                }
            }).collect_vec();

            x.push(move_axis.clone());
            x.extend({
                (ins_bf..self.shape_len()+1).filter_map(|i|{
                    if i != (move_axis){
                        Some(i)
                    }else{
                        None
                    }
                })
            });
            Some(x)
        })
    }

    // ****************** comp sims ****************

    fn _limit_size<T>(&self, x: Vec<BTypes>, max_size: Vec<T>) -> Vec<BTypes>
        where BTypes: PartialOrd<T>
    {
        let mut new_shape = x;
        for i in (0.. new_shape.len()){
            let mut next_idx = (i + 1) % new_shape.len();
            while new_shape[i] > max_size[i]{
                new_shape[i] = new_shape[i].floordiv(&BTypes::Int(2), true);
                next_idx = {
                    if new_shape[next_idx] <= max_size[next_idx]{
                        next_idx
                    }else{
                        (next_idx + 1) % new_shape.len()
                    }
                };
                new_shape[next_idx] = &new_shape[next_idx] * &BTypes::Int(2);
            }
        }
        return new_shape
    }

    // fn limit_dims_to_max(&mut self, global_max: Vec<isize>, local_max: Vec<isize>){
    //     if self.global_dims() > 0{
    //         if !global_max.is_empty(){
    //              let mut tmp: Vec<isize> = global_max[..self.global_dims() as usize].to_vec();

    //             if !local_max.is_empty(){
    //                 tmp.extend_from_slice(&local_max[..self.local_dims.clone() as usize]);
    //             }

    //             if &BTypes::Int(*global_max.iter().max().unwrap()) < self.full_shape()[..self.global_dims() as usize].into_iter().max().unwrap(){
    //                 self.reshape_and_permute(Some(|x: Vec<BTypes>|{
                        
    //                     self._limit_size(x.into_iter().map(|x|{
    //                         if let BTypes::Int(i) = x{
    //                             i
    //                         }else{
    //                             panic!()
    //                         }
    //                     }), {
    //                         tmp.push(vec![MAX; self.full_shape().len() - tmp.len()]);
    //                         tmp
    //                     })
    //                 }).into_iter().map(|x| BTypes::Int(x)).collect_vec(), None)
    //             }
    //             assert!(global_max.iter().max() < self.full_shape()[..self.global_dims()].max());
    //             for i in (0..self.global_dims()-1){
    //                 if i < global_max.len() && self.full_shape()[i] > global_max[i]{
                        
    //                 }
    //             }
    //         }
    //     }
    // }

    fn limit_dims_to_max<T>(&mut self, global_max: Vec<isize>, local_max: Vec<isize>)
        where T: Fn(Vec<BTypes>) -> Vec<BTypes>
    {
        if self.global_dims() > 0{
            if !global_max.is_empty(){
                let mut tmp = global_max[..self.global_dims() ].to_vec();
                if !local_max.is_empty(){
                    tmp.extend_from_slice(&local_max[..self.local_dims])
                }else{
                    tmp.extend_from_slice(&vec![])
                }

                if global_max.iter().max().map(|x| &BTypes::Int(*x)) < self.full_shape()[..self.global_dims()].into_iter().max() {
                    // let tmp = tmp.clone(); // Clone tmp variable
                    self.reshape_and_permute(Some(move |x| {
                        let mut tmp = tmp.clone(); // Clone tmp again to modify it
                        tmp.extend_from_slice(&vec![MAX as isize; self.full_shape().len() - tmp.len()]);
                        self._limit_size(x, tmp)
                    }), None)
                }
                assert!(global_max.iter().max().map(|x| &BTypes::Int(x.clone())) >= self.full_shape()[..self.global_dims()].into_iter().max());
                for i in 0..self.global_dims() - 1{
                    if i < global_max.len() && self.full_shape()[i] > global_max[i]{
                        let mut order = (0..self.full_shape().len()).into_iter().collect_vec();
                        swap(&mut order[i], &mut order[self.global_dims() -1]);
                        self.reshape_and_permute::<T>(None, Some(order));
                        if helpers::DEBUG.clone().deref().lock().unwrap().value > 3{
                            println!("permuted global dim {:?} due to allocation exceeds global limit", order)
                        }
                    }
                }
            }
        }
    }

    fn alias_buffer(&mut self, i: usize, pattern: Vec<isize>){
        assert_eq!(pattern.len(), self.sts[i].shape().len());

        let mut bst = BTypes::Int(1);
        let real_strides = self.sts[i].real_strides(false);
        let shp = self.sts[i].shape().into_iter().zip(pattern.iter()).map(|(s, p)|{
            if p != &0{
                s
            }else{
                BTypes::Int(1)
            }
        }).collect_vec();
        let mut stride = vec![BTypes::Int(0); pattern.len()];
        for priority in (1.. pattern.iter().max().unwrap() + 1){
            for (j, p) in pattern.into_iter().enumerate(){
                if priority == p && real_strides[j] != Some(BTypes::Int(0)){
                    stride[j] = bst;
                    bst = &shp[j] * &bst
                }
            }
        }
        self.sts.push(ShapeTracker::new(vec![Rc::new(View::create(&shp, Some(&stride), BTypes::Int(0), None))]));
        self.bufs.push(BufferTypes::LocalBuffer(LocalBuffer::new(format!("ldata{}", i), self.sts[self.sts.len() - 1].size(), DTYPES_DICT.clone().get(&dtype::TYPES::float32).unwrap().clone())));
        if helpers::DEBUG.clone().deref().lock().unwrap().borrow().value >= 4{
            println!("aliasin buffer {:?}", self.sts[i]);
        }
        self.local_alias[&(i as isize)] = self.bufs.last().unwrap().clone()
    }

    fn _apply_tc_opt(&mut self, use_tensor_cores: isize, axis: usize, opt_level: isize) -> bool{
        if use_tensor_cores > 0 && self.opts.is_some() && self.opts.unwrap().has_local && self.reduceop.is_some() && self.reduceop.unwrap().op == Op::ReduceOps(ReduceOps::SUM) && TENSOR_CORES.clone().contains_key(&self.opts.unwrap().device){
            for tc in &TENSOR_CORES.clone()[&self.opts.unwrap().device]{
                let has_cast = tc.dtype_in != tc.dtype_out;

                if has_cast && !(self.reduceop.unwrap().src[0].op == Op::UnaryOps(UnaryOps::CAST) && self.reduceop.unwrap().src[0].arg == Some(Items::Dtype(tc.dtype_out.clone()))){
                    continue
                }

                let mul_op = {
                    if has_cast{
                        self.reduceop.unwrap().src[0].src[0]
                    }else{
                        self.reduceop.unwrap().src[0]
                    }
                };

                let buf_index = |src: &LazyOp| -> Option<usize>{
                    if src.op == Op::BufferOps(BufferOps::LOAD)&& {
                        match &src.arg{
                            Some(Items::Buffer(b)) => {
                                match b{
                                    BufferTypes::ConstBuffer(c) => c.dtype == tc.dtype_in,
                                    BufferTypes::LocalBuffer(c) => c.dtype == tc.dtype_in,
                                    BufferTypes::MemBuffer(c) => c.dtype == tc.dtype_in
                                }
                            },
                            _ => panic!()
                        }
                    } {
                        return self.bufs.iter().position(|x| {

                            if let Some(Items::Buffer(b)) = &src.arg{
                                b == x
                            }else{
                                false
                            }
                        })
                    }

                    if opt_level >= 1 && src.op == Op::UnaryOps(UnaryOps::CAST) && {
                        match &src.arg{
                            Some(Items::Dtype(d)) => d == &tc.dtype_in,
                            _ => panic!()
                        }
                    }{
                         return self.bufs.iter().position(|x|{
                            match &src.src[0].arg{
                                Some(Items::Buffer(b)) => {
                                    b == x
                                },
                                _ => false
                            }
                        })
                    }
                    return None
                }


                let buf0 = buf_index(&mul_op.src[0]);
                let buf1 = buf_index(&mul_op.src[1].deref());
                if buf0.is_none() || buf1.is_none(){
                    continue;
                }

                let buf0_strides = self.sts[buf0.unwrap()].real_strides(false);
                let buf1_strides = self.sts[buf1.unwrap()].real_strides(false);
                let reduce_sz = self.full_shape()[self.first_reduce()];

                let axis_buf0 = buf0_strides[..self.first_reduce()].iter().enumerate().filter_map(|(i, s)|{
                    if s.unwrap()==0 && &self.full_shape()[i]%&tc.dims[0] == 0{
                        Some((i, self.full_shape()[i], buf1_strides[i]))
                    }else{
                        None
                    }
                }).collect_vec();
                
                let axis_buf1 = buf1_strides[..self.first_reduce()].iter().enumerate().filter_map(|(i, s)|{
                    if s.unwrap()==0 && &self.full_shape()[i]%&tc.dims[0] == 0{
                        Some((i, self.full_shape()[i], buf1_strides[i]))
                    }else{
                        None
                    }
                }).collect_vec();

                if !(axis_buf0.is_empty() && axis_buf1.is_empty() && &reduce_sz % &tc.dims[2] == 0 && reduce_sz >= tc.dims[2]){
                    continue;
                }

                if !(self.shape_len() - self.first_reduce() == 1 || opt_level>=1){
                    continue;
                }

                let axis_choices = iproduct!(axis_buf0, axis_buf1).collect_vec();
                if !(axis < axis_choices.len()){
                    continue;
                }

                let s0 = axis_choices[(axis_choices.len()-axis + 1) as usize].0.0;
                let s1 = axis_choices[(axis_choices.len() -axis + 1) as usize].1.0;

                assert!(s0 != s1 && &self.full_shape()[s0] % &tc.dims[0] == 0 && &self.full_shape()[s1] % &tc.dims[1] == 0);

                if DEBUG.clone().lock().unwrap().value.borrow() >= &3{
                    print!("TENSOR CORES {:?} {:?} {:?}", axis_buf0, axis_buf1, tc)
                }
                let tc_opts = TensorCoreOptions{
                    bufs: (buf0.unwrap(), buf1.unwrap()),
                    axes: vec![s0, s1],
                    axes_exist: vec![true, true]
                };

                self.tensor_core_opts = Some(tc_opts);
                if self.apply_opt(Opt{
                    op: OptOps::UNROLL,
                    axis: Some(0),
                    amt: Some(tc.dims[2].clone())
                }, false).is_err(){
                    return false
                }else{
                    let mut sizes = Vec::new();

                    for dim in 0..2{
                        let thread_sizes: isize = tc.threads.iter().filter(|x|x.0 == dim).map(|x| x.1).product();
    
                        sizes.push(thread_sizes);
                        
    
                    }
    
                    for (i, &sz) in sizes.iter().enumerate(){
                        if tc.dims[i] > sz{
                            self.apply_opt(Opt { op: OptOps::UPCAST, axis: Some(tc_opts.axes[i]), amt: Some(tc.dims[i] / sz) }, false);
                        }
                    }
                    for (tc_dim, tc_amt) in tc.threads{
                        self.apply_opt(Opt { op: OptOps::LOCAL, axis: Some(tc_opts.axes[tc_dim as usize]), amt: Some(tc_amt) }, false);
                    }
    
                    if use_tensor_cores == 1{
                        self.tensor_core = Some(*tc);
                    }
                    return true
                }
            }
        }
        
        true
    }

    fn apply_tensor_cores(&mut self, use_tensor_cores: usize, extra_opts: Option<Vec<Opt>>) -> bool{
        if !(self.opts.clone().unwrap().hash_tensor_cores && use_tensor_cores != 2){
            return false
        }
        
        if self.apply_opt(Opt { op: OptOps::TC, axis: Some(0), amt: Some(0) }, true).is_ok(){
            let tc_opts = self.tensor_core_opts;
            if let Some(tc_o) = &tc_opts{
                if let Some(e_o) = extra_opts{
                    for opt in e_o{
                        if self.apply_opt(opt, true).is_err(){
                            return false
                        }else{
                            continue;
                        }
                    }
                    return true
                }else{

                    if tc_o.axes_exist[1]{
                        let ax_div: isize = {
                            (5..0).into_iter().filter_map(|x|{
                                if &self.full_shape()[tc_o.axes[1]] % &x == 0{
                                    Some(x)
                                }else{
                                    None
                                }
                            }).collect_vec()[0]
                        };
                        if ax_div != 1{
                            if self.apply_opt(Opt{
                                op: OptOps::UPCAST,
                                axis: Some(tc_o.axes[1]),
                                amt: Some(ax_div)
                            }, true).is_err(){
                                return false;
                            }
                        }
                    }

                    if tc_o.axes_exist[0]{
                        let ax_div: isize = {
                            (5..0).into_iter().filter_map(|x|{
                                if &self.full_shape()[tc_o.axes[0]] % &x == 0{
                                    Some(x)
                                }else{
                                    None
                                }
                            }).collect_vec()[0]
                        };
                        if ax_div != 1{
                            if self.apply_opt(Opt{
                                op: OptOps::UPCAST,
                                axis: Some(tc_o.axes[0]),
                                amt: Some(ax_div)
                            }, true).is_err(){
                                return false
                            }
                        }
                    }

                    if self.tensor_core.is_some() && tc_o.axes_exist[0]{
                        for upc in [4, 2]{
                            if &self.full_shape()[tc_o.axes[0]] % &upc == 0{
                                self.apply_opt(Opt { op: OptOps::LOCAL, axis: Some(tc_o.axes[0]), amt: Some(upc) }, true);
                                break;
                            }
                            continue;
                        }
                    }
                }
            }
            return true
        }else{
            false
        }


    }
    fn apply_opt<'a>(&mut self, opt: Opt, append_opt: bool) -> Result<String, String>{
        assert!(!self.dont_use_locals || vec![OptOps::LOCAL, OptOps::GROUP, OptOps::GROUPTOP, OptOps::UPCASTMID].contains(&opt.op));

        let mut axis: usize;
        if opt.axis.is_none(){
            axis = opt.axis.clone().unwrap();
            if opt.op == OptOps::UNROLL{
                 axis = axis + self.first_reduce();
            }else if vec![OptOps::GROUP, OptOps::GROUPTOP].contains(&opt.op){
                axis = self.first_reduce() + axis;
            }else{
                axis = axis + 0;
            }
        }else{
            panic!()
        }

        assert!(axis < self.full_shape().len());

        let mut amt;
        if opt.amt.is_some(){
            amt = {
                if opt.amt != Some(0){
                    opt.amt.clone().unwrap()
                }else{
                    match self.full_shape()[axis]{
                        BTypes::Int(i) => i,
                        _ => panic!()
                    }
                }
            };

            assert!(amt != 1);

            if opt.op != OptOps::PADTO{
                assert!(&self.full_shape()[axis] % &amt ==  0)
            }
        }else{
            amt = -1;
        }

        if self.reduceop.is_some() && vec![OptOps::GROUP, OptOps::GROUPTOP].contains(&opt.op) || self.group_for_reduces == 0 && !vec![OptOps::NOLOCALS, OptOps::PADTO].contains(&opt.op){
            let acc_sz = {
                match get_lazyop_info(self.reduceop.unwrap()).dtype{
                    DTypes::ImageDType(dt) => dt.base.itemsize,
                    DTypes::DType(dt) => dt.itemsize,
                    DTypes::PtrDType(dt) => dt.itemsize
                }
            };

            let upcast_sz: BTypes = self.full_shape()[self.shape_len() - self.upcasted..].into_iter().product();
            let local_sz: BTypes = self.full_shape()[self.first_reduce() - self.local_dims.. self.first_reduce()+ self.group_for_reduces].into_iter().product();
            assert!(&(&BTypes::Int(amt * acc_sz)* &upcast_sz)*&local_sz <= BTypes::Int(self.opts.clone().unwrap().shared_max as isize));
        }

        match &opt.op{
            OptOps::TC => {
                assert!(self.applied_opts.len() == 0);
                assert!(opt.axis.is_some() && opt.amt.is_some());
                let use_tensor_cores = getenv("TC".to_string(), 1.to_string());
                assert!(use_tensor_cores == 2.to_string() || self.opts.clone().unwrap().hash_tensor_cores);
                assert!(self._apply_tc_opt(use_tensor_cores.parse().unwrap(), opt.axis.unwrap(), opt.amt.unwrap()));
    
                self.applied_opts.push(opt);
    
                
            },
            OptOps::LOCAL => {
                assert!(self.opts.clone().unwrap().has_local);
                assert!(axis < self.global_dims());
                self.shift_to(axis, amt, false, Some(self.first_reduce() - self.local_dims));
                self.local_dims+=1;
                
            },
            OptOps::GROUP | OptOps::GROUPTOP => {
                assert!({
                    match &self.opts{
                        Some(x) => x.has_local && x.has_shared,
                        None => return Err(String::new())
                    }
                });

                assert!(axis >= (self.first_reduce() + self.group_for_reduces.clone()) && axis < (self.shape_len()-self.upcasted.clone()));
                assert!(self.tensor_core.is_none());
                self.shift_to( axis, amt, {opt.op == OptOps::GROUPTOP}, Some(self.first_reduce() - self.local_dims));
                self.group_for_reduces += 1;
                
            },
            OptOps::UNROLL => {
                assert!(axis < (self.shape_len()-self.upcasted));
                assert!(amt <= 32);
                if self.full_shape()[axis] == amt && axis == self.first_reduce(){
                    self.local_dims += 1;
                }
                if self.full_shape()[axis] == amt && axis < (self.first_reduce() + self.group_for_reduces){
                    self.group_for_reduces -= 1;
                }
                self.shift_to(axis, amt, false, None);
                self.upcast();
                
            },
            OptOps::UPCAST => {
                assert!(axis < self.first_reduce());
                assert!(!(self.tensor_core.is_some() && axis >= (self.first_reduce() - self.tensor_core.clone().unwrap().threads.len())));
                assert!(amt <= 8);
                self.shift_to(axis, amt, false, None);
                self.upcast();
                
            },
            OptOps::UPCASTMID => {
                assert!({
                    match &self.bufs[0]{
                        BufferTypes::ConstBuffer(c) => {
                            match &c.dtype{
                                DTypes::DType(d) => d.name.starts_with("image"),
                                DTypes::ImageDType(d) => d.name.starts_with("image"),
                                DTypes::PtrDType(d) => d.name.starts_with("image")
                            }
                        },
                        BufferTypes::LocalBuffer(c) => {
                            match &c.dtype{
                                DTypes::DType(d) => d.name.starts_with("image"),
                                DTypes::ImageDType(d) => d.name.starts_with("image"),
                                DTypes::PtrDType(d) => d.name.starts_with("image")
                            }
                        },
                        BufferTypes::MemBuffer(c) => {
                            match &c.dtype{
                                DTypes::DType(d) => d.name.starts_with("image"),
                                DTypes::ImageDType(d) => d.name.starts_with("image"),
                                DTypes::PtrDType(d) => d.name.starts_with("image")
                            }
                        },
                    }
                } && ! self.float4_axis(0).is_empty() && self.group_for_reduces != 0 && self.first_reduce() <= 2 && self.sts[0].shape().iter().product::<BTypes>() > 1);

                let axes = self.sts[0].unit_stride_axes(false);
                assert!(axes.len() == 1);
                assert!(axes[0] == axis);
                assert!(amt == 4);
                self.shift_to( axis, amt, false, Some(self.first_reduce() + self.group_for_reduces));
                self.group_for_reduces += 1;
            },
            OptOps::NOLOCALS => {
                assert!(self.opts.clone().unwrap().has_local && !self.dont_use_locals);
                assert!(self.local_dims == 0 && self.group_for_reduces == 0);
                self.dont_use_locals = true;
            },
            OptOps::PADTO => {
                assert!(!self.vars.is_empty());
                assert!(axis < self.first_reduce());
                let mut padded = false;
                for (i, st) in self.sts.iter().enumerate(){
                    assert!(self.sts[i].shape()[axis] > amt/2);
                    let ru: BTypes = &round_up(self.sts[i].shape()[axis], amt) - &self.sts[i].shape()[axis];
                    self.sts[i] = {
                        let mut vec = vec![(BTypes::Int(0), BTypes::Int(0)); axis];
                        vec.push((BTypes::Int(0), ru));
                        vec.extend_from_slice(&vec![(BTypes::Int(0), BTypes::Int(0)); (st.shape().len() - axis - 1)]);

                        st.pad(&vec)
                        
                    };
                }
                padded = true;
                assert!(padded);
            }
        }
        if append_opt{
            self.applied_opts.push(opt);
            if let Some(x) = &mut self.tensor_core_opts{
                x.fix_axes(axis);
            }
        }
        Ok(String::new())
    }

    fn required_optimizations(&mut self) -> Result<String, String>{
        match &self.bufs[0]{
            BufferTypes::ConstBuffer(c) => {
                if let DTypes::ImageDType(i) = &c.dtype{
                    let unit_stride_axes_mul_4 = self.sts[0].unit_stride_axes(true).into_iter().filter_map(|x|{
                        if &self.sts[0].shape()[x]% &BTypes::Int(4) == 0{
                            Some(x)
                        }else{
                            None
                        }
                    }).collect_vec();

                    assert!(unit_stride_axes_mul_4.len() >= 1);
                    if !unit_stride_axes_mul_4.is_empty() && unit_stride_axes_mul_4.iter().all(|x| x < &(self.shape_len() - self.upcasted.clone())){
                        return self.apply_opt(Opt{
                            op: OptOps::UPCAST,
                            axis: Some(unit_stride_axes_mul_4[0]),
                            amt: Some(4)
                        }, true);
                    }else{
                        Err(String::new())
                    }
                }else{
                    Err(String::new())
                }
            },
            _ => Err(String::new())
        }
    }

    fn hand_coded_optimizations(&mut self) -> Result<String, String>{
        let _ = self.required_optimizations();
        let mv_blocksize: usize = getenv("MV_BLOCKSIZE".to_string(), 4.to_string()).parse().unwrap();
        let mv_threads_per_row: usize = getenv("MV_THREADS_PER_ROW".to_string(), 8.to_string()).parse().unwrap();
        let mv_rows_per_thread:usize = getenv("MV_ROWS_PER_THREAD".to_string(), 4.to_string()).parse().unwrap();
        let mulop;
        if {
            if let Some(x) = &self.opts{
                x.has_local.clone()
            }else{
                return Err(String::new())
            }
        } && getenv("MV".to_string(), 1.to_string()).parse().unwrap() != 0 && (mv_blocksize > 1 || mv_threads_per_row > 1 || mv_rows_per_thread > 1)  && {
            if let Some(x) = &self.reduceop{
                x.op == Op::ReduceOps(ReduceOps::SUM)
            }else{
                false
            }
        } && self.full_shape().len() >= 2 && {
            if let Some(x) = &self.opts{
                x.has_shared.clone()
            }else{
                return Err(String::new())
            }
        } && {
            if let Some(x) = &self.reduceop{
                mulop = x.src[0].clone();
                mulop.op == Op::BinaryOps(BinaryOps::MUL) && mulop.src[0].op == Op::BufferOps(BufferOps::LOAD) && mulop.src[1].op == Op::BufferOps(BufferOps::LOAD)
            }else{
                return Err(String::new())
            }
        }{
            let st0: Rc<ShapeTracker> = self.sts[self.bufs.iter().position(|x| {
                if let Some(Items::Buffer(c)) = &mulop.src[0].arg{
                    c == x
                }else{
                    false
                }
            }).unwrap()];

            let st1: Rc<ShapeTracker> = self.sts[self.bufs.iter().position(|x| {
                if let Some(Items::Buffer(c)) = &mulop.src[1].arg{
                    c == x
                }else{
                    false
                }
            }).unwrap()];

            let strides0 = st0.real_strides(false);
            let strides1 = st1.real_strides(false);

            let has_expanded_axis = |shape: &Vec<BTypes>, strides: &Vec<Option<BTypes>>| -> bool{
                shape.iter().zip(strides.iter()).any(|(s, st)|{
                    s > &BTypes::Int(1) && st == &Some(BTypes::Int(0))
                })
            };

            if strides0[self.first_reduce()] == Some(BTypes::Int(1)) && !(has_expanded_axis(&st0.shape(), &strides0) && has_expanded_axis(&st1.shape(), &strides1)){
                for global_idx in 0..self.global_dims(){
                    if &self.full_shape()[self.first_reduce()] % &BTypes::Int(mv_threads_per_row.clone() as isize) == BTypes::Int(0) && &self.full_shape()[global_idx] % &BTypes::Int((mv_blocksize & mv_rows_per_thread) as isize) == BTypes::Int(0){
                        if DEBUG.clone().lock().unwrap().value >= 3{
                            println!()
                        }
                        if mv_threads_per_row > 1{
                            self.apply_opt(Opt { op: OptOps::GROUP, axis: Some(0), amt: Some(mv_threads_per_row as isize) }, true);
                        }
                        if mv_blocksize > 1{
                            self.apply_opt(Opt { op: OptOps::LOCAL, axis: Some(global_idx), amt: Some(mv_blocksize as isize) }, true);
                        }
                        if mv_rows_per_thread > 1{
                            self.apply_opt(Opt { op: OptOps::UPCAST, axis: Some(global_idx), amt: Some(mv_rows_per_thread as isize) }, true);
                        }
                        return Ok(String::new())
                    }
                }
            }
        }
        if {
            if let Some(x) = &self.opts{
                x.has_local.clone() && x.has_shared
            }else{
                return Err(String::new())
            }
        } && all_int(&self.sts[0].shape()[..self.first_reduce()]){
            if !self.float4_axis(0).is_empty() && self.first_reduce() <= 2 && self.first_reduce() + 1 <= self.shape_len() && self.sts[0].shape()[..self.first_reduce()].iter().product() <= BTypes::Int(2048){
                for sz in {
                    if self.sts[0].shape()[..self.first_reduce()].iter().product() <= BTypes::Int(32){
                        vec![BTypes::Int(256), BTypes::Int(16)]
                    }else{
                        vec![BTypes::Int(16)]
                    }
                }{
                    if self.sts.iter().all(|st|{
                        &st.shape()[self.first_reduce()] % &sz == BTypes::Int(0) || st.shape()[self.first_reduce()] == BTypes::Int(1)
                    }){
                        if let Err(x) = self.apply_opt(Opt { op: OptOps::GROUPTOP, axis: Some(0), amt: Some({
                            match sz{
                                BTypes::Int(i) => i,
                                _ => return Err(String::new())
                            }
                        }) }, true){
                            return Err(String::new())
                        }
                        break;
                    }
                }
            }
            if {
                match &self.bufs[0]{
                    BufferTypes::ConstBuffer(c) => c.dtype.name().starts_with("image"),
                    BufferTypes::LocalBuffer(c) => c.dtype.name().starts_with("image"),
                    BufferTypes::MemBuffer(c) => c.dtype.name().starts_with("image"),
                }
            } && !self.float4_axis(0).is_empty() && self.first_reduce() <= 2 && self.sts[0].shape().iter().product() > BTypes::Int(1){
                let axes = self.sts[0].unit_stride_axes(false);
                assert!(axes.len() == 1);
                if &self.sts[0].shape()[axes[0]] % &BTypes::Int(4) == BTypes::Int(0){
                    self.apply_opt(Opt { op: OptOps::UPCASTMID, axis: Some(axes[0]), amt: Some(4) }, true);
                }
            }
        }

        for (buf_index, buf) in self.bufs.iter().enumerate(){
            let unit_stride_axes_mul_4 = self.sts[buf_index].unit_stride_axes(true).into_iter().filter_map(|i|{
                if &self.sts[buf_index].shape()[i] % &BTypes::Int(4) == 0{
                    Some(i)
                }else{
                    None
                }
            }).collect_vec();

            if {
                match buf{
                    BufferTypes::ConstBuffer(c) => {
                        if let DTypes::ImageDType(_) = &c.dtype{
                            true
                        }else{
                            false
                        }
                    },
                    BufferTypes::LocalBuffer(c) => {
                        if let DTypes::ImageDType(_) = &c.dtype{
                            true
                        }else{
                            false
                        }
                    },
                    BufferTypes::MemBuffer(c) => {
                        if let DTypes::ImageDType(_) = &c.dtype{
                            true
                        }else{
                            false
                        }
                    },
                }
            }{
                if unit_stride_axes_mul_4.is_empty() && unit_stride_axes_mul_4.iter().all(|x|{
                    x < &(self.shape_len() - self.upcasted)
                }) && self.upcast_in_mid_reduce_axes().contains(&unit_stride_axes_mul_4[0]){
                    if unit_stride_axes_mul_4[0] < self.first_reduce(){
                        self.apply_opt(Opt { op: OptOps::UPCAST, axis: Some(unit_stride_axes_mul_4[0]), amt: Some(4) }, true);
                    }else{
                        self.apply_opt(Opt { op: OptOps::UNROLL, axis: Some(unit_stride_axes_mul_4[0] - self.first_reduce()), amt:Some(4) }, true);
                    }
                }
            }
        }
        if self.group_for_reduces == 0{
            return Ok(String::new())
        }

        let mut to_upcast = vec![];

        for axis in 0..self.first_reduce(){
            if let BTypes::Int(i) = &self.full_shape()[axis]{
                if i <= &7 && self.sts.iter().any(|st|{
                    st.axis_is_masked(axis)
                }) && &self.full_shape()[self.shape_len() - self.upcasted] * &to_upcast.iter().map(|j| &self.full_shape()[*j]).product() * &self.full_shape()[axis] <= BTypes::Int(49){
                    if DEBUG.clone().lock().unwrap().value >= 4{
                        println!()
                    }
                    to_upcast.push(axis);
                }
            }
        }

        for axis in to_upcast.into_iter().rev(){
            self.apply_opt(Opt { op: OptOps::UPCAST, axis: Some(axis), amt: Some(0) }, true);
        }


        let mut upcasted_axis = HashSet::new();

        while self.sts[0].shape()[..self.first_reduce()].into_iter().product() >= BTypes::Int(1024){
            let mut xb_choices = vec![];
            for (axis, upcast_amount) in iproduct!(0..self.first_reduce(), vec![3,4]){
                if !upcasted_axis.contains(&axis) && {
                    if let BTypes::Int(_) = self.full_shape()[axis]{
                        true
                    }else{
                        false
                    }
                } && &self.full_shape()[axis] % &upcast_amount == BTypes::Int(0) && {
                    self.sts.iter().enumerate().any(|(buf_index, st)|{
                        st.views[st.views.len() - 1].strides[axis] == BTypes::Int(0) && !{
                            self.upcasted_axis(buf_index).iter().any(|x| x.1 == Some(BTypes::Int(0)))
                        }
                    })
                }{
                    xb_choices.push((self.sts.iter().map(|st|{
                        st.views[st.views.len() - 1].strides[axis] > 0
                    }).sum(), self.sts.iter().map(|st| st.views[st.views.len() - 1].strides[axis]).sum(), axis, upcast_amount));


                }
            }

            if !xb_choices.is_empty(){
                xb_choices.sort();

                if DEBUG.clone().lock().unwrap().value >= 4{
                    println!()
                }
                self.apply_opt(Opt { op: OptOps::UPCAST, axis: Some(xb_choices[0].2), amt:  Some(xb_choices[0].3)}, true);
                upcasted_axis.insert(xb_choices[0].2);
            }else{
                break;
            }
        }

        if self.first_reduce() < self.shape_len() - self.upcasted && self.shape_offsets(self.full_buf_index).len() <= 4 || !{
            self.upcasted_axis(self.full_buf_index).into_iter().any(|(_,_,r)| r)
        } && (self.upcasted == 0 || self.full_shape()[self.full_shape().len() - self.upcasted..].iter().product() < BTypes::Int(64)){
            let s = self.full_unupcasted()[self.full_unupcasted().len() - 1];
            if s <= BTypes::Int(32){
                if let BTypes::Int(s_i) = s{
                    self.apply_opt(Opt { op: OptOps::UNROLL, axis: Some(self.full_unupcasted().len() - 1 - self.first_reduce()), amt: Some(0) }, true);
                    let s2 = self.full_unupcasted()[self.full_unupcasted().len() - 1];
                    if self.first_reduce() < (self.shape_len() - self.upcasted) && s <= BTypes::Int(3) && s2 <=3{
                        if let BTypes::Int(i_s2) = s2{
                            self.apply_opt(Opt { op: OptOps::UNROLL, axis: Some(self.full_unupcasted().len() - 1 - self.first_reduce()), amt: Some(0) }, true);
                        }
                    }
                }else{
                    if &self.full_unupcasted()[self.full_shape().len() - 1] % &BTypes::Int(4) == BTypes::Int(0){
                        self.apply_opt(Opt { op: OptOps::UNROLL, axis: Some(self.full_unupcasted().len() - 1 -self.first_reduce()), amt: Some(4) }, true);
                    }
                }
            }
        }

        if self.upcasted == 0 && !self.full_unupcasted().is_empty() && &self.full_unupcasted()[self.full_unupcasted().len() - 1] % &BTypes::Int(4) == BTypes::Int(0){
            self.apply_opt(Opt { op: OptOps::UPCAST, axis: Some(self.full_unupcasted().len() - 1), amt: Some(4) }, true);
        }

        if {
            if let Some(x) = &self.opts{
                x.has_local.clone()
            }else{
                return Err(String::new())
            }
        }{
            if !(&getenv("NOLOCALS".to_string(), "".to_string()) == "") && self.local_dims == 0 && !self.group_for_reduces < 0{
                self.apply_opt(Opt { op: OptOps::UPCAST, axis: None, amt: None }, true);
            }else{
                let local_axis_ranking = (0..self.full_shape()[..self.first_reduce()].len()).into_iter().map(|axis|{
                    ({
                        (0..self.sts.len()).any(|buf_index| self.sts[buf_index].views.last().map_or(false, |view| view.strides[axis] == 0))
                    }, axis)
                }).collect_vec();

                let mut to_local: Vec<(usize, usize)> = vec![];
                local_axis_ranking.sort_by_key(|x|({
                    if x.0 == true{
                        1
                    }else{
                        0
                    }
                }, -(x.1.clone() as isize)));
                for (_, axis) in local_axis_ranking{
                    let local_size = to_local.iter().map(|(_, sz)| sz).product();
                    let local_sz: Option<usize> = {
                        let mut local_sz: Option<usize> = None;
                        let candidates: Vec<isize> = if axis == 0 {
                            vec![32]
                        } else {
                            vec![16, 8, 4, 3, 2]
                        };
                        
                        for x in candidates {
                            if &self.full_shape()[axis] % &x == 0 && local_size * x <= 128 {
                                local_sz = Some(x as usize);
                                break;
                            }
                        }
                        
                        local_sz
                    };

                    if let Some(x) = local_sz{
                        to_local.push((axis, x));
                    }
                    continue;
                }

                let mut deleted_shape = 0;
                let pl = to_local.into_iter().take(3).collect_vec();
                pl.sort();

                for(axis, l_sz) in &mut pl{
                    *axis = *axis - deleted_shape;

                    let will_delete_shape = BTypes::Int(l_sz.clone() as isize) == self.full_shape()[*axis];

                    self.apply_opt(Opt { op: OptOps::LOCAL, axis: Some(*axis), amt: Some(*l_sz as isize) }, true);
                    if will_delete_shape{
                        deleted_shape += 1
                    }
                    continue;
                }
            }
        }
        Ok(String::new())
    }
}

    fn full_shape(sts: &Vec<Rc<ShapeTracker>>, full_buf_index: usize) -> Vec<BTypes>{
        sts[full_buf_index].clone().shape()
    }

    fn output_shape(sts: &Vec<Rc<ShapeTracker>>) -> Vec<BTypes>{
        return sts[0].clone().shape()
    }

    fn reshape_and_permute<T>(sts: &mut Vec<Rc<ShapeTracker>>, new_shape_fxn: Option<T>, axis: Option<Vec<usize>>)
        where
            T: Fn(Vec<BTypes>) -> BTypes
    {
        let mut new_sts = vec![];
        sts.into_iter().for_each(|mut st|{
            if let Some(x) = new_shape_fxn{
                st = st.reshape(&vec![x(st.shape())]).borrow_mut();
            }
            if let Some(x) = axis{
                st = st.permute(&x.into_iter().map(|y| y).collect_vec()).borrow_mut()
            }
            new_sts.push(st.clone())
        });
        *sts = new_sts;
    }

    fn shape_len(sts: &Vec<Rc<ShapeTracker>>) -> usize{
        sts[0].shape().len()
    }