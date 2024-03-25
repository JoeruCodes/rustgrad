use std::{any::Any, borrow::{Borrow, BorrowMut}, collections::{HashMap, HashSet}, fmt::{Debug, Display}, isize::MAX, ops::Deref, rc::Rc, sync::Arc};

use itertools::{iproduct, Itertools};
use lazy_static::lazy_static;
use num::Integer;

use crate::rustgrad::{device::Device, dtype::{self, DTypes, ImageDType, DTYPES_DICT}, helpers::{ansilen, colored, dedup, flatten, get_contraction}, ops::{get_lazyop_info, BufferOps, BufferTypes, FlopCounter, Items, LazyOp, MemBuffer, Op}, shape::{shapetracker::ShapeTracker, sym::{BTypes, NodeTypes}, view::strides_for_shape}};

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
    axis: Option<isize>,
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

#[derive(Clone)]
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
    bufs: (isize, isize),
    axes: Vec<isize>,
    axes_exist: Vec<bool>
}

impl TensorCoreOptions{
    fn fix_axes(&mut self, removed_axis: isize){
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
    full_buf_index: isize,
    sts: Vec<Rc<ShapeTracker>>,
    applied_opts: Vec<Opt>,
    group_for_reduces: isize,
    upcasted: isize,
    local_dims: isize,
    local_alias: HashMap<isize, LocalBuffer>,
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

        let reduce = full_shape(&sts, full_buf_index.unwrap_or(0)).iter().zip(Self::output_shape(&sts).iter()).enumerate().collect_vec();
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
        let group_for_reduces:isize = 0;
        let upcasted: isize = 0;
        let local_dims: isize = 0;
        let local_alias: HashMap<isize, LocalBuffer> = HashMap::new();
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
            full_buf_index: full_buf_index.unwrap() as isize,
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


    }


    fn full_shape(&self) -> Vec<BTypes>{
        self.sts[self.full_buf_index.clone() as usize].clone().shape()
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
                st = st.permute(&x.into_iter().map(|y| y as isize).collect_vec())
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
        self.local_dims -= all_ones[self.first_reduce() - self.local_dims as usize .. self.first_reduce()].iter().sum();
        self.upcasted -= all_ones[self.shape_len() -self.upcasted.clone() as usize ..].iter().sum();
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
        for (x, y) in self.sts[0].shape()[..self.shape_len() - self.upcasted.clone() as usize].iter().take(self.shape_len()-self.upcasted.clone() as usize).zip(self.full_shape().iter().take(self.shape_len() - self.upcasted.clone() as usize)){
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
            iproduct!(self.sts[i].shape()[self.shape_len() - self.upcasted.clone() as usize..].into_iter().step_by(self.sts[i].shape()[self.shape_len() - self.upcasted.clone() as usize..].len() - 1).map(|x| *x)).collect_vec()
        }else{
            vec![]
        }
    }

    fn float4_axis(&self, i:usize) -> Vec<isize>{
        self.sts[i].unit_stride_axes().iter().filter_map(|x|{
            if x >= &(self.shape_len()-self.upcasted.clone() as usize) && self.sts[i].shape()[x]%4 == 0{
                Some(x - self.shape_len() - self.upcasted.clone())
            }else{
                None
            }
        }).collect_vec()
    }

    fn upcasted_axis(&self, i: usize) -> Vec<(BTypes, Option<BTypes>, bool)> {
        let shape_len = self.shape_len();
        let sts_i_shape = self.sts[i].shape();
        let sts_i_real_strides = self.sts[i].real_strides(false);
        let full_shape = &self.full_shape();
        let upcasted = self.upcasted.clone();
    
        (0..upcasted).map(|idx| {
            let shape_idx = shape_len - upcasted as usize + idx as usize;
            let full_shape_idx = shape_len - upcasted as usize + idx as usize;
            (
                sts_i_shape[shape_idx],
                sts_i_real_strides[shape_idx],
                sts_i_shape[shape_idx] != full_shape[full_shape_idx]
            )
        }).collect()
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

    fn acc_offsets(&self, i: usize) -> Vec<isize> {
        if self.upcasted == 0 {
            return vec![0];
        }
        
        let upcasted_i = self.upcasted_axis(i);
        let acc_strides: Vec<i32> = upcasted_i.iter()
            .enumerate()
            .map(|(i, x)| x * (1 - upcasted_i.iter().rev().map(|x| x.2).nth(i).unwrap_or(0)))
            .collect();
        
        let shape = upcasted_i.iter().rev().map(|x| if x.2 { 1 } else { x.0 }).collect::<Vec<_>>();
        let strides_for_shape: Vec<BTypes> = strides_for_shape(shape);
        
        let product_iter = iproduct!(upcasted_i.iter().rev(), 0..);
        let offsets: Vec<isize> = product_iter
            .map(|(x, _)| {
                let inner_offsets: Vec<Vec<isize>> = upcasted_i.iter().enumerate()
                    .map(|(i, x)| (0..{
                        match &x.0{
                            BTypes::Int(i) => *i,
                            BTypes::Node(_) => panic!()
                        }
                    }).map(|y| y * acc_strides[i]).collect::<Vec<isize>>())

                    .collect_vec();
                inner_offsets
            })
            .flatten()
            .map(|x| x.iter().sum())
            .collect();
        
        offsets
    }
    fn get_float4_upcast_dim(&self, i: usize) -> Vec<usize>{
        let should_upcast = self.opts.supports_float4() && (match &self.bufs[i]{
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
            return self.sts[i].unit_stride_axes(false).iter().filter_map(|x|{
                if x>= self.shape_len() - self.upcasted && self.sts[i].shape()[x] > BTypes::Int(1){
                    Some(x.clone())
                } else{
                    None
                }
            })
        }
    }
    fn first_reduce(&self) -> usize {
        let mut index = 0;
        for (x, y) in self.sts[0].shape.iter().zip(self.full_shape.iter()) {
            if x != y {
                return index;
            }
            index += 1;
        }
        index
    }

    fn output_shape(&self) -> Vec<BTypes>{
        return self.sts[0].shape()
    }

    fn full_shape(&self) -> Vec<BTypes>{
        return self.sts[self.full_buf_index.clone()].shape()
    }

    fn full_unupcasted(&self) -> Vec<BTypes>{
        self.full_shape()[..&self.shape_len()-&self.upcasted].collect()
    }

    fn shape_len(&self) -> usize{
        self.sts[0].shape().len()
    }

    fn upcast_in_mid_reduce_axes(&self) -> Vec<isize>{
        (self.first_reduce()..self.first_reduce() as isize + self.group_for_reduces).filter(|j|{
            if &self.full_shape()[j] == &self.sts[0].shape()[j]{
                j
            }else{
                None
            }
        })
    }

    fn global_dims(&self) -> isize{
        return self.first_reduce() as isize -self.local_dims.clone()
    }

    fn colors(&self) -> Vec<String> {
        // first non local non reduce dims are global (blue)
        let mut colors = {vec!["blue".to_string(); self.global_dims];
        if !self.dont_use_locals {
            vec!["blue".to_string(); self.global_dims]
        }else{
            vec!["BLUE".to_string(); self.global_dims()]
        }};
        // after global are local_dims; warp ones used in tensor cores must be closest to first_reduce (cyan)
        for _ in 0..self.local_dims {
            colors.push("cyan".to_string());
        }
        // between first_reduce and first_reduce + group_for_reduces, they are either upcast mid reduce (white), or late upcasted (green)
        for i in self.first_reduce..self.first_reduce + self.group_for_reduces {
            if self.upcast_in_mid_reduce_axes().contains(&i) {
                colors.push("white".to_string());
            } else {
                colors.push("green".to_string());
            }
        }
        // between first_reduce + group_for_reduces and upcasted, they are reduce (red)
        let reduce_count = (self.shape_len - self.upcasted) - (self.first_reduce + self.group_for_reduces);
        for _ in 0..reduce_count {
            colors.push("red".to_string());
        }
        // upcasted dimensions are reduce (magenta) or normal (yellow)
        for i in (self.shape_len - self.upcasted)..self.shape_len {
            if self.full_shape[i] != self.sts[0].shape[i] {
                colors.push("magenta".to_string());
            } else {
                colors.push("yellow".to_string());
            }
        }
        assert_eq!(colors.len(), self.shape_len, "colors size mismatch");
        colors
    }
    
    fn colored_shape(&self, pad: Option<isize>, dense: bool) -> String{
        //pad: None, dense: False
        let mut ret = {
            self.full_shape().into_iter().map(|s|{
                if !dense{
                    match &s{
                        BTypes::Int(_) => format!("{:4d}", s),
                        BTypes::Node(_) => format!("{:?}", s)
                    }
                }else{
                    format!("{:?}", s)
                }
            }).zip(self.colors().iter()).map(|(s, color)|{
                colored(s.as_str(), Some(&s.as_str()), false)
            }).join("_")
        };

        if let Some(p) = pad{
            ret = ret + " " +  p as usize - ansilen(ret.as_str())
        }
        ret
    }


    // ********** base sims ****************

    fn upcast(&mut self){
        assert!(self.full_shape()[self.full_shape().len()-1] != BTypes::Int(1));
        self.upcasted += 1
    }

    fn shift_to(self, axis: isize, amount: isize, top: bool, inset_before: Option<usize>){
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
        self.reshape_and_permute(Some(|x|{
            let mut result = vec![];
            result.extend_from_slice(&x[0..axis]);
            if x[axis] > 1{
                let amt = BTypes::Int(amount);
                if top{
                    result.push(amt);
                    result.push(x[axis].floordiv(amt));
                }else{
                    result.push(x[axis].floordiv(amt));
                    result.push(amt)
                }
            }else{
                result.push(BTypes::Int(1));
                result.push(BTypes::Int(1));
            }
        }), {
            let mut x = (0..ins_bf).filter(|i|{
                if i != move_axis{
                    i
                }else{
                    None
                }
            }).collect_vec();

            x.extend_from_slice(&move_axis);
            x.extend({
                (ins_bf..self.shape_len()+1).filter(|i|{
                    if i != &(move_axis as usize){
                        i
                    }else{
                        None
                    }
                })
            });
            Some(x)
        })
    }

    // ****************** comp sims ****************

    fn _limit_size<T>(&self, x: Vec<isize>, max_size: Vec<T>) -> Vec<isize>{
        let mut new_shape = x;
        for i in (0.. new_shape.len()){
            let mut next_idx = (i + 1) % new_shape.len();
            while new_shape[i] > max_size[i]{
                new_shape[i] = new_shape[i].div_floor(2);
                next_idx = {
                    if new_shape[next_idx] <= max_size[next_idx]{
                        next_idx
                    }else{
                        (next_idx + 1) % new_shape.len()
                    }
                };
                new_shape[next_idx] = new_shape[next_idx] * 2;
            }
        }
        return new_shape
    }

    fn limit_dims_to_max(&mut self, global_max: Vec<isize>, local_max: Vec<isize>){
        if self.global_dims() > 0{
            if !global_max.is_empty(){
                 let mut tmp: Vec<isize> = global_max[..self.global_dims()];

                if !local_max.is_empty(){
                    tmp.extend_from_slice(&local_max[..self.local_dims]);
                }

                if global_max.iter().max() < self.full_shape()[..self.global_dims()].iter().max(){
                    self.reshape_and_permute(Some(|x|{
                        self._limit_size(x, {
                            tmp.push(vec![MAX; self.full_shape().len() - tmp.len()]);
                            tmp
                        })
                    }), None)
                }
                assert!(global_max.iter().max() < self.full_shape()[..self.global_dims()].max());
                for i in (0..self.global_dims()-1){
                    if i < global_max.len() && self.full_shape()[i] > global_max[i]{
                        
                    }
                }
            }
        }
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
                st = st.permute(&x.into_iter().map(|y| y as isize).collect_vec()).borrow_mut()
            }
            new_sts.push(st.clone())
        });
        *sts = new_sts;
    }

    fn shape_len(sts: &Vec<Rc<ShapeTracker>>) -> usize{
        sts[0].shape().len()
    }