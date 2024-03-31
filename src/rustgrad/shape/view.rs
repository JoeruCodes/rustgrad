use core::panic;
use std::borrow::{Borrow, BorrowMut};
use std::clone;
use std::cmp::max;
use std::collections::HashSet;
use std::iter::Skip;
use std::process::Output;
use std::{collections::HashMap, ops::Deref};
use std::rc::Rc;
use cached::proc_macro::cached;
use itertools::{any, Itertools};
use num::{abs, Integer, Num, ToPrimitive};

use crate::rustgrad::helpers::{all_int, argsort};

use super::sym::{BTypes, NodeMethods, NodeTypes, NumNode, Variable};

fn canonicalize_strides(shape: &Vec<BTypes>, strides: &Vec<BTypes>) -> Vec<BTypes> {
    shape
        .iter()
        .zip(strides)
        .map(|(s, st)| if *s == BTypes::Int(1) { BTypes::Int(0) } else { st.clone() })
        .collect()
}


pub fn strides_for_shape(shape: &Vec<BTypes>) -> Vec<BTypes> {
    if shape.is_empty() {
        return Vec::new();
    }

    let strides: Vec<BTypes> = shape[1..]
        .iter()
        .rev()
        .scan(1, |state, val| {
            let s = &BTypes::Int(state.clone().to_isize().unwrap()) * val;
            Some(s)
        })
        .collect();
    
    canonicalize_strides(shape, &strides.iter().rev().cloned().collect::<Vec<BTypes>>())
}

// fn merge_dims(shape: &[isize], strides: &[isize], mask: Option<&[(isize, isize)]>) -> Vec<(isize, isize, isize)> {
//     // Merge contiguous subparts or zero strided dims.
//     // ret = Vec[(merged_dims, stride, merged dims w/o zero stride), ...]
//     if shape.is_empty() {
//         return Vec::new();
//     }
//     assert_eq!(shape.len(), strides.len());
//     let mut ret = vec![(shape[0], strides[0], if strides[0] == 0 { shape[0] } else { 0 })];
//     // wrt merging zero strided dimensions
//     let mut merging = strides[0] == 0 && (mask.map_or(false, |m| m[0].1 - m[0].0 == 1) || shape[0] == 1);
//     for (i, (&sh, &st)) in shape[1..].iter().zip(strides[1..].iter()).enumerate() {
//         if sh == 1 {
//             continue;
//         }
//         if merging || ret.last().unwrap().1 == sh * st { // mergeable
//             let (merged_dims, _, _) = ret.last_mut().unwrap();
//             *merged_dims *= sh;
//             let (_, _, merged_dims_without_zero_stride) = ret.last_mut().unwrap();
//             *merged_dims_without_zero_stride = if merging { sh } else { *merged_dims_without_zero_stride * sh };
//         } else {
//             ret.push((sh, st, if st == 0 { sh } else { 0 })); // begin new
//         }
//         // merging ends with either non-zero strided dim or zero strided dim with mask range > 1
//         merging = st == 0 && (mask.map_or(false, |m| m[i + 1].1 - m[i + 1].0 == 1) || sh == 1);
//     }
//     ret
// }

// fn reshape_mask(view: &View, new_shape: &[BTypes]) -> (Option<Vec<(BTypes, BTypes)>>, bool) {
//     if view.mask.is_none() {
//         return (view.mask.clone(), false);
//     }
//     if view.mask.clone().unwrap().into_iter().any(|m|{
//         match &m{
//             (BTypes::Int(_), BTypes::Int(_)) => true,
//             (_, _ )=> false
//         }
//     } ) {
//         return (view.mask.clone(), true);
//     }
//     let mut new_mask: Vec<(BTypes, BTypes)> = Vec::new();
//     let m: Vec<(BTypes, BTypes)> = view.mask.clone().unwrap();
//     let mut r_masks: std::iter::Rev<std::slice::Iter<'_, (BTypes, BTypes)>> = m.iter().rev();
//     let mut r_shape: std::iter::Rev<std::slice::Iter<'_, BTypes>> = view.shape.iter().rev();
//     let mut r_new_shape: std::iter::Rev<std::slice::Iter<'_, BTypes>> = new_shape.iter().rev();
//     let mut curr_stride: BTypes = BTypes::Int(1);
//     let mut old_dim: BTypes = r_shape.next().unwrap_or(&BTypes::Int(1)).clone();
//     let mut new_dim: BTypes = r_new_shape.next().unwrap_or(&BTypes::Int(1)).clone();
//     let mut mask: (BTypes, BTypes) = r_masks.next().unwrap_or(&(BTypes::Int(0), BTypes::Int(1))).clone();
//     if &(&mask.1 - &mask.0) < &BTypes::Int(1) {
//         return (Some(vec![(BTypes::Int(0), BTypes::Int(0)); new_shape.len()]), false);
//     }

//     while new_mask.len() < new_shape.len() {
//         let (ref l, ref r) = mask;
//         let next_stride: BTypes = &new_dim * &curr_stride;

//         if &old_dim >= &next_stride {
//             if &old_dim == &next_stride {
//                 new_mask.push((l.floordiv(&curr_stride, true), &(r - &BTypes::Int(1)).floordiv(&curr_stride, true) + &BTypes::Int(1)));
//                 curr_stride = BTypes::Int(1);
//                 old_dim = r_shape.next().unwrap_or(&BTypes::Int(1)).clone();
//                 new_dim = r_new_shape.next().unwrap_or(&BTypes::Int(1)).clone();
//                 mask = r_masks.next().unwrap_or(&(BTypes::Int(0), BTypes::Int(1))).clone();
//                 if &mask.1 - &mask.0 < BTypes::Int(1) {
//                     return (Some(vec![(BTypes::Int(0), BTypes::Int(0)); new_shape.len()]), false);
//                 }
//             } else {
//                 if (l % &next_stride != BTypes::Int(0) || r % &next_stride != BTypes::Int(0)) && l.floordiv(&next_stride, true) != (r - &BTypes::Int(1)).floordiv(&next_stride, true) {
//                     return (view.mask.clone(), true);
//                 }
//                 new_mask.push(((l % &next_stride).floordiv(&curr_stride, true), &(&(r - &BTypes::Int(1)) % &next_stride).floordiv(&curr_stride, true)+ &BTypes::Int(1)));
//                 curr_stride = next_stride;
//                 new_dim = r_new_shape.next().unwrap_or(&BTypes::Int(1)).clone();
//             }
//         } else {
//             let next_mask: &(BTypes, BTypes) = r_masks.next().unwrap_or(&(BTypes::Int(0), BTypes::Int(1)));
//             if mask != (BTypes::Int(0), old_dim.clone()) && &next_mask.1 - &next_mask.0 != BTypes::Int(1) {
//                 return (view.mask.clone(), true);
//             }
//             mask = (&(&next_mask.0 * &old_dim) + &l, &(&(&next_mask.1 - &BTypes::Int(1)) * &old_dim) + &r);
//             old_dim = &old_dim * r_shape.next().unwrap_or(&BTypes::Int(1));
//         }
//     }

//     for mask in r_masks {
//         if mask != &(BTypes::Int(0), BTypes::Int(1)) {
//             return (Some(vec![(BTypes::Int(0), BTypes::Int(0)); new_shape.len()]), false);
//         }
//     }

//     (Some(new_mask.into_iter().rev().collect()), false)
// }

fn merge_dims(shape: &[isize], strides: &[isize], mask: &Option<&[(isize, isize)]>) -> Vec<(isize, isize, isize)> {
    if shape.is_empty() {
        return Vec::new();
    }
    
    let mut ret = Vec::with_capacity(shape.len());
    ret.push((shape[0], strides[0], if strides[0] == 0 { shape[0] } else { 0 }));

    let mut merging = strides[0] == 0 && (mask.map_or(false, |m| m[0].1 - m[0].0 == 1) || shape[0] == 1);

    for (i, (&sh, &st)) in shape[1..].iter().zip(strides[1..].iter()).enumerate() {
        if sh == 1 {
            continue;
        }
        if merging || ret.last().unwrap().1 == sh * st {
            let (merged_dims, _, merged_dims_without_zero_stride) = ret.last_mut().unwrap();
            *merged_dims *= sh;
            *merged_dims_without_zero_stride = if merging { sh } else { *merged_dims_without_zero_stride * sh };
        } else {
            ret.push((sh, st, if st == 0 { sh } else { 0 }));
        }
        merging = st == 0 && (mask.map_or(false, |m| m[i + 1].1 - m[i + 1].0 == 1) || sh == 1);
    }
    
    ret
}

fn reshape_mask(view: &View, new_shape: &Vec<BTypes>) -> (Option<Vec<(BTypes, BTypes)>>, bool) {
    if let Some(mask) = &view.mask {
        if mask.iter().any(|(l, r)| match (l, r) {
            (BTypes::Int(_), BTypes::Int(_)) => true,
            _ => false,
        }) {
            return (view.mask.clone(), true);
        }

        let mut new_mask: Vec<(BTypes, BTypes)> = Vec::with_capacity(new_shape.len());
        let mut r_masks: std::iter::Rev<std::slice::Iter<'_, (BTypes, BTypes)>> = mask.iter().rev();
        let mut r_shape: std::iter::Rev<std::slice::Iter<'_, BTypes>> = view.shape.iter().rev();
        let mut r_new_shape: std::iter::Rev<std::slice::Iter<'_, BTypes>> = new_shape.iter().rev();
        let mut curr_stride: BTypes = BTypes::Int(1);
        let mut old_dim: BTypes = r_shape.next().cloned().unwrap_or(BTypes::Int(1));
        let mut new_dim: BTypes = r_new_shape.next().cloned().unwrap_or(BTypes::Int(1));

        if &(&mask.last().unwrap().1 - &mask.last().unwrap().0) < &BTypes::Int(1) {
            return (Some(vec![(BTypes::Int(0), BTypes::Int(0)); new_shape.len()]), false);
        }

        while let Some((l, r)) = r_masks.next() {
            let next_stride: BTypes = &new_dim * &curr_stride;

            if &old_dim >= &next_stride {
                if &old_dim == &next_stride {
                    new_mask.push((l.floordiv(&curr_stride, true), &(r - &BTypes::Int(1)).floordiv(&curr_stride, true) + &BTypes::Int(1)));
                    curr_stride = BTypes::Int(1);
                    old_dim = r_shape.next().cloned().unwrap_or(BTypes::Int(1));
                    new_dim = r_new_shape.next().cloned().unwrap_or(BTypes::Int(1));
                    if &(&mask.last().unwrap().1 - &mask.last().unwrap().0) < &BTypes::Int(1) {
                        return (Some(vec![(BTypes::Int(0), BTypes::Int(0)); new_shape.len()]), false);
                    }
                } else {
                    if (l % &next_stride != BTypes::Int(0) || r % &next_stride != BTypes::Int(0)) && l.floordiv(&next_stride, true) != (r - &BTypes::Int(1)).floordiv(&next_stride, true) {
                        return (view.mask.clone(), true);
                    }
                    new_mask.push(((l % &next_stride).floordiv(&curr_stride, true), &(&(r - &BTypes::Int(1)) % &next_stride).floordiv(&curr_stride, true) + &BTypes::Int(1)));
                    curr_stride = next_stride;
                    new_dim = r_new_shape.next().cloned().unwrap_or(BTypes::Int(1));
                }
            } else {
                let next_mask = r_masks.next().unwrap_or(&(BTypes::Int(0), BTypes::Int(1)));
                if l != &BTypes::Int(0) && &next_mask.1 - &next_mask.0 != BTypes::Int(1) {
                    return (view.mask.clone(), true);
                }
                let l_term = &(&next_mask.0 * &old_dim);
                let r_term = &(&(&next_mask.1 - &BTypes::Int(1)) * &old_dim) + r;
                new_mask.push((l_term + l, r_term));
                old_dim = &old_dim * r_shape.next().unwrap_or(&BTypes::Int(1));
            }
        }

        return (Some(new_mask.into_iter().rev().collect()), false);
    }

    (view.mask.clone(), false)
}

fn un1d(shape: &Vec<BTypes>, offs: &BTypes) -> Vec<BTypes> {
    let strides: Vec<BTypes> = strides_for_shape(shape);
    let mut result: Vec<BTypes> = Vec::new();
    let mut offs_n = offs.clone();
    if strides.is_empty() {
        result.push(BTypes::Int(0));
    } else {
        for stride in &strides {
            let here = offs_n.floordiv(stride, true);
            result.push(here.clone());
            offs_n = offs - &(&here * stride);
        }
    }

    result
}
#[derive(Clone, Debug, PartialEq, Hash)]
pub struct View{
    pub shape: Vec<BTypes>,
    pub strides: Vec<BTypes>,
    pub offset: BTypes,
    pub mask: Option<Vec<(BTypes, BTypes)>>,
    pub contiguous: bool
}

impl View{
    pub fn size(&self) -> isize{
        let ret:BTypes = self.shape.iter().map(|x|{
            match x{
                BTypes::Node(n) => NodeTypes::max(n.clone().deref()).unwrap(),
                BTypes::Int(_) => x.clone()
            }
        }).fold(BTypes::Int(1), |acc, x| &acc * &x);

        match ret{
            BTypes::Int(i) => i,
            BTypes::Node(_) => panic!("{}", format!("{:?} is not int", ret))
        }
    }

    // fn create(shape: &[BTypes], strides:Option<&[BTypes]>, offset: BTypes, mask: Option<Vec<(BTypes, BTypes)>>) -> View{
    //     let mut strides_n: Vec<BTypes>;
    //     let mut offset_n = offset;
    //     let mut mask_n = mask;
    //     if let None = strides{
    //         strides_n = strides_for_shape(shape);
    //     }else{
    //         strides_n = canonicalize_strides(shape, strides.unwrap());
    //     }
    //     if let Some(m) = &mask_n{
    //         if m.iter().zip(shape.iter()).all(|(x, y)|{
    //             *x == (BTypes::Int(0), y.clone())
    //         }){
    //             mask_n = None;
    //         }
    //     }
    //     let contiguous = offset_n == BTypes::Int(0) && mask_n.is_none() && strides == Some(&strides_for_shape(shape));
    //     let m = mask_n.clone().unwrap();
    //     let elim =  m.iter().map(|(b,e)|{
    //         !(&(b + &BTypes::Int(1)) < e)
    //     });
    //     if mask_n.is_some() && elim.clone().any(|x|x){
    //         if m.iter().any(|(b, e)|{
    //             !(b < e)
    //         }){
    //             strides_n = vec![BTypes::Int(0); shape.len()];
    //             offset_n = BTypes::Int(0);
    //             mask_n = Some(vec![(BTypes::Int(0), BTypes::Int(0)); shape.len()]);
    //         }

    //         offset_n = &offset_n + &elim.clone().enumerate().map(|(i, e)|{
    //             if e{
    //                 &strides_n[i] * &(mask_n.clone().unwrap()[i].0)
    //             } else {
    //                 BTypes::Int(0)
    //             }
    //         }).sum();

    //         strides_n = strides_n.iter().zip(elim).map(|(st, e)|{
    //             if e{
    //                 BTypes::Int(0)
    //             } else{
    //                 st.clone()
    //             }
    //         }).collect();
    //     }
    //     View{shape: shape.to_vec(), strides: strides_n, offset: offset_n, mask: mask_n, contiguous: contiguous}
    // }
    pub fn create(shape: &Vec<BTypes>, strides: Option<&Vec<BTypes>>, offset: BTypes, mask: Option<Vec<(BTypes, BTypes)>>) -> View {
        let mut strides_n: Vec<BTypes> = strides.unwrap_or(&strides_for_shape(shape)).to_vec();
    
        let mask_n = mask.map(|m| {
            if m.iter().cloned().zip(shape.iter()).all(|(x, y)| x == (BTypes::Int(0), y.clone())) {
                None
            } else {
                Some(m)
            }
        }).flatten();
    
        let contiguous = offset == BTypes::Int(0) && mask_n.is_none() && strides == Some(&strides_for_shape(shape));
    
        if let Some(mask_ref) = &mask_n {
            if mask_ref.iter().any(|(b, e)| !(b < e)) {
                strides_n = vec![BTypes::Int(0); shape.len()];
                return View {
                    shape: shape.to_vec(),
                    strides: strides_n,
                    offset: BTypes::Int(0),
                    mask: Some(vec![(BTypes::Int(0), BTypes::Int(0)); shape.len()]),
                    contiguous: true,
                };
            }
    
            let elim = mask_ref.iter().map(|(b, e)| !(&(b + &BTypes::Int(1)) < e));
    
            let offset_n = elim.clone().enumerate().fold(offset, |acc, (i, e)| {
                if e {
                    &acc + &(&strides_n[i] * &mask_ref[i].0)
                } else {
                    acc
                }
            });
    
            strides_n = strides_n.iter().zip(elim).map(|(st, e)| {
                if e {
                    BTypes::Int(0)
                } else {
                    st.clone()
                }
            }).collect();
    
            return View {
                shape: shape.to_vec(),
                strides: strides_n.to_vec(),
                offset: offset_n,
                mask: mask_n,
                contiguous: contiguous,
            };
        }
    
        View {
            shape: shape.to_vec(),
            strides: strides_n.to_vec(),
            offset: offset,
            mask: mask_n,
            contiguous: contiguous,
        }
    }

    pub fn vars(&self) -> HashSet<Rc<NodeTypes>> {
        let mut flatten_mask = Vec::new();
        if let Some(v) = &self.mask {
            flatten_mask.extend(v.iter().flat_map(|(x, y)| vec![x.clone(), y.clone()]));
        }
    
        let mut vec = Vec::new();
        vec.extend_from_slice(&self.shape);
        vec.extend_from_slice(&self.strides);
        vec.push(self.offset.clone());
        vec.extend_from_slice(&flatten_mask);
    
        let f = if let BTypes::Node(n) = vec[0].clone(){
            Some(vec![n])
        }else{
            None
        };
        vec.iter().flat_map(|x| {
            if let BTypes::Node(n) = x {
                n.vars()
            } else {
                HashSet::new()
            }
        }).collect()
    }

    // fn unbind(&self) -> (View, HashMap<Rc<NodeTypes>, Option<isize>>){
    //     let var_unboundvar_val: Vec<(Rc<NodeTypes>, (Rc<NodeTypes>, Option<isize>))> = self.vars().iter().flat_map(|v| {
    //         if let NodeTypes::Variable(vv) = v.clone().deref() {
    //             if let Some(vvv) = &vv.val {
    //                 // Return a value here
    //                 return Some((v.clone(), v.clone().unbind()));
    //             }
    //         }
    //         None
    //     }).collect::<Vec<_>>();

    //     let mut unbound_vars: HashMap<Rc<NodeTypes>, Rc<NodeTypes>> = HashMap::new();
        
    //     var_unboundvar_val.iter().for_each(|(v, (uv, _))|{
    //         unbound_vars.insert(v.clone(), uv.clone());
    //     });

    //     let new_shape: Vec<BTypes> = self.shape.iter().map(|s|{
    //         match s{
    //             BTypes::Int(i) => s.clone(),
    //             BTypes::Node(n) => BTypes::Node(n.clone().substitute(&unbound_vars))
    //         }
    //     }).collect_vec();


    //     let new_strides: Vec<BTypes> = self.strides.iter().map(|s|{
    //         match s{
    //             BTypes::Int(i) => s.clone(),
    //             BTypes::Node(n) => BTypes::Node(n.clone().substitute(&unbound_vars))
    //         }
    //     }).collect_vec();

    //     let new_offset = match &self.offset{
    //         BTypes::Int(_) => self.offset.clone(),
    //         BTypes::Node(n) => BTypes::Node(n.clone().substitute(&unbound_vars)) 
    //     };
    //     let new_mask: Option<Vec<(BTypes, BTypes)>> = match &self.mask{
    //         Some(m) => {
    //             Some(m.iter().map(|(a, b)|{
    //                 match (a,b){
    //                     (BTypes::Int(i), BTypes::Int(ii)) => (a.clone(), b.clone()),
    //                     (BTypes::Int(i), BTypes::Node(n)) => (a.clone(), BTypes::Node(n.clone().substitute(&unbound_vars))),
    //                     (BTypes::Node(n), BTypes::Int(_)) => (BTypes::Node(n.clone().substitute(&unbound_vars)), b.clone()),
    //                     (BTypes::Node(n), BTypes::Node(nn)) => (BTypes::Node(n.clone().substitute(&unbound_vars)), BTypes::Node(nn.clone().substitute(&unbound_vars)))
    //                 }
    //             }).collect::<Vec<_>>())
    //         }
    //         None => None
    //     };
    //     let mut hm: HashMap<Rc<NodeTypes>, Option<isize>> = HashMap::new();
    //     var_unboundvar_val.into_iter().for_each(|x: (Rc<NodeTypes>, (Rc<NodeTypes>, Option<isize>))| {hm.insert(x.1, x.1.1);});
    //     (View::create(&new_shape, Some(&new_strides), new_offset, new_mask), hm)
    // }
    pub fn unbind(&self) -> (View, HashMap<Rc<NodeTypes>, Option<isize>>) {
        let var_unboundvar_val: Vec<(Rc<NodeTypes>, (Rc<NodeTypes>, Option<isize>))> = self.vars()
            .iter()
            .filter_map(|v| {
                if let NodeTypes::Variable(vv) = v.clone().deref() {
                    if let Some(_) = &vv.val {
                        return Some((v.clone(), v.unbind()));
                    }
                }
                None
            })
            .collect();
    
        let unbound_vars: HashMap<Rc<NodeTypes>, Rc<NodeTypes>> = var_unboundvar_val
            .iter()
            .map(|(v, (uv, _))| (v.clone(), uv.clone()))
            .collect();
    
        let new_shape: Vec<BTypes> = self.shape
            .iter()
            .cloned()
            .map(|s| match s {
                BTypes::Int(_) => s,
                BTypes::Node(n) => BTypes::Node(n.substitute(&unbound_vars)),
            })
            .collect();
    
        let new_strides: Vec<BTypes> = self.strides
            .iter()
            .cloned()
            .map(|s| match s {
                BTypes::Int(_) => s,
                BTypes::Node(n) => BTypes::Node(n.substitute(&unbound_vars)),
            })
            .collect();
    
        let new_offset = match &self.offset {
            BTypes::Int(_) => self.offset.clone(),
            BTypes::Node(n) => BTypes::Node(n.clone().substitute(&unbound_vars)),
        };
    
        let new_mask: Option<Vec<(BTypes, BTypes)>> = self.mask.as_ref().map(|m| {
            m.iter().cloned().map(|(a, b)| match (&a, &b) {
                (BTypes::Int(_), BTypes::Int(_)) => (a, b),
                (BTypes::Int(_), BTypes::Node(n)) => (a, BTypes::Node(n.substitute(&unbound_vars))),
                (BTypes::Node(n), BTypes::Int(_)) => (BTypes::Node(n.substitute(&unbound_vars)), b),
                (BTypes::Node(n), BTypes::Node(nn)) => (BTypes::Node(n.substitute(&unbound_vars)), BTypes::Node(nn.substitute(&unbound_vars))),
            }).collect()
        });
    
        let hm: HashMap<Rc<NodeTypes>, Option<isize>> = var_unboundvar_val
            .into_iter()
            .map(|x| (x.1.0, x.1.1))
            .collect();
    
        (View::create(&new_shape, Some(&new_strides), new_offset, new_mask), hm)
    }

    pub fn reshape(&self, new_shape: &Vec<BTypes>) -> Option<View>{
        if &self.shape == new_shape{
            return Some(self.clone())
        }

        assert!(new_shape.iter().all(|s|{
            if s >= &BTypes::Int(0){
                true
            } else{
                false
            }
        }), "{}", format!("shape can't contain negative numbers {:?}", new_shape));

        if self.shape.iter().any(|x|{
            if x == &BTypes::Int(0){
                new_shape.iter().any(|y|{
                    if y == x{
                        panic!("cannot reshape 0 size to {:?}", new_shape)
                    }else{
                        true
                    }
                });
                true
            }else{
                false
            }
        }){
            return Some(View::create(&new_shape, None, BTypes::Int(0), None));
        }

        if all_int(&self.shape){
            assert!(new_shape.iter().all(|s|{
                match s{
                    BTypes::Int(_) => true,
                    BTypes::Node(n) => {
                        match n.clone().deref(){
                            NodeTypes::Variable(_) => true,
                            _ => false
                        }
                    }
                }
            }), "{}", format!("{:?} -> {:?} contains non (int, Variable) dim", self.shape, new_shape));

            if self.shape.iter().fold(BTypes::Int(1), |acc, x|{
                &acc * x
            }) != new_shape
            .iter()
            .map(|s| {
                match s {
                    BTypes::Int(_) => s.clone(),
                    BTypes::Node(n) => {
                        match n.clone().deref() {
                            NodeTypes::Variable(v) => {
                                BTypes::Int(v.val().unwrap())
                            }
                            _ => {
                                n.clone()
                                    .vars()
                                    .into_iter()
                                    .map(|x| {
                                        match x.clone().deref() {
                                            NodeTypes::Variable(ii) => BTypes::Int(ii.val().unwrap()),
                                            _ => unreachable!(),
                                        }
                                    }).fold(BTypes::Int(1), |acc, x| &acc * &x) // Collect the transformed values into a single vector
                            }
                        }
                    }
                }
            })
            .fold(BTypes::Int(1), |acc, x| &acc * &x){
                panic!("size mismatched, can't reshape {:?} -> {:?}", self.shape, new_shape)
            }
        }

        if new_shape.is_empty() && self.mask.clone().unwrap().iter().any(|(mx, my)|{
            if mx == my{
                true
            } else{
                false
            }
        }){
            return None
        }

        if self.contiguous{
            return Some(View::create(new_shape, None, BTypes::Int(0), None));
        }

        let mut strides = vec![];
        let r_new_shape = new_shape.iter().rev().cloned().collect_vec();
        // merge_dims(&self.shape.iter().map(|x|{
        //     if let BTypes::Int(i) = x{
        //         i
        //     }else{
        //         unreachable!()
        //     }
        // }).collect::<Vec<isize>>(), &self.strides.iter().map(|x|{
        //     if let BTypes::Int(i) = x{
        //         i
        //     }else{
        //         unreachable!()
        //     }
        // }).collect::<Vec<isize>>(), &Some(&self.mask.clone().unwrap().into_iter().map(|(x, y)|{
        //     if let BTypes::Int(i) = x{
        //         if let BTypes::Int(ii) = y{
        //             (i, ii)
        //         } else{
        //             unreachable!()
        //         }
        //     }else{
        //         unreachable!()
        //     }
        // }).collect::<Vec<(isize, isize)>>())).into_iter().rev().for_each(|(merged_dim, mut new_stride, real_dim)| {
        //     let mut acc = BTypes::Int(1);
        //     while &acc <= &BTypes::Int(merged_dim.clone() as isize) && &acc != &BTypes::Int(merged_dim.clone() as isize) {
        //         if let Some(new_dim) = r_new_shape.iter().next().map(|x|Some(x)).unwrap_or(None) {
        //             strides.push(new_stride);
        //             if new_dim != &BTypes::Int(1) {
        //                 acc = &acc * new_dim;
        //                 if &acc < &BTypes::Int(real_dim as isize) {
        //                     new_stride = new_stride * {match new_dim{
        //                         BTypes::Int(i) => *i as isize,
        //                         BTypes::Node(n) => unreachable!()
        //                     }};
        //                 } else {
        //                     acc = BTypes::Int(0);
        //                 }
        //             }
        //         } else {
        //             break; // If next() returns None, exit loop
        //         }
        //     }
        //     if acc != BTypes::Int(merged_dim as isize) {
        //         break;
        //     }
        // });

        for (merged_dim, mut new_stride, real_dim) in merge_dims(
            &self.shape.iter().cloned().map(|x| match x {
                BTypes::Int(i) => i,
                _ => unreachable!(),
            }).collect::<Vec<isize>>(),
            &self.strides.iter().cloned().map(|x| match x {
                BTypes::Int(i) => i,
                _ => unreachable!(),
            }).collect::<Vec<isize>>(),
            &Some(
                &self.mask.clone().unwrap().into_iter().map(|(x, y)| {
                    if let (BTypes::Int(i), BTypes::Int(ii)) = (x, y) {
                        (i, ii)
                    } else {
                        unreachable!()
                    }
                }).collect::<Vec<(isize, isize)>>()
            )
        ).into_iter().rev() {
            let mut acc = BTypes::Int(1);
            while &acc <= &BTypes::Int(merged_dim.clone() as isize) && &acc != &BTypes::Int(merged_dim.clone() as isize) {
                if let Some(new_dim) = r_new_shape.iter().next().map(|x| Some(x)).unwrap_or(None) {
                    strides.push(new_stride);
                    if new_dim != &BTypes::Int(1) {
                        acc = &acc * new_dim;
                        if &acc < &BTypes::Int(real_dim as isize) {
                            new_stride = new_stride * {
                                match new_dim {
                                    BTypes::Int(i) => *i as isize,
                                    BTypes::Node(_) => unreachable!()
                                }
                            };
                        } else {
                            acc = BTypes::Int(0);
                        }
                    }
                } else {
                    break; // If next() returns None, exit loop
                }
            }
            if acc != BTypes::Int(merged_dim as isize) {
                break;
            }
        }
        strides.extend(vec![0; new_shape.len() * strides.len()]);
        let (new_mask, extra) = reshape_mask(&self, &new_shape);
        if !extra{
            let c_s = {match &new_mask{
                Some(n_m) => n_m.iter().map(|(b, e)|{
                    e-b
                }).collect_vec(),
                None => new_shape.clone()
            }};
            let new_strides = canonicalize_strides(&c_s, &strides.into_iter().rev().map(|x| BTypes::Int(x as isize)).collect:: <Vec<BTypes>>());
            let extra_offset = {
                match &self.mask{
                    Some(m_p) => &m_p.iter().zip(self.strides.iter()).map(|(m, s)|{
                        &m.0 * s
                    }).fold(BTypes::Int(0), |acc, x| &acc + &x) - &new_mask.clone()?.iter().zip(new_strides.iter()).map(|(m, s)|{
                        &m.0 * s
                    }).fold(BTypes::Int(0), |acc, x| &acc + &x),

                    None => BTypes::Int(0)
                }
            };

            return Some(View::create(&new_shape, Some(&new_strides), &self.offset + &extra_offset, new_mask));
        }
        None
    }

    pub fn invert(&self, out_shape: &[BTypes]) -> Option<View>{
        let mut ret = View::create(&self.shape, None, BTypes::Int(0), None);
        if let Some(m) = &self.mask{
            ret = ret.shrink(m);
        }
        ret = ret.stride(&self.strides.iter().map(|x|{
            if x < &BTypes::Int(0){
                -1
            } else{
                1
            }
        }).collect::<Vec<_>>()).permute(&argsort(&self.strides.iter().map(|x|{
            if x > &BTypes::Int(0){
                -x
            }else{
                x.clone()
            }
        }).collect_vec()));

        if ret.shape.iter().fold(BTypes::Int(1), |acc, x| &acc * x) == out_shape.iter().fold(BTypes::Int(1), |acc, x| &acc * x){
            return Some(ret);
        }
        None
        
    }

    pub fn minify(&self) -> View{
        let min_shape = merge_dims(&self.shape.iter().map(|x|{
            match &x{
                BTypes::Int(i) => i.clone(),
                _ => panic!()
            }
        }).collect::<Vec<_>>(), &self.strides.iter().map(|x|{
            match &x{
                BTypes::Int(i) => i.clone(),
                _ => panic!()
            }
        }).collect::<Vec<_>>(), &Some(&self.mask.clone().unwrap().iter().map(|(x, y)|{
            match (x, y){
                (BTypes::Int(i), BTypes::Int(ii)) => (*i as isize, *ii as isize),
                (_, _) => panic!()
            }
        }).collect_vec())).into_iter().map(|x| x.0);

        let nv = self.reshape(&min_shape.into_iter().map(|x| BTypes::Int(x as isize)).collect_vec());
        if let Some(nv_p) = nv{
            nv_p
        }else{
            self.clone()
        }

    }

    fn _unsafe_resize(&self, arg: &Vec<(BTypes, BTypes)>, mask: Option<Vec<(BTypes, BTypes)>>) -> View{
        let offset: BTypes = self.strides.iter().zip(arg.iter()).map(|(s, x)| s * &x.0).sum();
        let mut mask = mask;
        if let Some(m_p) = &self.mask{
            let nmask = m_p.iter().zip(arg.iter()).map(|((mx, my), (ax, ay))|{
                (max(BTypes::Int(0), std::cmp::min(mx - ay, ay - ax)), max(BTypes::Int(0), std::cmp::min(my- ax, ay-ax)))
            });

            mask = {
                match &mask{
                    Some(m_a) => {
                        Some(nmask.zip(m_a.iter()).map(|((mx1, my1), (mx2, my2))|{
                            (max(mx1, mx2.clone()), max(my1, my2.clone()))
                        }).collect_vec())
                    }
                    None => Some(nmask.collect_vec())
                }
            };
        }
        let shape = arg.iter().map(|(x, y)|{
            y-x
        });
        if let Some(m_p) = &mask{
            if  m_p.iter().zip(shape.clone()).all(|(m, s)|{
                if m.0 == BTypes::Int(0) && m.1 == s{
                    true
                }else{
                    false
                }
            }){
                mask = None;
            }
        }

        return View::create(&shape.map(|s|{
            match &s{
                BTypes::Int(_) => s,
                BTypes::Node(n) =>{
                    if let NodeTypes::NumNode(nn) =n.deref(){
                        BTypes::Int(nn.b)
                    }else{
                        s
                    }
                }
            }
        }).collect_vec(), Some(&self.strides), &self.offset + &offset, mask)
    }

    pub fn pad(&self, arg: &Vec<(BTypes, BTypes)>) -> View{
        assert!(arg.iter().all(|(b, e)| b>= &BTypes::Int(0) && e>=&BTypes::Int(0)) && arg.len() == self.shape.len(), "{}", format!("{:?}, {:?}", self.shape, arg));
        if arg.iter().any(|(b ,e)|{
            b != &BTypes::Int(0) || e != &BTypes::Int(0)
        }){
            let zvarg = self.shape.iter().zip(arg.iter()).map(|(s, (b,e))|{
                (-b, s + e)
            }).collect_vec();
    
            let mask = self.shape.iter().zip(arg.iter()).map(|(s, (b,e))|{
                (b.clone(), s + b)
            }).collect_vec();
    
            self._unsafe_resize(&zvarg, Some(mask))
        }else{
            self.clone()
        }
    }

    pub fn shrink(&self, arg: &Vec<(BTypes, BTypes)>) ->  View{
        assert!(self.shape.iter().zip(arg.iter()).all(|(s, (b, e))|{
            &BTypes::Int(0) <= b && b<=e && e<=s
        }) && arg.len() == self.shape.len(), "{}", format!("invalid shrink {:?} for {:?}", arg, self.shape));

        return self._unsafe_resize(arg, None)
    }

    pub fn expand(&self, new_shape: &Vec<BTypes>) -> View{
        if new_shape.len() != self.shape.len(){
            panic!("expand arg {:?} must hasve same number of dims as shape {:?}", new_shape, self.shape)
        }

        if self.shape.contains(&BTypes::Int(0)){
            assert!(self.shape.iter().zip(new_shape.iter()).all(|(s, x)|{
                (s == x && x == &BTypes::Int(0)) || (s > &BTypes::Int(0) && &(x % s) == &BTypes::Int(0))
            }), "{}", format!("cant expand {:?} into {:?}", self.shape, new_shape));

            return View::create(&new_shape, None, BTypes::Int(0), None);
        }
        assert!(self.shape.iter().zip(new_shape.iter()).zip(self.strides.iter()).all(|((s, x) , st)|{
            (s == x || (s == &BTypes::Int(1) && st == &BTypes::Int(0)))
        }), "{}", format!("cant expand {:?} into {:?}", self.shape, new_shape));

        let mask: Option<Vec<(BTypes, BTypes)>> = {
            match &self.mask {
                Some(m_p) => {
                    Some(
                        m_p.iter()
                            .zip(self.shape.iter())
                            .zip(new_shape.iter())
                            .filter_map(|((m, s), ns)| {
                                if s != ns {
                                    if m != &(BTypes::Int(0), BTypes::Int(1)) {
                                        Some((BTypes::Int(0), BTypes::Int(0)))
                                    } else {
                                        Some((BTypes::Int(0), ns.clone()))
                                    }
                                } else {
                                    None // Skip this iteration
                                }
                            })
                            .collect_vec(),
                    )
                }
                None => None,
            }
        };
        return View::create(&new_shape, Some(&self.strides), self.offset.clone(), mask)

    }

    pub fn permute(&self, axis: &Vec<usize>) -> View{
        assert!(axis.iter().all(|x|{
            x >= &0 && x < &(self.shape.len())
        }), "invalid permute {:?} for {:?}", axis, self.shape);

        assert!({
            let mut x = HashSet::new();
            axis.iter().map(|a| x.insert(a));
            x.len()
        } == self.shape.len(), "{}", format!("cant permute {:?} with {:?}", self.shape, axis));

        return View::create(&axis.iter().map(|a|{
            self.shape[a.clone()].clone()
        }).collect::<Vec<BTypes>>(), Some(&axis.iter().map(|a|{
            self.strides[a.clone()].clone()
        }).collect::<Vec<BTypes>>()), self.offset.clone(), {
            match &self.mask{
                Some(m) => Some(axis.iter().map(|a|{
                    m[a.clone()].clone()
                }).collect::<Vec<(BTypes, BTypes)>>()),
                None =>  None
            }
        })
    }

    pub fn stride(&self, mul: &[isize]) -> View{
        assert!(mul.iter().all(|x|{
            x != &0
        }), "{}", format!("invalid stride {:?} for {:?}", mul, self.shape));

        let strides = self.strides.iter().zip(mul.iter()).map(|(z, m)| z*&BTypes::Int(m.clone())).collect_vec();
        let new_shape = self.shape.iter().zip(mul.iter()).map(|(s, m)| (s+ &BTypes::Int((abs(m.clone()) - 1))).floordiv(&BTypes::Int(abs(m.clone())), true)).collect_vec();
        let offset = self.shape.iter().zip(self.strides.iter()).zip(mul.iter()).filter_map(|((s, z), m)|{
            if m < &0{
                Some(&(s - &BTypes::Int(1)) * z)
            }else{
                None
            }
            
        }).fold(BTypes::Int(0), |acc, x| &acc + &x);
        let mask = {
            match &self.mask {
                Some(mask) => {
                    Some(mask.iter().zip(self.shape.iter()).zip(mul.iter()).map(|(((mx, my), s), m)| {
                        // Assign intermediate values to variables before using them
                        let x = if m > &0 {
                            mx.clone()
                        } else {
                            s - my
                        };
                        let y = if m > &0 {
                            my.clone()
                        } else {
                            s - mx
                        };
                        let abs_m = abs(m.clone());
        
                        (
                            &x + &BTypes::Int((abs_m - 1).div_floor(&abs_m) as isize),
                            &y + &BTypes::Int((abs_m - 1).div_floor(&abs_m) as isize),
                        )
                    }).collect_vec())
                }
                None => None
            }
        };

        return View::create(&new_shape, Some(&strides), &self.offset + &offset, mask)
    }
}

impl<'a> std::ops::Add<&'a View> for &'a View
{
    type Output = Option<View>;
    fn add(self, vm1: &'a View) -> Option<View> {
        let vm2 = self;
        if vm2.contiguous.clone(){
            return Some(vm1.clone());
        }
        if vm1.contiguous.clone() && &vm1.shape == &vm2.shape{
            return Some(vm2.clone())
        }
        let ret:Option<View> = vm2.reshape(&vm1.shape);
        if vm1.contiguous.clone() && vm1.size() == vm2.size() && ret.is_some(){
            return Some(ret.unwrap());
        }
        if vm1.mask.is_some(){
            if vm1.mask.clone().unwrap().into_iter().any(|(b, e)|{
                if !(b < e){
                    false;
                }
                true          
            }){
                return Some(View::create(&vm1.shape, Some(&vec![BTypes::Int(0); vm1.shape.len()]), BTypes::Int(0), Some(vec![(BTypes::Int(0),BTypes::Int(0)); vm1.shape.len()])))
            }
            let merged = vm2 + &vm1.shrink(&vm1.mask.clone()?);
            
            return Some(merged?.pad(&vm1.mask.clone()?.iter().zip(vm1.shape.iter()).filter_map(|((b,e), s)|{
                if let (BTypes::Int(i), BTypes::Int(ii), BTypes::Int(iii)) = (b, e, s){
                    return Some((b.clone(), (s-e)))
                }
                None
            }).collect()))
            // return (merged && merged.pad(vm1.mask.clone().unwrap().iter().zip(vm1.shape.iter()).map(|((b,e), s)|{
            //     (b, s-e)
            // }).collect()));
        }

        let origin = un1d(&vm2.shape, &vm1.offset);
        let mut terms: Vec<Vec<(usize, BTypes)>> = vec![vec![]; origin.len()];
        let mut strides: Vec<BTypes> = vec![BTypes::Int(0); vm1.shape.len()];

        let _ = vm1.strides.iter().enumerate().map(|(d1, st)|{
            if st != &BTypes::Int(0){
                let _ = origin.iter().zip(un1d(&vm2.shape, &(&vm1.offset + st)).into_iter()).enumerate().map(|(d2, (o, ref mut s1))|{
                    *s1 = &*s1 - o;
                    terms[d2].push((d1, s1.clone()));
                    strides[d1] = &strides[d1] + &(&*s1 * &vm2.strides.clone()[d2])
                });
            }
        });

        let idxs: Vec<Rc<NodeTypes>> = vm1.shape.iter().enumerate().map(|(i, s)|{
            Variable::init(format!("idx{}", i), 0, &(s-&BTypes::Int(1)))
        }).collect_vec();

        let mut merged_size = BTypes::Int(1);
        let mut merged_term = NumNode::init(&0);
        let mut extends: Vec<(BTypes, Rc<NodeTypes>)> = vec![];

        terms.iter().rev().zip(vm2.shape.iter().rev().zip(origin.iter().rev())).for_each(|(term, (s, o))|{
            merged_term = ((merged_term.deref() + NodeTypes::sum(&term.iter().map(|(d1, s1)|{
                idxs[*d1].deref() * &(s1 * &merged_size)
            }).collect_vec()).deref()).deref() + o).deref() * &merged_size;

            merged_size = &merged_size * s ;

            if !(&merged_size <= merged_term.clone().deref()) && !(<&NodeTypes as Into<bool>>::into(merged_term.clone().deref().n2i_lt(&0).deref())){
                extends.push((merged_size.clone(), merged_term.clone()));
                merged_size = BTypes::Int(1);
                merged_term = NumNode::init(&0);
            }
        });

        if (merged_term.clone().deref() == NumNode::init(&0).deref()).into(){
            return None
        }
        let vm2_shape = extends.iter().rev().cloned().map(|(s,_)|{
            s
        }).collect_vec();
        if vm2_shape != vm2.shape{
            let reshaped_vm2: Option<View> = vm2.reshape(&vm2_shape);
            // return reshaped_vm2 && reshaped_vm2 + vm1;
            return reshaped_vm2.clone().and(&reshaped_vm2? + vm1)
        }

        if vm2.mask.is_some(){
            let mut newb = vec![BTypes::Int(0); vm1.shape.len()];
            let mut newe = vm1.shape.clone();
            let mut bad = false;

            // vm2.mask.clone().unwrap().iter().zip(origin.iter()).zip(extends.iter()).enumerate().filter_map(|(d2, (((b, e), o), (_, t)))|{
            //     if !(t.min().unwrap() < b || t.max().unwrap() >= e){
            //         continue
            //     } else if let BTypes::Node(_) = &o{
            //         if let BTypes::Node(_) = &b{
            //             if let BTypes::Node(_) = &e{
            //                 bad = true;
            //                 continue;
            //             }
            //             bad = true;
            //             continue;
            //         }
            //         bad =true;
            //         continue;
            //     } else{
            //         let term = terms[d2];
            //         if term.len() != 1{
            //             if term.is_empty() && !newe.is_empty(){
            //                 newe[0] = BTypes::Int(0);
            //             } else{
            //                 bad = true;
            //             }
            //             continue;
            //         }
            //         let d1, s1 = term[0].clone();
            //         if let  BTypes::Node(_) = &s1{
            //             if let BTypes::Node(_) = &newe[d1.clone()]{
            //                 bad = true;
            //                 continue;
            //             }
            //             bad = true;
            //             continue;
            //         }
            //         newb[d1] = max(newb[d1.clone()], if s1 > 0{
            //             if let BTypes::Int(i) = ((b-o)/s1){
            //                 BTypes::Int(i.ceil())
            //             }
            //         } else{
            //             if let BTypes::Int(i) = ((e - o - BTypes::Int(1))/s1){
            //                 BTypes::Int(i.ceil())
            //             }
            //         });
            //     }
            // });
            
            let _ = vm2.mask.clone().unwrap().iter().zip(origin.iter()).zip(extends.iter()).enumerate().filter_map(|(d2, (((b, e), o), (_, t)))| {
                if !(&BTypes::Int(t.min().unwrap()) < b || &t.max().unwrap() >= e) {
                    return None;
                } else if let BTypes::Node(_) = &o {
                    if let BTypes::Node(_) = &b {
                        if let BTypes::Node(_) = &e {
                            bad = true;
                            return None;
                        }
                        bad = true;
                        return None;
                    }
                    bad = true;
                    return None;
                } else {
                    let term = &terms[d2];
                    if term.len() != 1 {
                        if term.is_empty() && !newe.is_empty() {
                            newe[0] = BTypes::Int(0);
                        } else {
                            bad = true;
                        }
                        return None;
                    }
                    let (d1, s1) = term[0].clone();
                    if let BTypes::Node(_) = &s1 {
                        if let BTypes::Node(_) = &newe[d1.clone()] {
                            bad = true;
                            return None;
                        }
                        bad = true;
                        return None;
                    }
                    newb[d1] = max(
                        newb[d1.clone()].clone(),
                        if s1 > BTypes::Int(0) {
                            if let BTypes::Int(i) = &(b - o) / &s1 {
                                BTypes::Int(i)
                            } else {
                                unreachable!(); 
                            }
                        } else {
                            if let BTypes::Int(i) = &(&(e - o) - &BTypes::Int(1)) / &s1 {
                                BTypes::Int(i)
                            } else {
                                unreachable!();
                            }
                        },
                    );
                }
                Some(())
            });
            if newb.iter().zip(newe.iter()).zip(vm1.shape.iter()).any(|((b, e), s)|{
                if b != &BTypes::Int(0) || e != s{
                    false
                } else{
                    true
                }
                
            }){
                return vm2 + &View::create(&vm1.shape, Some(&vm1.strides), vm1.offset.clone(), Some(newb.into_iter().zip(newe.into_iter()).map(|(x, y)| (x, y)).collect::<Vec<(BTypes, BTypes)>>()));
            }
            if bad{
                return None;
            }
        }
        let mut offsets_n = origin
        .into_iter()
        .zip(vm2.strides.iter())
        .map(|(origin_val, stride_val)| &origin_val * stride_val)
        .collect::<Vec<BTypes>>();
    
        offsets_n.push(vm2.offset.clone());

    
        return Some(View::create(
            &vm1.shape,
            Some(&strides),
            offsets_n
            .iter()
            .fold(BTypes::Int(0), |acc, val| &acc + val),
            None,
        ));

    }
}