use core::panic;
use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::ops::{Mul, Neg, Rem, Sub};
use std::rc::Weak;
use std::sync::{Arc, Mutex};
use std::{any::Any, collections::HashMap, fmt::Debug, ops::Deref, rc::Rc};

use itertools::Itertools;
use num::complex::ComplexFloat;
use num::integer::gcd;
use num::{Num, ToPrimitive};

use crate::rustgrad::helpers::{partition, ContextVar, DEBUG, REPR};
type N = Rc<NodeTypes>;
// pub enum Context {
//     DEBUG,
//     REPR,
// }
#[derive(Debug, Clone)]
pub struct Variable {
    pub expr: String,
    min: Option<isize>,
    max: Option<BTypes>,
    pub val: Option<isize>,
    ptr: RefCell<Option<Weak<NodeTypes>>>,
}
#[derive(Clone, Hash)]
pub enum BTypes {
    Node(N),
    Int(isize),
}
#[derive(Debug, Clone)]
pub struct OpNode {
    pub a: N,
    pub b: BTypes,
    min: Option<isize>,
    max: Option<BTypes>,
    ptr: RefCell<Option<Weak<NodeTypes>>>,
}
#[derive(Debug, Clone)]
pub struct NumNode {
    pub b: isize,
    min: Option<isize>,
    max: Option<BTypes>,
    ptr: RefCell<Option<Weak<NodeTypes>>>,
}
#[derive(Debug, Clone)]
pub struct RedNode {
    pub nodes: Vec<N>,
    min: Option<isize>,
    max: Option<BTypes>,
    ptr: RefCell<Option<Weak<NodeTypes>>>,
}
#[derive(Debug, Clone)]
pub enum NodeTypes {
    Variable(Variable),
    NumNode(NumNode),
    LtNode(OpNode),
    MulNode(OpNode),
    DivNode(OpNode),
    ModNode(OpNode),
    SumNode(RedNode),
    AndNode(RedNode),
}
pub trait NodeMethods {
    fn render(&self, ops: &Option<Box<dyn Any>>, ctx: &Option<Arc<Mutex<ContextVar>>>) -> String;

    fn vars(&self) -> HashSet<N>;

    fn substitute(&self, var_vals: &HashMap<N, N>) -> N;

    fn unbind(&self) -> (N, Option<isize>);

    //to be cached
    fn key(&self) -> String;

    //to be cached
    fn hash(&self) -> u64;

    fn sum(nodes: &Vec<N>) -> N;

    fn ands(nodes: &Vec<N>) -> N;

    fn min(&self) -> Option<isize>;
    fn max(&self) -> Option<BTypes>;

    fn flat_components(&self) -> Vec<N>;

    fn ptr(&self) -> N;

    fn get_bounds(&self) -> (Option<isize>, Option<BTypes>);
}

impl NodeMethods for NodeTypes {
    fn render(&self, ops: &Option<Box<dyn Any>>, ctx: &Option<Arc<Mutex<ContextVar>>>) -> String {
        match self {
            NodeTypes::Variable(v) => ops
                .as_ref()
                .and_then(|_| {
                    ctx.as_ref().map(|c| {
                        let c_lk = c.lock().unwrap();
                        if c_lk.borrow().deref() == DEBUG.clone().deref().lock().unwrap().deref() {
                            format!(
                                "{}[{:?}-{:?}{}]",
                                &v.expr,
                                self.min().unwrap(),
                                self.max().unwrap(),
                                v.val
                                    .as_ref()
                                    .map_or_else(|| String::new(), |val| format!("={}", val))
                            )
                        } else if REPR.clone().deref().lock().unwrap().deref() == c_lk.deref() {
                            format!(
                                "Variable('{}', {:?}, {:?}){}",
                                &v.expr,
                                self.min().unwrap(),
                                self.max().unwrap(),
                                v.val
                                    .as_ref()
                                    .map_or_else(|| String::new(), |val| format!(".bind({})", val))
                            )
                        } else {
                            v.expr.clone()
                        }
                    })
                })
                .unwrap_or_else(|| v.expr.clone()),
            NodeTypes::NumNode(n) => ops
                .as_ref()
                .and_then(|_| {
                    ctx.as_ref().map(|c| {
                        let c_lk = c.deref().lock().unwrap();
                        if REPR.clone().deref().lock().unwrap().deref() == c_lk.deref() {
                            format!("NumNode({})", &n.b)
                        } else {
                            format!("{}", &n.b)
                        }
                    })
                })
                .unwrap_or_else(|| format!("{}", n.b)),

            NodeTypes::LtNode(l) => {
                assert!(&self.min().unwrap() != &self.max().unwrap());
                format!("{} / {:?}", l.a.clone().render(ops, ctx), l.b.clone())
            }

            NodeTypes::MulNode(m) => {
                assert!(&self.min().unwrap() != &self.max().unwrap());
                render_mulnode(m, ops, ctx)
            }

            NodeTypes::ModNode(md) => {
                assert!(&self.min().unwrap() != &self.max().unwrap());
                format!("{} % {:?}", md.a.clone().render(ops, ctx), &md.b)
            }

            NodeTypes::SumNode(s) => {
                assert!(&self.min().unwrap() != &self.max().unwrap());
                let mut vec: Vec<String> =
                    s.nodes.iter().map(|x| x.clone().render(ops, ctx)).collect();
                vec.sort();
                format!("{}", vec.join("+"))
            }
            NodeTypes::AndNode(a) => {
                assert!(&self.min().unwrap() != &self.max().unwrap());
                let mut vec: Vec<String> =
                    a.nodes.iter().map(|x| x.clone().render(ops, ctx)).collect();
                vec.sort();
                format!("{}", vec.join(" and "))
            }
            NodeTypes::DivNode(d) => {
                assert!(self.min().unwrap() != self.max().unwrap());
                format!("{} // {:?}", d.a.render(ops, ctx), &d.b)
            }
        }
    }
    fn vars(&self) -> HashSet<N> {
        match self {
            NodeTypes::Variable(_) => {
                let mut x = HashSet::new();
                x.insert(self.ptr());
                x
            },
            NodeTypes::LtNode(n)
            | NodeTypes::DivNode(n)
            | NodeTypes::ModNode(n)
            | NodeTypes::MulNode(n) => {
                let mut result = n.a.clone().vars();
                if let BTypes::Node(b_n) = &n.b {
                    result.extend(b_n.clone().vars());
                }
                result
            }
            NodeTypes::SumNode(n) | NodeTypes::AndNode(n) => {
                n.nodes.iter().flat_map(|x| x.clone().vars()).collect()
            }
            _ => HashSet::new(),
        }
    }
    fn substitute(&self, var_vals: &HashMap<N, N>) -> N {
        match self {
            NodeTypes::Variable(_) => var_vals
                .get(&self.ptr())
                .cloned()
                .unwrap_or_else(|| self.ptr()),
            NodeTypes::NumNode(_) => self.ptr(),
            NodeTypes::LtNode(l) => match &l.b {
                BTypes::Int(i) => l.a.clone().substitute(var_vals).n2i_lt(i),
                BTypes::Node(n) => {
                    l.a.clone()
                        .substitute(var_vals)
                        .n2n_lt(n.clone().substitute(var_vals).deref())
                }
            },
            NodeTypes::MulNode(m) => match &m.b {
                BTypes::Int(i) => m.a.clone().substitute(var_vals).clone().deref() * i,
                BTypes::Node(n) => {
                    m.a.clone().substitute(var_vals).clone().deref()
                        * n.clone().substitute(var_vals).clone().deref()
                }
            },
            NodeTypes::DivNode(d) => d.a.clone().substitute(var_vals).floordiv(&d.b, true),
            NodeTypes::ModNode(m) => match &m.b {
                BTypes::Int(i) => m.a.clone().substitute(var_vals).deref() % i,
                BTypes::Node(n) => m.a.clone().substitute(var_vals).deref() % n.clone().deref(),
            },
            NodeTypes::SumNode(s) => Self::sum(
                &s.nodes
                    .iter()
                    .map(|n| n.clone().substitute(var_vals))
                    .collect(),
            ),
            NodeTypes::AndNode(a) => Self::ands(
                &a.nodes
                    .iter()
                    .map(|n| n.clone().substitute(var_vals))
                    .collect(),
            ),
        }
    }
    fn unbind(&self) -> (N, Option<isize>) {
        match self {
            NodeTypes::Variable(v) => {
                assert!(v.val.is_some());
                (self.ptr(), v.val.clone())
            }
            _ => {
                let mut map = HashMap::new();
                self.vars().iter().for_each(|v| match v.clone().deref() {
                    NodeTypes::Variable(var) => {
                        if var.val.is_some() {
                            map.insert(v.clone().ptr(), v.unbind().0).unwrap();
                        }
                    }
                    _ => {}
                });
                (self.substitute(&map), None)
            }
        }
    }

    fn key(&self) -> String {
        self.render(&None, &Some(DEBUG.clone()))
    }

    fn hash(&self) -> u64 {
        let mut s = DefaultHasher::new();
        self.key().hash(&mut s);
        s.finish()
    }

    fn sum(nodes: &Vec<N>) -> N {
        let mut nd = vec![];
        nodes.into_iter().for_each(|n| {
            if !n.clone().max().is_none() || !n.clone().min().is_none() {
                nd.push(n.clone());
            }
        });

        if nd.is_empty() {
            return NumNode::init(&0);
        }
        if nd.len() == 1 {
            return nd[0].clone();
        }

        let mut mul_groups: HashMap<N, BTypes> = HashMap::new();
        let mut num_node_sum: isize = 0;

        for nodes in &NodeTypes::new_sum(&nd).flat_components() {
            match nodes.clone().deref() {
                NodeTypes::NumNode(n) => {
                    num_node_sum = num_node_sum +n.b;
                }
                NodeTypes::MulNode(n) => match &n.b {
                    BTypes::Node(b_n) => {
                        let getter: &BTypes =
                            mul_groups.get(&n.a).unwrap_or_else(|| &BTypes::Int(0));

                        match getter {
                            BTypes::Int(i) => {
                                mul_groups
                                    .insert(n.a.clone(), BTypes::Node(b_n.clone().deref() + i));
                            }
                            BTypes::Node(bb_n) => {
                                mul_groups.insert(
                                    n.a.clone(),
                                    BTypes::Node(bb_n.clone().deref() + b_n.clone().deref()),
                                );
                            }
                        }
                    }
                    BTypes::Int(i) => {
                        let getter: &BTypes =
                            mul_groups.get(&n.a).unwrap_or_else(|| &BTypes::Int(0));
                        match getter {
                            BTypes::Int(ii) => {
                                mul_groups.insert(n.a.clone(), BTypes::Int(ii + i));
                            }
                            BTypes::Node(bb_n) => {
                                mul_groups
                                    .insert(n.a.clone(), BTypes::Node(bb_n.clone().deref() + i));
                            }
                        }
                    }
                },

                _ => {
                    let getter: &BTypes =
                        mul_groups.get(nodes).unwrap_or_else(|| &BTypes::Int(0));

                    match getter {
                        BTypes::Int(i) => {
                            mul_groups.insert(nodes.clone(), BTypes::Int(i + 1));
                        }
                        BTypes::Node(n) => {
                            mul_groups
                                .insert(nodes.clone(), BTypes::Node(n.clone().deref() + &1));
                        }
                    }
                }
            }
        }
        let mut new_nodes: Vec<N> = vec![];
        mul_groups.into_iter().for_each(|(a, b_sum)| match &b_sum {
            BTypes::Int(i) => {
                if i != &0 {
                    if i != &1 {
                        new_nodes.push(NodeTypes::new_mul(a, b_sum));
                    } else {
                        new_nodes.push(a);
                    }
                }
            }
            BTypes::Node(n) => {
                if n.clone().deref() != &0 {
                    if n.clone().deref() != &1 {
                        new_nodes.push(NodeTypes::new_mul(a, b_sum));
                    } else {
                        new_nodes.push(a);
                    }
                }
            }
        });

        if !num_node_sum == 0 {
            new_nodes.push(NumNode::init(&num_node_sum).ptr());
        }

        if new_nodes.len() > 1 {
            create_node(NodeTypes::new_sum(&new_nodes))
        } else if new_nodes.len() == 1 {
            create_node(new_nodes[0].clone())
        } else {
            create_node(NumNode::init(&0))
        }
    }

    fn ptr(&self) -> N {
        match self {
            NodeTypes::DivNode(n)
            | NodeTypes::LtNode(n)
            | NodeTypes::ModNode(n)
            | NodeTypes::MulNode(n) => n.ptr.borrow().as_ref().unwrap().upgrade().unwrap(),
            NodeTypes::AndNode(n) | NodeTypes::SumNode(n) => {
                n.ptr.borrow().as_ref().unwrap().upgrade().unwrap()
            }
            NodeTypes::Variable(n) => n.ptr.borrow().as_ref().unwrap().upgrade().unwrap(),
            NodeTypes::NumNode(n) => n.ptr.borrow().as_ref().unwrap().upgrade().unwrap(),
        }
    }

    fn min(&self) -> Option<isize> {
        match self {
            NodeTypes::DivNode(n)
            | NodeTypes::LtNode(n)
            | NodeTypes::ModNode(n)
            | NodeTypes::MulNode(n) => n.min.clone(),
            NodeTypes::AndNode(n) | NodeTypes::SumNode(n) => n.min.clone(),
            NodeTypes::Variable(n) => n.min.clone(),
            NodeTypes::NumNode(n) => n.min.clone(),
        }
    }

    fn max(&self) -> Option<BTypes> {
        match self {
            NodeTypes::DivNode(n)
            | NodeTypes::LtNode(n)
            | NodeTypes::ModNode(n)
            | NodeTypes::MulNode(n) => Some(n.max.clone()?.sint_infer().clone()),
            NodeTypes::AndNode(n) | NodeTypes::SumNode(n) => {
                Some(n.max.clone()?.sint_infer().clone())
            }
            NodeTypes::Variable(n) => Some(n.max.clone()?.sint_infer().clone()),
            NodeTypes::NumNode(n) => Some(n.max.clone()?.sint_infer().clone()),
        }
    }

    fn flat_components(&self) -> Vec<N> {
        match self {
            NodeTypes::SumNode(s) => {
                let mut result = vec![];
                s.nodes.iter().for_each(|x| match x.clone().deref() {
                    NodeTypes::SumNode(_) => result.extend(x.clone().flat_components()),
                    _ => result.push(x.clone()),
                });
                result
            }
            _ => {
                panic!("Not for {:?}", self)
            }
        }
    }

    fn ands(nodes: &Vec<N>) -> N {
        if nodes.is_empty() {
            return NumNode::init(&0);
        }
        if nodes.len() == 0 {
            return nodes[0].clone();
        }
        let mut tmp = vec![];
        nodes.iter().for_each(|n| {
            if n.min().unwrap() != n.max().unwrap() {
                tmp.push(n.clone())
            }
        });

        if tmp.len() > 1 {
            return create_node(NodeTypes::new_and(&tmp));
        } else if tmp.len() == 1 {
            return create_node(tmp[0].clone());
        } else {
            create_node(NumNode::init(&1))
        }
    }

    fn get_bounds(&self) -> (Option<isize>, Option<BTypes>) {
        match self {
            NodeTypes::LtNode(l) => {
                if &l.b == l.a.clone().deref() {
                    return (Some(0), Some(BTypes::Int(0)));
                }
                match &l.b {
                    BTypes::Int(i) => {
                        if &l.a.clone().max().unwrap() < i {
                            return (Some(1), Some(BTypes::Int(1)));
                        } else if i <= &l.a.clone().min().unwrap() {
                            return (Some(0), Some(BTypes::Int(0)));
                        } else {
                            return (Some(0), Some(BTypes::Int(1)));
                        }
                    }
                    BTypes::Node(n) => {
                        if l.a.clone().max().unwrap() < n.clone().min().unwrap() {
                            return (Some(1), Some(BTypes::Int(1)));
                        } else if n.clone().max().unwrap() <= l.a.clone().min().unwrap() {
                            return (Some(0), Some(BTypes::Int(0)));
                        } else {
                            return (Some(0), Some(BTypes::Int(0)));
                        }
                    }
                }
            }
            NodeTypes::MulNode(m) => {
                assert!(m.a.clone().min().unwrap() >= 0);
                // match m.a.clone().max().unwrap(){
                //     BTypes::Int(i) => {
                //         match &m.b{
                //             BTypes::Int(ii) => {
                //                 if ii>= &0{
                //                     return (Some(m.a.clone().min().unwrap() * ii), Some(BTypes::Int(i*ii)))
                //                 } else {
                //                     return (Some(i * ii), Some(BTypes::Int(m.a.clone().min().unwrap() * ii)))
                //                 }
                //             }
                //             BTypes::Node(ii) => {
                //                 if ii.clone().n2i_ge(&0).deref().into(){
                //                     return (Some(m.a.clone().min().unwrap() * ii.min().unwrap()), Some(BTypes::Node(ii.clone().deref() * &i)))
                //                 } else {
                //                     return (Some(i * ii.clone().min().unwrap()), Some(BTypes::Node(ii.clone().deref() * &m.a.min().unwrap())))
                //                 }
                //             }
                //         }
                //     }
                //     BTypes::Node(i) => {
                //         match &m.b{
                //             BTypes::Int(ii) => {
                //                 if ii>= &0{
                //                     return (Some(m.a.clone().min().unwrap() * ii), Some(BTypes::Int(i*ii)))
                //                 } else {
                //                     return (Some(i * ii), Some(BTypes::Int(m.a.clone().min().unwrap() * ii)))
                //                 }
                //             }
                //         }
                //     }
                // }

                match (m.a.clone().max().unwrap(), &m.b){
                    (BTypes::Int(i), BTypes::Int(ii)) => {
                        if ii >= &0{
                            return (Some(&m.a.clone().min().unwrap() * ii), Some(BTypes::Int(i * ii)))
                        } else{
                            return (Some(i * ii), Some(BTypes::Int(m.a.clone().min().unwrap() * ii)))
                        }
                    }
                    (BTypes::Node(i), BTypes::Int(ii)) => {
                        if i.clone().min().unwrap() >= 0{
                            return (Some(&m.a.clone().min().unwrap() * ii), Some(BTypes::Node(i.clone().deref() * ii)))
                        } else{
                            // return (Some(i.clone().deref() * ii), Some(BTypes::Int(m.a.clone().min().unwrap() * ii)))
                            panic!("Sint has int max values and a tuple of int, sint has to be infered");
                        }
                    }
                    (BTypes::Int(i), BTypes::Node(ii)) => {
                        if ii.clone().min().unwrap() >= 0{
                            match ii.clone().max().unwrap(){
                                BTypes::Int(iii) => {
                                    return (Some(m.a.clone().min().unwrap() * ii.clone().min().unwrap()), Some(BTypes::Int(i * iii)))
                                }
                                BTypes::Node(n) => {
                                    return (Some(m.a.clone().min().unwrap() * ii.clone().min().unwrap()), Some(BTypes::Node(n.clone().deref()*&i)))
                                }
                            }
                        }
                        else{
                            match ii.clone().max().unwrap(){
                                BTypes::Int(iii) => {
                                    return (Some(i * ii.clone().min().unwrap()), Some(BTypes::Int(m.a.clone().min().unwrap() * iii)))
                                }
                                BTypes::Node(n) => {
                                    return (Some(i * ii.clone().min().unwrap()), Some(BTypes::Node(n.clone().deref() * &m.a.clone().min().unwrap())))
                                }
                            }
                        }
                    }
                    (BTypes::Node(i), BTypes::Node(ii)) => {
                        if ii.clone().min().unwrap() >= 0{
                            match ii.clone().max().unwrap(){
                                BTypes::Int(iii) => {
                                    return (Some(m.a.clone().min().unwrap() * ii.clone().min().unwrap()), Some(BTypes::Node(i.clone().deref() * &iii)))
                                }
                                BTypes::Node(n) => {
                                    return (Some(m.a.clone().min().unwrap() * ii.clone().min().unwrap()), Some(BTypes::Node(n.clone().deref()*i.clone().deref())))
                                }
                            }
                        }
                        else{
                            // match ii.clone().max().unwrap(){
                            //     BTypes::Int(iii) => {
                            //         return (Some(i.clone().deref() * &ii.clone().min().unwrap()), Some(BTypes::Int(m.a.clone().min().unwrap() * iii)))
                            //     }
                            //     BTypes::Node(n) => {
                            //         return (Some(i.clone().deref() * &ii.clone().min().unwrap()), Some(BTypes::Node(n.clone().deref() * &m.a.clone().min().unwrap())))
                            //     }
                            // }

                            panic!("Sint has int max values and a tuple of int, sint has to be infered")
                        }
                    }
                }
            }
            NodeTypes::DivNode(d) => {
                if let BTypes::Int(i) = &d.b{
                    assert!(d.a.clone().min().unwrap() >= 0);
                    match d.a.clone().max().unwrap(){
                        BTypes::Int(ii) => {
                            return (Some((d.a.clone().min().unwrap() / i)), Some(BTypes::Int((ii/i))))
                        }
                        BTypes::Node(n) => {
                            return (Some((d.a.clone().min().unwrap() / i)), Some(BTypes::Node(n.clone().floordiv(&d.b, true))))
                        }
                    }
                }
                panic!()
            }
            NodeTypes::ModNode(m) => {
                if let BTypes::Int(i) = &m.b{
                    assert!(m.a.clone().min().unwrap() >= 0);
                    match &m.a.clone().max().unwrap(){
                        BTypes::Int(ii) => {
                            if (ii - &m.a.clone().min().unwrap() >= i.clone()) || (&m.a.clone().min().unwrap() != ii && m.a.clone().min().unwrap() % i >= ii % i){
                                return (Some(0), Some(BTypes::Int(i - 1)))
                            }
                            return (Some(m.a.clone().min().unwrap() % i), Some(BTypes::Int(ii % i)))
                        }
                        BTypes::Node(ii) => {
                            if ((ii.clone().deref() - &m.a.clone().min().unwrap()).n2i_ge(i).deref().into()) || (ii.clone().deref() == &m.a.clone().min().unwrap() && (&m.a.clone().min().unwrap() % i.i2n_ge((ii.clone().deref() % i).deref()).deref()).deref().into()){
                                return (Some(0), Some(BTypes::Int(i - 1)))
                            }
                            return (Some(m.a.clone().min().unwrap() % i), Some(BTypes::Node(ii.clone().deref() % i)))
                        }
                    }
                }
                panic!()
            }

            NodeTypes::SumNode(s) => {
                let mut int_sum = 0;
                let mut node_sum: Option<Rc<NodeTypes>> = None;
            
                for x in &s.nodes {
                    match x.max().unwrap() {
                        BTypes::Int(i) => int_sum += i,
                        BTypes::Node(n) => node_sum = Some(node_sum.unwrap_or_else(|| NumNode::init(&0)).deref() + n.clone().deref()),
                    }
                }
            
                let min_sum: isize = s.nodes.iter().map(|x| x.clone().min().unwrap()).sum();
                let result_node = node_sum.map(|node| BTypes::Node(node.deref() + &int_sum));
            
                (Some(min_sum), result_node)
            }

            NodeTypes::AndNode(a) => {
                (Some(a.nodes.iter().map(|x| x.clone().min().unwrap()).min().unwrap()), Some(btypes_max(a.nodes.iter().map(|x|x.clone().max().unwrap()).collect())))
            }

            _ => {
                panic!()
            }
        }
    }
}

impl Debug for BTypes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BTypes::Int(i) => write!(f, "{}", i),
            BTypes::Node(n) => write!(f, "{:?}", Rc::clone(n).deref()),
        }
    }
}

fn sym_render(a: &BTypes, ops: &Option<Box<dyn Any>>, ctx: &Option<Arc<Mutex<ContextVar>>>) -> String {
    match a {
        BTypes::Int(i) => {
            format!("{}", i)
        }
        BTypes::Node(n) => n.clone().render(&ops, &ctx),
    }
}


pub fn sym_infer(a: &BTypes, var_vals: HashMap<Rc<NodeTypes>, isize>) -> isize{
    match a{
        BTypes::Node(n) => {
            let mut hm: HashMap<N, N> =  HashMap::new();
            var_vals.iter().for_each(|(k, v)| {hm.insert(k.clone(), NumNode::init(v));});
            let ret = n.clone().substitute(&hm);

            if let NodeTypes::NumNode(nn) = ret.deref(){
                return nn.b
            }else{
                panic!("{}", format!("sym infer didnt produce NumNode from {:?} with {:?}", n.deref(), var_vals))
            }
        },
        BTypes::Int(i) => {
            i.clone()
        }
    }
    
}
fn render_mulnode(m: &OpNode, ops: &Option<Box<dyn Any>>, ctx: &Option<Arc<Mutex<ContextVar>>>) -> String {
    match m.a.clone().deref() {
        NodeTypes::Variable(v_a) => match &m.b {
            BTypes::Node(n) => match n.clone().deref() {
                NodeTypes::Variable(v_b) => {
                    if &v_b.expr < &v_a.expr {
                        format!(
                            "{} * {}",
                            sym_render(&m.b, ops, ctx),
                            m.a.clone().deref().render(ops, ctx)
                        )
                    } else {
                        format!(
                            "{} * {}",
                            m.a.clone().deref().render(ops, ctx),
                            sym_render(&m.b, ops, ctx)
                        )
                    }
                }
                _ => {
                    format!(
                        "{} * {}",
                        m.a.clone().deref().render(ops, ctx),
                        sym_render(&m.b, ops, ctx)
                    )
                }
            },
            _ => format!(
                "{} * {}",
                m.a.clone().deref().render(ops, ctx),
                sym_render(&m.b, ops, ctx)
            ),
        },
        _ => format!(
            "{} * {}",
            m.a.clone().deref().render(ops, ctx),
            sym_render(&m.b, ops, ctx)
        ),
    }
}

fn btypes_max(b: Vec<BTypes>) -> BTypes {
    // Check if the vector is empty
    if b.is_empty() {
        // Return a default value or handle the empty case as needed
        // For now, I'll return an arbitrary default value (you may want to change this)
        return BTypes::Int(0);
    }

    let mut max = b[0].clone();

    for i in &b[1..] {
        match i {
            BTypes::Int(ii) => match &max {
                BTypes::Int(iii) => {
                    if ii > iii {
                        max = i.clone();
                    }
                }
                BTypes::Node(nnn) => {
                    if ii.i2n_gt(nnn.clone().deref()).deref().into() {
                        max = i.clone();
                    }
                }
            },
            BTypes::Node(nn) => match &max {
                BTypes::Int(iii) => {
                    if nn.n2i_gt(iii).deref().into() {
                        max = i.clone();
                    }
                }
                BTypes::Node(nnn) => {
                    if nn.n2n_gt(nnn.clone().deref()).deref().into() {
                        max = i.clone();
                    }
                }
            },
        }
    }

    max
}
impl NodeTypes {
    fn new_lt(a: N, b: BTypes) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::LtNode(OpNode {
            a: a.clone(),
            b: b.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::LtNode(OpNode {
            a,
            b,
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::LtNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }

    pub fn new_ge(lhs: N, b: BTypes) -> N{
        create_lt_node(-lhs.deref(), &-&b - &BTypes::Int(1))
    }
    fn new_mul(a: N, b: BTypes) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::MulNode(OpNode {
            a: a.clone(),
            b: b.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::MulNode(OpNode {
            a,
            b,
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::MulNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }

    fn new_div(a: N, b: BTypes) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::DivNode(OpNode {
            a: a.clone(),
            b: b.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::DivNode(OpNode {
            a,
            b,
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::DivNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }

    fn new_mod(a: N, b: BTypes) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::ModNode(OpNode {
            a: a.clone(),
            b: b.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::ModNode(OpNode {
            a,
            b,
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::ModNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }

    fn new_sum(a: &Vec<N>) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::SumNode(RedNode {
            nodes: a.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::SumNode(RedNode {
            nodes: a.clone(),
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::SumNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }
    fn new_and(a: &Vec<N>) -> N {
        let (min, max) = NodeTypes::get_bounds(&NodeTypes::AndNode(RedNode {
            nodes: a.clone(),
            min: None,
            max: None,
            ptr: RefCell::new(None),
        }));

        let nd = Rc::new(NodeTypes::AndNode(RedNode {
            nodes: a.clone(),
            min,
            max,
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::AndNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }
    pub fn floordiv(&self, b: &BTypes, factoring_allowed: bool) -> Rc<Self> {
        match self {
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Int(i) => match b {
                    BTypes::Int(ii) => {
                        if i % ii == 0 {
                            n.a.clone().deref() * &((i / ii))
                        } else if ii % i == 0 && i > &0 {
                            n.a.floordiv(&BTypes::Int((ii / i)), true)
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                    BTypes::Node(nn) => {
                        if (i % nn.clone().deref()).deref() == &0 {
                            n.a.clone().deref() * (NodeTypes::rfloordiv(i, b)).deref()
                        } else if (nn.clone().deref() % i).deref() == &0 && i > &0 {
                            n.a.clone().floordiv(
                                &BTypes::Node(nn.clone().floordiv(&n.b, factoring_allowed)),
                                true,
                            )
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                },
                BTypes::Node(i) => match b {
                    BTypes::Int(ii) => {
                        if (i.clone().deref() % ii).deref() == &0 {
                            n.a.clone().deref() * (i.clone().floordiv(b, true)).deref()
                        } else if (ii % i.clone().deref()).deref() == &0
                            && i.n2i_gt(&0).deref().into()
                        {
                            n.a.clone()
                                .floordiv(&BTypes::Node(NodeTypes::rfloordiv(ii, &n.b)), true)
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                    BTypes::Node(nn) => {
                        if (i.clone().deref() % nn.clone().deref()).deref() == &0 {
                            n.a.clone().deref() * i.floordiv(b, true).deref()
                        } else if (nn.clone().deref() % i.clone().deref()).deref() == &0
                            && i.n2i_gt(&0).deref().into()
                        {
                            n.a.clone().floordiv(
                                &BTypes::Node(nn.clone().floordiv(&n.b, factoring_allowed)),
                                true,
                            )
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                },
            },
            NodeTypes::DivNode(n) => match &n.b {
                BTypes::Int(i) => match b {
                    BTypes::Int(ii) => {
                        n.a.clone()
                            .floordiv(&BTypes::Int(ii * i), factoring_allowed)
                    }
                    BTypes::Node(ii) => {
                        n.a.clone()
                            .floordiv(&BTypes::Node(ii.clone().deref() * i), factoring_allowed)
                    }
                },
                BTypes::Node(i) => match b {
                    BTypes::Int(ii) => {
                        n.a.clone()
                            .floordiv(&BTypes::Node(i.clone().deref() * ii), factoring_allowed)
                    }
                    BTypes::Node(ii) => n.a.clone().floordiv(
                        &BTypes::Node(ii.clone().deref() * i.clone().deref()),
                        factoring_allowed,
                    ),
                },
            },
            NodeTypes::ModNode(n) => match b {
                BTypes::Int(i) => match &n.b {
                    BTypes::Int(ii) => {
                        if ii % i == 0 {
                            n.a.clone().floordiv(b, true).deref() % &(ii / i)
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                    BTypes::Node(nn) => {
                        if (nn.clone().deref() % i).deref() == &0 {
                            n.a.clone().floordiv(b, factoring_allowed).deref()
                                % nn.clone().floordiv(b, true).deref()
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                },
                BTypes::Node(i) => match &n.b {
                    BTypes::Int(ii) => {
                        if (ii % i.clone().deref()).deref() == &0 {
                            n.a.clone().floordiv(b, factoring_allowed).deref()
                                % NodeTypes::rfloordiv(ii, b).deref()
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                    BTypes::Node(nn) => {
                        if (nn.clone().deref() % i.clone().deref()).deref() == &0 {
                            n.a.clone().floordiv(b, factoring_allowed).deref()
                                % nn.clone().floordiv(b, factoring_allowed).deref()
                        } else {
                            self.floordiv(b, factoring_allowed)
                        }
                    }
                },
            },
            NodeTypes::SumNode(_) => match b {
                BTypes::Int(i) => {
                    if self == i {
                        return NumNode::init(&1).ptr();
                    }

                    let mut fully_divided: Vec<N> = vec![];

                    let mut rest: Vec<N> = vec![];
                    if i == &1 {
                        return self.ptr();
                    }
                    if !factoring_allowed {
                        return self.floordiv(b, factoring_allowed);
                    }
                    let mut _gcd = i.clone();
                    let mut divisior = 1;

                    self.flat_components()
                        .iter()
                        .for_each(|n| match n.clone().deref() {
                            NodeTypes::NumNode(x) => {
                                if &x.b % i == 0 {
                                    fully_divided.push(n.clone().floordiv(b, factoring_allowed));
                                } else {
                                    rest.push(n.clone());
                                    _gcd = num::integer::gcd(
                                        _gcd,
                                        x.b.clone(),
                                    );
                                }
                            }
                            NodeTypes::MulNode(x) => match &x.b {
                                BTypes::Int(ii) => {
                                    if ii % i == 0 {
                                        fully_divided
                                            .push(n.clone().floordiv(b, factoring_allowed));
                                    } else {
                                        rest.push(n.clone());
                                        match &x.b {
                                            BTypes::Int(iii) => {
                                                _gcd = num::integer::gcd(
                                                    _gcd.clone(),
                                                    iii.clone(),
                                                );
                                                if divisior == 1 && i % ii == 0 {
                                                    divisior = iii.clone()
                                                }
                                            }
                                            _ => {
                                                _gcd = 1;
                                            }
                                        }
                                    }
                                }
                                BTypes::Node(_) => {
                                    _gcd = 1;
                                }
                            },
                            _ => {
                                rest.push(n.clone());
                                _gcd = 1;
                            }
                        });
                    if _gcd > 1 {
                        return (NodeTypes::sum(&fully_divided).deref()
                            + NodeTypes::sum(&rest)
                                .floordiv(&BTypes::Int(_gcd), factoring_allowed)
                                .deref())
                        .floordiv(&BTypes::Int((i / _gcd)), factoring_allowed);
                    }
                    if divisior > 1 {
                        (NodeTypes::sum(&fully_divided).deref()
                            + NodeTypes::sum(&rest)
                                .floordiv(&BTypes::Int(divisior), factoring_allowed)
                                .deref())
                        .floordiv(&BTypes::Int((i / divisior)), factoring_allowed)
                    } else {
                        NodeTypes::sum(&fully_divided).deref()
                            + NodeTypes::sum(&rest).floordiv(b, factoring_allowed).deref()
                    }
                }
                BTypes::Node(n_b1) => {
                    if self == n_b1.clone().deref() {
                        return NumNode::init(&1);
                    }
                    let mut fully_divided: Vec<N> = vec![];

                    let mut rest: Vec<N> = vec![];

                    self.flat_components().into_iter().for_each(|x| {
                        if (x.clone().deref() % n_b1.clone().deref()).deref() == &0 {
                            fully_divided.push(x.floordiv(b, factoring_allowed));
                        } else {
                            rest.push(x)
                        }
                    });
                    let sum_fully_divided = create_node(NodeTypes::new_sum(&fully_divided));
                    if sum_fully_divided.clone().deref() != &0 {
                        return (sum_fully_divided.deref()
                            + create_node(NodeTypes::new_sum(&rest)).deref())
                        .floordiv(b, factoring_allowed);
                    }
                    self.floordiv(b, false)
                }
            },
            _ => match b {
                BTypes::Node(n) => match n.clone().deref() {
                    NodeTypes::NumNode(num) => {
                        self.floordiv(&BTypes::Int(num.b.clone()), factoring_allowed)
                    }
                    _ => {
                        if self == n.clone().deref() {
                            NumNode::init(&1)
                        } else if (n.clone().deref() - self).min().unwrap() > 0
                            && self.min().unwrap() >= 0
                        {
                            NumNode::init(&0)
                        } else {
                            panic!("Not supported: {:?}, {:?}", self, b)
                        }
                    }
                },
                BTypes::Int(i) => {
                    assert!(i != &0);
                    if i < &0 {
                        return self.floordiv(&BTypes::Int(-i), factoring_allowed).deref() * &-1;
                    }
                    if i == &1 {
                        return self.ptr();
                    }

                    if self.min().unwrap() < 0 {
                        let offset: isize = self.min().unwrap() / i.clone();
                        return (self + &(-&offset * i)).floordiv(b, false).deref() + &offset;
                    }
                    create_node(NodeTypes::new_div(self.ptr(), b.clone()))
                }
            },
        }
    }
    fn rfloordiv(a: &isize, b: &BTypes) -> N {
        NumNode::init(a).floordiv(b, true)
    }

    pub fn n2n_le(&self, other: &Self) -> N {
        self.n2n_lt(&(other + &1))
    }
    pub fn n2n_gt(&self, other: &NodeTypes) -> N {
        -(self.n2n_lt(&((-other).deref() + &1))).deref()
    }
    pub fn n2n_lt(&self, other: &NodeTypes) -> N {
        match self {
            NodeTypes::MulNode(_) => self.n2n_le(other),
            NodeTypes::SumNode(_) => self.n2n_lt(other),
            _ => create_node(Self::new_lt(self.ptr(), BTypes::Node(other.ptr()))),
        }
    }
    pub fn n2n_ge(&self, other: &NodeTypes) -> N {
        (-self).n2n_lt(((-other).deref() + &1).deref())
    }
    pub fn n2i_lt(&self, other: &isize) -> N {
        match self {
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Node(_) => self.n2i_lt(other),
                BTypes::Int(i) => {
                    if i == &-1 {
                        return self.n2i_lt(other);
                    }
                    let sgn;
                    if i > &0 {
                        sgn = 1;
                    } else {
                        sgn = -1;
                    }
                    (n.a.clone().deref() * &sgn)
                        .n2i_lt(&((other + i.clone().abs() - 1) / i.clone().abs()))
                }
            },
            NodeTypes::SumNode(s) => {
                let mut temp = other.clone();
                let mut new_sum: Vec<N> = vec![];
                let mut numn = other.clone();

                s.nodes.iter().for_each(|n| {
                    if let NodeTypes::NumNode(num) = n.clone().deref() {
                        numn = numn - num.b.clone();
                    } else {
                        new_sum.push(n.clone());
                    }
                });
                let mut lhs = NodeTypes::sum(&new_sum);
                let nodes;

                if let NodeTypes::SumNode(s) = lhs.clone().deref() {
                    nodes = s.nodes.clone();
                } else {
                    nodes = vec![lhs.clone()]
                }

                assert!(
                    nodes.iter().all(|nd| {
                        match nd.deref() {
                            NodeTypes::DivNode(n)
                            | NodeTypes::LtNode(n)
                            | NodeTypes::ModNode(n) => {
                                if let BTypes::Int(_) = &n.b {
                                    return true;
                                }
                                return false;
                            }
                            NodeTypes::NumNode(_) => return true,
                            NodeTypes::MulNode(n) => {
                                if let BTypes::Int(_) = &n.b {
                                    return true;
                                }
                                return false;
                            }
                            _ => return false,
                        }
                    }),
                    "Not Supported"
                );

                let (muls, others) = partition(nodes, |x| {
                    if let NodeTypes::MulNode(m) = x.clone().deref() {
                        match &m.b {
                            BTypes::Int(i) => {
                                if i > &0 && x.max().unwrap() >= *other {
                                    return true;
                                } else {
                                    return false;
                                }
                            }
                            BTypes::Node(n) => {
                                if n.n2i_gt(&0).deref().into() && x.max().unwrap() >= *other {
                                    return true;
                                } else {
                                    return false;
                                }
                            }
                        }
                    } else {
                        return false;
                    }
                });

                let mut mul_gcd = other.clone();

                muls.iter().for_each(|x| match x.clone().deref() {
                    NodeTypes::DivNode(n)
                    | NodeTypes::LtNode(n)
                    | NodeTypes::ModNode(n)
                    | NodeTypes::MulNode(n) => {
                        if let BTypes::Int(i) = &n.b {
                            mul_gcd = gcd(
                                mul_gcd.clone(),
                                i.clone(),
                            );
                        } else {
                            panic!("There is a bug here idiot");
                        }
                    }
                    NodeTypes::AndNode(_) | NodeTypes::SumNode(_) => {
                        panic!("There is a bug here idiot");
                    }
                    NodeTypes::NumNode(n) => {
                        mul_gcd = gcd(
                            mul_gcd,
                            n.b.clone(),
                        );
                    }

                    NodeTypes::Variable(_) => {
                        panic!("There is a bug here idiot");
                    }
                });

                let all_others = NodeTypes::sum(&others);
                if all_others.min().unwrap() >= 0 && all_others.max().unwrap() < mul_gcd {
                    lhs = NodeTypes::sum(
                        &muls
                            .into_iter()
                            .map(|v| v.floordiv(&BTypes::Int(mul_gcd), true))
                            .collect(),
                    );
                    temp = (temp / mul_gcd)
                }

                lhs.n2i_lt(&temp)
            }
            _ => create_node(Self::new_lt(self.ptr(), BTypes::Int(*other))),
        }
    }
    pub fn n2i_le(&self, other: &isize) -> N {
        self.n2i_lt(&(other + 1))
    }
    pub fn n2i_gt(&self, other: &isize) -> N {
        (-self).n2i_lt(&(-other))
    }

    pub fn n2i_ge(&self, other: &isize) -> N {
        (-self).n2i_lt(&(-other + 1))
    }

    fn recursive_max(&self) -> isize{
        match self.max().unwrap(){
            BTypes::Int(i) => {
                return i
            }
            BTypes::Node(n) => {
                NodeTypes::recursive_max(n.clone().deref())
            }
        }
    }
}
impl NumNode {
    pub fn init(a: &isize) -> N {
        let nd = Rc::new(NodeTypes::NumNode(NumNode {
            b: a.clone(),
            min: Some(a.clone()),
            max: Some(BTypes::Int(a.clone())),
            ptr: RefCell::new(None),
        }));

        if let NodeTypes::MulNode(l) = nd.clone().deref() {
            l.ptr
                .borrow_mut()
                .replace(Rc::downgrade(&nd));
        }

        nd
    }
}

impl Variable{
    pub fn new(expr: &String, val: &Option<isize>, min: &Option<isize>, max: &Option<BTypes>) -> N{
        let un_max = max.clone().unwrap();
        let un_min = min.clone().unwrap();
        match &un_max{
            BTypes::Int(i) => {
                assert!(i >= &0 && &un_min <= i);
                if &un_min == i{
                    return NumNode::init(&un_min)
                } 
            }
            BTypes::Node(n) => {
                assert!(&un_min >= &0 && un_min.borrow().i2n_le(n.clone().deref()).deref().into(), "{}", format!("invalid Variable {:?}, {:?}, {:?}", expr, min, max));
                if n.clone().deref()==&un_min{
                    return NumNode::init(&un_min)
                }
            }
        }
        
        let nd = Rc::new(NodeTypes::Variable(Variable { expr: expr.clone(), min: Some(un_min), max: Some(un_max), val: val.clone(), ptr:RefCell::new(None) }));

        if let NodeTypes::Variable(v) = nd.clone().deref(){
            v.ptr.borrow_mut().replace(Rc::downgrade(&nd));
        }

        nd
    }

    pub fn init(expr: String, nmin: isize, nmax: &BTypes) -> Rc<NodeTypes>{

        
        let nd = Rc::new(NodeTypes::Variable(Variable { expr: expr, min: Some(nmin), max: Some(nmax.clone()), val: None, ptr:RefCell::new(None) }));

        if let NodeTypes::Variable(v) = nd.clone().deref(){
            v.ptr.borrow_mut().replace(Rc::downgrade(&nd));
        }

        nd
    }
    pub fn val(&self) -> Option<isize>{
        assert!(self.val.is_some(), "{}", format!("Varible isn't bound can't access val of {:?}", self));
        self.val.clone()
    }
}
trait I2Ncmp {
    fn i2n_lt(&self, other: &NodeTypes) -> N;
    fn i2n_le(&self, other: &NodeTypes) -> N;
    fn i2n_gt(&self, other: &NodeTypes) -> N;
    fn i2n_ge(&self, other: &NodeTypes) -> N;
}

impl I2Ncmp for isize {
    fn i2n_lt(&self, other: &NodeTypes) -> N {
        other.n2i_gt(self)
    }
    fn i2n_le(&self, other: &NodeTypes) -> N {
        other.n2i_gt(&(self + 1))
    }
    fn i2n_gt(&self, other: &NodeTypes) -> N {
        (-other).n2i_gt(&(-self))
    }
    fn i2n_ge(&self, other: &NodeTypes) -> N {
        (-other).n2i_gt(&(-self + 1))
    }
}

fn create_node(ret: N) -> N {
    assert!(ret.max().unwrap() >= ret.min().unwrap());
    if ret.min().unwrap() == ret.max().unwrap() {
        NumNode::init(&ret.min().unwrap())
    } else {
        ret
    }
}

impl Into<bool> for &NodeTypes {
    fn into(self) -> bool {
        match self {
            NodeTypes::DivNode(_)
            | NodeTypes::LtNode(_)
            | NodeTypes::ModNode(_)
            | NodeTypes::MulNode(_) => {
                !(self.min().unwrap() == self.max().unwrap() && self.min().unwrap() == 0)
            }
            _ => !(self.min().unwrap() == self.max().unwrap() && self.min().unwrap() == 0),
        }
    }
}

impl PartialEq<NodeTypes> for NodeTypes {
    fn eq(&self, other: &NodeTypes) -> bool {
        self.key() == other.key()
    }
}

impl PartialEq<isize> for BTypes {
    fn eq(&self, other: &isize) -> bool {
        match self {
            BTypes::Int(i) => i == other,
            BTypes::Node(n) => n.clone().deref() == other,
        }
    }
}

impl PartialEq<BTypes> for isize {
    fn eq(&self, other: &BTypes) -> bool {
        match other {
            BTypes::Int(i) => i == self,
            BTypes::Node(n) => n.clone().deref() == self,
        }
    }
}
impl<'a> std::ops::Neg for &'a NodeTypes {
    type Output = N;

    fn neg(self) -> Self::Output {
        self * &-1
    }
}

impl<'a> std::ops::Add<&'a NodeTypes> for &'a NodeTypes {
    type Output = N;

    fn add(self, rhs: Self) -> Self::Output {
        NodeTypes::sum(&vec![self.ptr(), rhs.ptr()])
    }
}

impl<'a> std::ops::Add<&'a isize> for &'a NodeTypes {
    type Output = N;

    fn add(self, rhs: &'a isize) -> Self::Output {
        NodeTypes::sum(&vec![self.ptr(), NumNode::init(&rhs)])
    }
}

impl<'a> std::ops::Sub<&'a isize> for &'a NodeTypes {
    type Output = N;

    fn sub(self, rhs: &'a isize) -> Self::Output {
        (self) + &-rhs
    }
}

impl<'a> std::ops::Sub<&'a NodeTypes> for &'a NodeTypes {
    type Output = N;

    fn sub(self, rhs: &'a NodeTypes) -> Self::Output {
        self + (-rhs).deref()
    }
}

impl<'a> std::ops::Sub<&'a NodeTypes> for &'a isize {
    type Output = N;

    fn sub(self, rhs: &'a NodeTypes) -> Self::Output {
        rhs - self
    }
}

impl PartialEq<isize> for NodeTypes {
    fn eq(&self, other: &isize) -> bool {
        NumNode::init(other).deref() == self
    }
}

impl <'a> Sub<&'a BTypes> for &'a BTypes{
    type Output = BTypes;

    fn sub(self, rhs: &'a BTypes) -> Self::Output {
        match (self, rhs){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int(i - ii)
            }
            (BTypes::Int(i), BTypes::Node(n)) => {
                BTypes::Node(i - n.clone().deref())
            }
            (BTypes::Node(n), BTypes::Int(i)) => {
                BTypes::Node(n.clone().deref() - i)
            }
            (BTypes::Node(n), BTypes::Node(nn)) => {
                BTypes::Node(n.clone().deref() - nn.clone().deref())
            }
        }
    }
}

impl <'a> std::ops::Add<&'a BTypes> for &'a BTypes{
    type Output = BTypes;

    fn add(self, rhs: &'a BTypes) -> Self::Output {
        match (self, rhs){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int(i + ii)
            }
            (BTypes::Int(i), BTypes::Node(n)) => {
                BTypes::Node(n.clone().deref() + i)
            }
            (BTypes::Node(n), BTypes::Int(i)) => {
                BTypes::Node(n.clone().deref() + i)
            }
            (BTypes::Node(n), BTypes::Node(nn)) => {
                BTypes::Node(n.clone().deref() + nn.clone().deref())
            }
        }
    }
}

impl <'a> std::ops::Add<&'a NodeTypes> for &'a BTypes{
    type Output = N;

    fn add(self, rhs: &'a NodeTypes) -> Self::Output {
        match &self{
            BTypes::Int(i) => rhs + i,
            BTypes::Node(n) => n.clone().deref() + rhs
        }
    }
}
impl <'a> std::ops::Mul<&'a NodeTypes> for &'a BTypes{
    type Output = N;

    fn mul(self, rhs: &'a NodeTypes) -> Self::Output {
        match &self{
            BTypes::Int(i) => rhs *i,
            BTypes::Node(n) => n.clone().deref() * rhs
        }
    }
}
impl <'a > std::ops::Add<&'a BTypes> for &'a NodeTypes{
    type Output = N;
    fn add(self, rhs: &'a BTypes) -> Self::Output {
        match &rhs{
            BTypes::Int(i) => self + i,
            BTypes::Node(n) => n.clone().deref() + self
        }
    }
}
impl <'a > std::ops::Mul<&'a BTypes> for &'a NodeTypes{
    type Output = N;
    fn mul(self, rhs: &'a BTypes) -> Self::Output {
        match &rhs{
            BTypes::Int(i) => self * i,
            BTypes::Node(n) => n.clone().deref() * self
        }
    }
}
impl <'a> std::ops::Mul<&'a BTypes> for &'a BTypes{
    type Output = BTypes;

    fn mul(self, rhs: &'a BTypes) -> Self::Output {
        match (self, rhs){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int(i * ii)
            }
            (BTypes::Int(i), BTypes::Node(n)) => {
                BTypes::Node(n.clone().deref() * i)
            }
            (BTypes::Node(n), BTypes::Int(i)) => {
                BTypes::Node(n.clone().deref() * i)
            }
            (BTypes::Node(n), BTypes::Node(nn)) => {
                BTypes::Node(n.clone().deref() * nn.clone().deref())
            }
        }
    }
}

impl <'a> std::ops::Div<&'a BTypes> for &'a BTypes{
    type Output = BTypes;

    fn div(self, rhs: &'a BTypes) -> Self::Output {
        match (self, rhs){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int(i / ii)
            }
            _ => unreachable!()
        }
    }
}
impl Eq for BTypes{}
impl Ord for BTypes{
    fn max(self, other: Self) -> Self
        where
            Self: Sized, {
        if self > other{
            self
        } else{
            other
        }
    }

    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        unimplemented!()
    }
}
impl <'a> std::ops::Rem<&'a BTypes> for &'a BTypes{
    type Output = BTypes;

    fn rem(self, rhs: &'a BTypes) -> Self::Output {
        match (self, rhs){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int(i % ii)
            }
            (BTypes::Int(i), BTypes::Node(n)) => {
                BTypes::Node(n.clone().deref() % i)
            }
            (BTypes::Node(n), BTypes::Int(i)) => {
                BTypes::Node(n.clone().deref() % i)
            }
            (BTypes::Node(n), BTypes::Node(nn)) => {
                BTypes::Node(n.clone().deref() % nn.clone().deref())
            }
        }
    }
}
impl PartialEq<BTypes> for BTypes {
    fn eq(&self, other: &BTypes) -> bool {
        match (self, other) {
            (BTypes::Int(i), BTypes::Int(ii)) => i == ii,
            (BTypes::Node(n), BTypes::Int(ii)) => n.clone().deref() == ii,
            (BTypes::Int(i), BTypes::Node(nn)) => nn.clone().deref() == i,
            (BTypes::Node(n), BTypes::Node(nn)) => n.clone().deref() == nn.clone().deref(),
        }
    }
}
impl Eq for NodeTypes {}

impl PartialOrd<isize> for BTypes {
    fn ge(&self, other: &isize) -> bool {
        match self {
            BTypes::Int(i) => i >= other,
            BTypes::Node(n) => n.n2i_ge(other).deref().into(),
        }
    }
    fn gt(&self, other: &isize) -> bool {
        match self {
            BTypes::Int(i) => i > other,
            BTypes::Node(n) => n.n2i_gt(other).deref().into(),
        }
    }
    fn le(&self, other: &isize) -> bool {
        match self {
            BTypes::Int(i) => i <= other,
            BTypes::Node(n) => n.n2i_le(other).deref().into(),
        }
    }
    fn lt(&self, other: &isize) -> bool {
        match self {
            BTypes::Int(i) => i < other,
            BTypes::Node(n) => n.n2i_lt(other).deref().into(),
        }
    }

    fn partial_cmp(&self, _other: &isize) -> Option<std::cmp::Ordering> {
        unimplemented!()
    }
}

impl PartialEq<NodeTypes> for BTypes {
    fn eq(&self, other: &NodeTypes) -> bool {
        match self {
            BTypes::Int(i) => other == i,
            BTypes::Node(n) => other == n.clone().deref(),
        }
    }
}

impl PartialOrd<BTypes> for BTypes {
    fn ge(&self, other: &BTypes) -> bool {
        match (self, other) {
            (BTypes::Int(i), BTypes::Int(ii)) => i >= ii,
            (BTypes::Node(n), BTypes::Int(ii)) => n.clone().n2i_ge(ii).deref().into(),
            (BTypes::Int(i), BTypes::Node(nn)) => i.i2n_ge(nn.clone().deref()).deref().into(),
            (BTypes::Node(n), BTypes::Node(nn)) => {
                n.clone().n2n_ge(nn.clone().deref()).deref().into()
            }
        }
    }
    fn gt(&self, other: &BTypes) -> bool {
        match (self, other) {
            (BTypes::Int(i), BTypes::Int(ii)) => i > ii,
            (BTypes::Node(n), BTypes::Int(ii)) => n.clone().n2i_gt(ii).deref().into(),
            (BTypes::Int(i), BTypes::Node(nn)) => i.i2n_gt(nn.clone().deref()).deref().into(),
            (BTypes::Node(n), BTypes::Node(nn)) => {
                n.clone().n2n_gt(nn.clone().deref()).deref().into()
            }
        }
    }
    fn le(&self, other: &BTypes) -> bool {
        match (self, other) {
            (BTypes::Int(i), BTypes::Int(ii)) => i <= ii,
            (BTypes::Node(n), BTypes::Int(ii)) => n.clone().n2i_le(ii).deref().into(),
            (BTypes::Int(i), BTypes::Node(nn)) => i.i2n_le(nn.clone().deref()).deref().into(),
            (BTypes::Node(n), BTypes::Node(nn)) => {
                n.clone().n2n_le(nn.clone().deref()).deref().into()
            }
        }
    }
    fn lt(&self, other: &BTypes) -> bool {
        match (self, other) {
            (BTypes::Int(i), BTypes::Int(ii)) => i < ii,
            (BTypes::Node(n), BTypes::Int(ii)) => n.clone().n2i_lt(ii).deref().into(),
            (BTypes::Int(i), BTypes::Node(nn)) => i.i2n_lt(nn.clone().deref()).deref().into(),
            (BTypes::Node(n), BTypes::Node(nn)) => {
                n.clone().n2n_lt(nn.clone().deref()).deref().into()
            }
        }
    }

    fn partial_cmp(&self, other: &BTypes) -> Option<std::cmp::Ordering> {
        unimplemented!()
    }
}
impl PartialOrd<NodeTypes> for BTypes {
    fn gt(&self, other: &NodeTypes) -> bool {
        match self {
            BTypes::Int(i) => i.i2n_gt(other).deref().into(),
            BTypes::Node(n) => n.n2n_gt(other).deref().into(),
        }
    }
    fn ge(&self, other: &NodeTypes) -> bool {
        match self {
            BTypes::Int(i) => i.i2n_ge(other).deref().into(),
            BTypes::Node(n) => n.n2n_ge(other).deref().into(),
        }
    }
    fn le(&self, other: &NodeTypes) -> bool {
        match self {
            BTypes::Int(i) => i.i2n_le(other).deref().into(),
            BTypes::Node(n) => n.n2n_le(other).deref().into(),
        }
    }

    fn lt(&self, other: &NodeTypes) -> bool {
        match self {
            BTypes::Int(i) => i.i2n_lt(other).deref().into(),
            BTypes::Node(n) => n.n2n_lt(other).deref().into(),
        }
    }
    fn partial_cmp(&self, other: &NodeTypes) -> Option<std::cmp::Ordering> {
        unimplemented!()
    }
}
impl BTypes {
    fn sint_infer(&self) -> &Self {
        match self {
            BTypes::Node(n) => match n.clone().deref() {
                NodeTypes::AndNode(_)
                | NodeTypes::LtNode(_)
                | NodeTypes::DivNode(_)
                | NodeTypes::ModNode(_)
                | NodeTypes::NumNode(_) => {
                    panic!("Unexpected node type")
                }
                _ => return self,
            },
            _ => return self,
        }
    }

    pub fn floordiv(&self, b: &BTypes, factoring_allowed: bool) -> BTypes{
        match (self, b){
            (BTypes::Int(i), BTypes::Int(ii)) => {
                BTypes::Int((i /ii))
            }
            (BTypes::Int(i), BTypes::Node(n)) => {
                BTypes::Node(NodeTypes::rfloordiv(i, b))
            }
            (BTypes::Node(n), BTypes::Int(i)) => {
                BTypes::Node(n.clone().floordiv(b, true))
            }
            (BTypes::Node(n), BTypes::Node(nn)) => {
                BTypes::Node(n.clone().floordiv(b, true))
            }
        }
    }
}

impl Mul<&NodeTypes> for &NodeTypes {
    type Output = N;

    fn mul(self, rhs: &NodeTypes) -> Self::Output {
        match &self {
            NodeTypes::NumNode(n) => rhs * &n.b,
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Int(i) => n.a.clone().deref() * (rhs * i).deref(),
                BTypes::Node(nn) => n.a.clone().deref() * (nn.clone().deref() * rhs).deref(),
            },
            NodeTypes::SumNode(s) => NodeTypes::sum(
                &s.nodes
                    .iter()
                    .map(|n| n.clone().deref() * rhs)
                    .collect::<Vec<N>>(),
            ),
            _ => match rhs {
                NodeTypes::NumNode(n) => {
                    create_node(NodeTypes::new_mul(self.ptr(), BTypes::Int(n.b.clone())))
                }
                _ => create_node(NodeTypes::new_mul(self.ptr(), BTypes::Node(rhs.ptr()))),
            },
        }
    }
}

impl Mul<&isize> for &NodeTypes {
    type Output = N;

    fn mul(self, rhs: &isize) -> Self::Output {
        match &self {
            NodeTypes::NumNode(n) => NumNode::init(&(n.b * rhs)),
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Int(i) => n.a.clone().deref() * &(i * rhs),
                BTypes::Node(nn) => n.a.clone().deref() * (nn.clone().deref() * rhs).deref(),
            },
            NodeTypes::SumNode(s) => NodeTypes::sum(
                &s.nodes
                    .iter()
                    .map(|n| n.clone().deref() * rhs)
                    .collect::<Vec<N>>(),
            ),
            _ => {
                if rhs == &0 {
                    return NumNode::init(&0);
                } else if rhs == &1 {
                    return self.ptr();
                } else {
                    create_node(NodeTypes::new_mul(self.ptr(), BTypes::Int(rhs.clone())))
                }
            }
        }
    }
}
impl Rem<&NodeTypes> for &NodeTypes {
    type Output = N;

    fn rem(self, rhs: &NodeTypes) -> Self::Output {
        match rhs {
            NodeTypes::NumNode(n) => self % &n.b,
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Int(i) => (n.a.clone().deref() * (i % rhs).deref()).deref() % rhs,
                BTypes::Node(i) => {
                    (n.a.clone().deref() * (i.clone().deref() % rhs).deref()).deref() % rhs
                }
            },
            NodeTypes::ModNode(_) => self % rhs,
            NodeTypes::SumNode(s) => {
                if self == rhs {
                    return NumNode::init(&0);
                }
                if (rhs - self).min().unwrap() > 0 {
                    return self.ptr();
                }
                let mut result = vec![];
                s.nodes.iter().for_each(|n| match n.clone().deref() {
                    NodeTypes::NumNode(_) => {
                        result.push(n.clone().deref() % rhs);
                    }
                    NodeTypes::MulNode(_) => {
                        result.push(n.clone().deref() % rhs);
                    }
                    _ => {
                        result.push(n.clone());
                    }
                });
                NodeTypes::sum(&result).deref() % rhs
            }
            _ => {
                if self == rhs {
                    return NumNode::init(&0);
                }
                if (rhs - self).min().unwrap() > 0 && self.min().unwrap() >= 0 {
                    self.ptr()
                } else {
                    panic!("Not supported: {:?} % {:?}", self, rhs);
                }
            }
        }
    }
}

impl Rem<&isize> for &NodeTypes {
    type Output = N;

    fn rem(self, rhs: &isize) -> Self::Output {
        match &self {
            NodeTypes::MulNode(n) => match &n.b {
                BTypes::Int(i) => (n.a.clone().deref() * &(i % rhs)).deref() % rhs,
                BTypes::Node(i) => {
                    (n.a.clone().deref() * (i.clone().deref() % rhs).deref()).deref() % rhs
                }
            },
            NodeTypes::SumNode(s) => {
                if self == rhs {
                    return NumNode::init(&0);
                }
                let mut result = vec![];
                s.nodes.iter().for_each(|n| match n.clone().deref() {
                    NodeTypes::NumNode(_) => {
                        result.push(n.clone().deref() % rhs);
                    }
                    NodeTypes::MulNode(_) => {
                        result.push(n.clone().deref() % rhs);
                    }
                    _ => {
                        result.push(n.clone());
                    }
                });
                NodeTypes::sum(&result).deref() % rhs
            }

            NodeTypes::ModNode(n) => match &n.b {
                BTypes::Int(i) => {
                    if i % rhs == 0 {
                        n.a.clone().deref() % rhs
                    } else {
                        self % rhs
                    }
                }
                _ => self % rhs,
            },
            _ => {
                assert!(rhs > &0);

                if rhs == &1 {
                    return NumNode::init(&0);
                }
                if self.min().unwrap() >= 0 && self.max().unwrap() < rhs.clone() {
                    return self - &(rhs.clone() * ((self.min().unwrap() / rhs)));
                }
                if self.min().unwrap() < 0 {
                    return (self - &(((self.min().unwrap() / rhs)) * rhs)).deref() % rhs;
                }

                create_node(NodeTypes::new_mod(self.ptr(), BTypes::Int(rhs.clone())))
            }
        }
    }
}


impl Rem<&isize> for &BTypes{
    type Output = BTypes;
    fn rem(self, rhs: &isize) -> Self::Output {
        match self{
            BTypes::Int(i) => BTypes::Int(i % rhs),
            BTypes::Node(n) => BTypes::Node(n.clone().deref() % rhs)
        }
    }
}
impl Rem<&NodeTypes> for &isize {
    type Output = N;

    fn rem(self, rhs: &NodeTypes) -> Self::Output {
        NumNode::init(&self).deref() % rhs
    }
}

impl Hash for NodeTypes {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.key().hash(state);
    }
}

impl<'a> std::iter::Product<&'a BTypes> for BTypes {
    fn product<I: Iterator<Item = &'a BTypes>>(iter: I) -> BTypes {
        iter.fold(Self::Int(1), |acc, x| &acc * x)
    }
}

impl std::iter::Sum<BTypes> for BTypes{
    fn sum<I: Iterator<Item = BTypes>>(iter: I) -> Self {
        iter.fold(Self::Int(0), |acc, s| &acc + &s)
    }
}

impl Neg for &BTypes{
    type Output = BTypes;

    fn neg(self) -> Self::Output {
        match self{
            BTypes::Int(i) => BTypes::Int( -i.clone()),
            BTypes::Node(n) => BTypes::Node(-n.clone().deref())
        }
    }
}

fn create_lt_node(lhs: N, b: BTypes) -> N{
    let mut mut_b = b;
    let mut n_lhs = lhs;
    if let NodeTypes::SumNode(s) = n_lhs.clone().deref(){
        if let BTypes::Int(i) = &mut_b{
            let mut new_sum = vec![];
            s.nodes.iter().for_each(|x|{
                if let NodeTypes::NumNode(n) = x.clone().deref(){
                    mut_b = &mut_b - &BTypes::Int(n.b.clone());
                }else{
                    new_sum.push(x.clone())
                }
            });

            n_lhs = NodeTypes::sum(&new_sum);
            let nodes = {
                if let NodeTypes::SumNode(ss) = n_lhs.clone().deref(){
                    ss.nodes.clone()
                }else{
                    vec![n_lhs.clone()]
                }
            };

            assert!(nodes.iter().all(|node|{
                match node.clone().deref(){
                    NodeTypes::MulNode(m) => true,
                    NodeTypes::AndNode(a) | NodeTypes::SumNode(a) => {
                        false
                    },
                    NodeTypes::DivNode(n)| NodeTypes::LtNode(n) | NodeTypes::ModNode(n) => {
                        if let BTypes::Int(_) = n.b{
                            true
                        }else{
                            false
                        }
                    },
                    NodeTypes::Variable(v) => {
                        false
                    }
                    NodeTypes::NumNode(n) => {
                        true
                    }
                }
            }), "not supported");

            let (muls, others) = partition(nodes, |x|{
                match x.clone().deref(){
                    NodeTypes::MulNode(m) => m.b > BTypes::Int(0) && &x.max().unwrap() >= &mut_b,
                    _ => false
                }
            });

            if !muls.is_empty(){
                let mut mul_gcd = mut_b.clone();
                muls.iter().for_each(|x| {
                    match x.clone().deref(){
                        NodeTypes::DivNode(n) | NodeTypes::LtNode(n) | NodeTypes::ModNode(n) | NodeTypes::MulNode(n) => {
                            match &n.b{
                                BTypes::Int(i) =>{
                                    if let BTypes::Int(ii) = &mul_gcd{
                                        mul_gcd = BTypes::Int(gcd(ii.clone() as isize, i.clone() as isize) as isize)
                                    }else{
                                        panic!()
                                    }
                                    
                                }
                                _ => panic!()
                            }
                        }
                        NodeTypes::NumNode(n) => {
                            if let BTypes::Int(ii) = &mul_gcd{
                                    mul_gcd = BTypes::Int(gcd(ii.clone() as isize, n.b.clone() as isize) as isize)
                                }else{
                                    panic!()
                            }
                        }
                        _ => panic!()
                    }
                });

                let all_others = NodeTypes::sum(&others);
                if all_others.clone().min().unwrap() >= 0 && &all_others.clone().max().unwrap() < &mul_gcd{
                    n_lhs = NodeTypes::sum(&muls.iter().map(|mul| mul.clone().floordiv(&mul_gcd, true)).collect_vec());
                    mut_b = mut_b.floordiv(&mul_gcd, true);
                }
            }
        }
        if let NodeTypes::SumNode(_) = n_lhs.clone().deref(){
            return create_node(NodeTypes::new_lt(n_lhs, mut_b));
        } else{
            return create_lt_node(n_lhs, mut_b);
        }
    }
    if let NodeTypes::MulNode(mul_n) = n_lhs.clone().deref(){
        if let BTypes::Node(n) = &mut_b{
            if let BTypes::Node(nn) = &mul_n.b{
                if &mul_n.b == &BTypes::Int(-1){
                    return create_node(NodeTypes::new_lt(n_lhs, mut_b));
                }
                return create_node(NodeTypes::new_lt(n_lhs, mut_b));
            }
            return create_node(NodeTypes::new_lt(n_lhs, mut_b));
        }
        let sgn = {
            if mul_n.b > BTypes::Int(0){
                0
            }else{
                -1
            }
        };
        let n_abs = {
            match &mul_n.b{
                BTypes::Int(i) => BTypes::Int(i.abs()),
                BTypes::Node(_) => panic!()
            }
        };
        return create_node(NodeTypes::new_lt(mul_n.a.clone().deref() * &sgn, (&(&mut_b + &n_abs) - &BTypes::Int(1)).floordiv(&mul_n.b, true)));
    }
    return create_node(NodeTypes::new_lt(n_lhs, mut_b));
}