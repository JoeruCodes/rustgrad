use std::{any::Any, collections::{HashMap, HashSet}, fmt::Debug, hash::Hash, ops::{Add, Mul, Neg, Sub}, path::Display};
pub mod helpers;
pub trait Node<B, Min, Max>: Any
where Max: PartialEq<Min>, Min: PartialEq<isize>
{
    fn get_b<'a>(&'a self) -> &'a B;
    fn get_min<'a>(&'a self) -> &'a Min;
    fn get_max<'a>(&'a self) -> &'a Max;
    fn render(&self, ops: Option<RenderOps<B, Min, Max>>, ctx: String) -> String;
    fn vars(&self) -> HashSet<Variable>;
    fn substitute(&mut self, var_vals: HashMap<Variable, SubstitutableVals>) -> Result<(), anyhow::Error>;
    fn unbind(&mut self) -> Option<isize>;
    fn key(&self) -> String{
        return self.render(None, "DEBUG".to_string());
    }
}

pub trait CastAny
    where Self: Sized + 'static
{
    fn as_any(self: Box<Self>) -> Box<dyn Any>{
        self
    }
}
pub trait NodeEq<B, Min, Max>: Node<B, Min, Max>
where Max: PartialEq<Min> + 'static, Min: PartialEq<isize> + 'static, B: 'static
{
    fn eq_n(lhs: &dyn Node<B, Min, Max>, rhs: &dyn Node<B, Min, Max>) -> bool{
        lhs.key() == rhs.key()
    } 
}
pub trait SumSeq<B, Min, Max>: Node<B, Min, Max>
    where Max: PartialEq<Min>, Min: PartialEq<isize>, dyn Node<B, Min, Max>: Sized + Clone
{
    fn sum(nodes: Vec<Box<dyn Node<B, Min, Max>>>) -> Box<dyn Node<B, Min, Max>>{
        if nodes.is_empty(){
            // return NumNode(0)
        }

        if nodes.len() == 1{
            return nodes[0].clone();
        }

        let mut mul_groups: HashMap<dyn Node<B, Min, Max>, isize> = HashMap::new();
        let mut num_node_sum = 0;

        todo!()
    }
}

impl<B, Min, Max> CastAny for dyn Node<B, Min, Max>
where dyn Node<B, Min, Max>: Sized, Max: 'static, Min: 'static, B: 'static
{

}
impl<B, Min, Max> Debug for dyn Node<B, Min, Max>
    where (B, Min, Max) : 'static, Max: PartialEq<Min>, Min: PartialEq<isize>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.render(None, "REPR".to_string()))
    }
}

impl<B, Min, Max> ToString for dyn Node<B, Min, Max>
where (B, Min, Max) : 'static, Max: PartialEq<Min>, Min: PartialEq<isize>
{
    fn to_string(&self) -> String {
        return format!("<{}>", self.key())
    }
}

impl<B, Min, Max> Hash for dyn Node<B, Min, Max>
where (B, Min, Max) : 'static, Max: PartialEq<Min>, Min: PartialEq<isize>
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state)
    }
}

impl<B, Min, Max> Into<bool> for &dyn Node<B, Min, Max>
where (B, Min, Max) : 'static, Max: PartialEq<Min>, Min: PartialEq<isize>
{
    fn into(self) -> bool {
        !(self.get_max() == self.get_min() && self.get_min() == &0)
    }
}

impl<B, Min, Max> Neg for Box<dyn Node<B, Min, Max>>
where
    Self: Mul<isize, Output = Self> + Sized,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1
    }
}

impl<B, Min, Max> Add for Box<dyn Node<B, Min, Max>>
    where dyn Node<B, Min, Max>: SumSeq<B, Min, Max> + Sized + Clone,
    Max: PartialEq<Min>, Min: PartialEq<isize>,
    Self: SumSeq<B, Min, Max> + Sized + Clone
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self::sum(vec![self, rhs])
    }
}

impl <B, Min, Max> Add<isize> for Box<dyn Node<B, Min, Max>>
    where dyn Node<B, Min, Max>: SumSeq<B, Min, Max> + Sized + Clone,
    Max: PartialEq<Min>, Min: PartialEq<isize>,
    Self: SumSeq<B, Min, Max> + Sized + Clone
{
    type Output = Self;
    fn add(self, rhs: isize) -> Self::Output {
        Self::sum(vec![self, Box::new(NumNode::new(rhs))])
    }
}

impl <B, Min, Max> Add<Box<dyn Node<B, Min, Max>>> for isize
    where dyn Node<B, Min, Max>: SumSeq<B, Min, Max> + Sized + Clone,
    Max: PartialEq<Min>, Min: PartialEq<isize>,
    Box<(dyn Node<B, Min, Max>)>: SumSeq<B, Min, Max>
{
    type Output = Box<(dyn Node<B, Min, Max>)>;
    fn add(self, rhs: Box<dyn Node<B, Min, Max>>) -> Self::Output {
        rhs + self
    }    
}

impl <B, Min, Max> Sub for Box<dyn Node<B, Min, Max>>
where dyn Node<B, Min, Max>: SumSeq<B, Min, Max> + Sized + Clone,
Max: PartialEq<Min>, Min: PartialEq<isize>,
Self: SumSeq<B, Min, Max> + Sized + Clone +  Neg<Output = Self>
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl <B, Min, Max> Sub<isize> for Box<dyn Node<B, Min, Max>>
    where dyn Node<B, Min, Max>: SumSeq<B, Min, Max> + Sized + Clone,
    Max: PartialEq<Min>, Min: PartialEq<isize>,
    Self: SumSeq<B, Min, Max> + Sized + Clone
{
    type Output = Self;
    fn sub(self, rhs: isize) -> Self::Output {
        self + (-rhs)
    }
}

impl <B, Min, Max> Sub<Box<dyn Node<B, Min, Max>>> for isize
    where dyn Node<B, Min, Max>: SumSeq<B, Min, Max> + Sized + Clone,
    Max: PartialEq<Min>, Min: PartialEq<isize>,
    Box<(dyn Node<B, Min, Max>)>: SumSeq<B, Min, Max> + Neg<Output = Box<(dyn Node<B, Min, Max>)>>
{
    type Output = Box<(dyn Node<B, Min, Max>)>;
    fn sub(self, rhs: Box<dyn Node<B, Min, Max>>) -> Self::Output {
        -rhs + self
    }    
}

impl <B, Min, Max> PartialEq for Box<dyn Node<B, Min, Max>>
    where Max: PartialEq<Min>, Min: PartialEq<isize>, Self: NodeEq<B, Min, Max>
{
    fn eq(&self, other: &Box<dyn Node<B, Min, Max>>) -> bool {
        Self::eq_n(self, other)
    }
}


pub struct RenderOps<B, Min, Max> {
    render_map: HashMap<&'static str, fn(&dyn Node<B, Min, Max>, Option<&str>) -> String>,
}
pub struct Variable{}
pub struct NumNode{}
impl<B, Min, Max> Node<B, Min, Max> for NumNode
where Max: PartialEq<Min>, Min: PartialEq<isize>,
{
    fn get_b<'a>(&'a self) -> &'a B {
        todo!()
    }
    fn get_max<'a>(&'a self) -> &'a Max {
        todo!()
    }
    fn get_min<'a>(&'a self) -> &'a Min {
        todo!()
    }
    fn key(&self) -> String {
        todo!()
    }
    fn render(&self, ops: Option<RenderOps<B, Min, Max>>, ctx: String) -> String {
        todo!()
    }
    fn substitute(&mut self, var_vals: HashMap<Variable, SubstitutableVals>) -> Result<(), anyhow::Error> {
        todo!()
    }
    fn unbind(&mut self) -> Option<isize> {
        todo!()
    }
    fn vars(&self) -> HashSet<Variable> {
        todo!()
    }
}
impl NumNode{
    pub fn new(num: isize) -> Self{
        todo!()
    }
}
pub struct SumNode{}
pub enum SubstitutableVals{
    Variable(Variable),
    NumNode(NumNode)
}