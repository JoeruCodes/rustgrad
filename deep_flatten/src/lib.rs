use std::iter::Flatten;
pub trait DeepFlattenIteratorOf<Depth,T> {
    type DeepFlatten: Iterator<Item = T>;
    fn deep_flatten(this: Self) -> Self::DeepFlatten;
}

// base condition
impl<I: Iterator> DeepFlattenIteratorOf<(),I::Item> for I {
    type DeepFlatten = Self;
    fn deep_flatten(this: Self) -> Self::DeepFlatten {
        this
    }
}

// other recurse conditions
impl<Depth, I: Iterator, T> DeepFlattenIteratorOf<(Depth,),T> for I
where
    Flatten<I>: DeepFlattenIteratorOf<Depth,T>,
    I: Iterator,
    <I as Iterator>::Item: IntoIterator, 
{
    type DeepFlatten = <Flatten<I> as DeepFlattenIteratorOf<Depth,T>>::DeepFlatten;
    fn deep_flatten(this: Self) -> Self::DeepFlatten {
        DeepFlattenIteratorOf::deep_flatten(this.flatten())
    }
}

// wrapper type to help out type inference
pub struct DeepFlatten<Depth, I, T>
where
    I: DeepFlattenIteratorOf<Depth, T>,
{
    inner: I::DeepFlatten,
}

pub trait DeepFlattenExt: Iterator + Sized {
    fn deep_flatten<Depth, T>(self) -> DeepFlatten<Depth, Self, T>
    where
        Self: DeepFlattenIteratorOf<Depth, T>,
    {
        DeepFlatten {
            inner: DeepFlattenIteratorOf::deep_flatten(self)
        }
    }
}
impl<I: Iterator> DeepFlattenExt for I {}
impl<Depth, I, T> Iterator for DeepFlatten<Depth, I, T>
where
    I: DeepFlattenIteratorOf<Depth, T>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}