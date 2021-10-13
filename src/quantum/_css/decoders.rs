use std::ops::Deref;

use sparse_bin_mat::SparseBinVecBase;

pub struct Css<X, Z = X> {
    pub x: X,
    pub z: Z,
}

impl<X, Z> Css<X, Z> {
    pub fn new(x: X, z: Z) -> Self {
        Self { x, z }
    }
}

impl<T> Css<T> {
    pub fn map<F, S>(&self, func: F) -> Css<S>
    where
        F: Fn(&T) -> S,
    {
        Css {
            x: func(&self.x),
            z: func(&self.z),
        }
    }

    pub fn both<F>(&self, func: F) -> bool
    where
        F: Fn(&T) -> bool,
    {
        func(&self.x) && func(&self.z)
    }
}

impl<X, Z> Css<X, Z> {
    pub fn map_each<F, G, XX, ZZ>(&self, funcs: Css<F, G>) -> Css<XX, ZZ>
    where
        F: Fn(&X) -> XX,
        G: Fn(&Z) -> ZZ,
    {
        Css {
            x: (funcs.x)(&self.x),
            z: (funcs.z)(&self.z),
        }
    }
}

pub type CssSyndrome<T> = Css<SparseBinVecBase<T>>;

impl<T> CssSyndrome<T>
where
    T: Deref<Target = [usize]>,
{
    pub fn is_trivial(&self) -> bool {
        self.both(|syndrome| syndrome.is_zero())
    }
}
