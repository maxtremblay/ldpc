use std::ops::Deref;

use pauli::{Pauli, PauliOperator};
use serde::{Serialize, Deserialize};
use sparse_bin_mat::{SparseBinSlice, SparseBinVec, SparseBinVecBase};

#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct Css<X, Z = X> {
    pub x: X,
    pub z: Z,
}

impl<T> Css<T> {
    pub fn map<'a, F, S>(&'a self, func: F) -> Css<S>
    where
        F: Fn(&'a T) -> S,
    {
        Css {
            x: func(&self.x),
            z: func(&self.z),
        }
    }

    pub fn map_with_pauli<'a, F, S>(&'a self, func: F) -> Css<S>
    where
        F: Fn(&'a T, Pauli) -> S,
    {
        Css {
            x: func(&self.x, Pauli::X),
            z: func(&self.z, Pauli::Z),
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
    pub fn map_each<'a, F, G, XX, ZZ>(&'a self, funcs: Css<F, G>) -> Css<XX, ZZ>
    where
        F: Fn(&'a X) -> XX,
        G: Fn(&'a Z) -> ZZ,
    {
        Css {
            x: (funcs.x)(&self.x),
            z: (funcs.z)(&self.z),
        }
    }

    pub fn pair<XX, ZZ>(self, other: Css<XX, ZZ>) -> Css<(X, XX), (Z, ZZ)> {
        Css {
            x: (self.x, other.x),
            z: (self.z, other.z),
        }
    }

    pub fn combine_with<F, T>(self, func: F) -> T
    where
        F: Fn(X, Z) -> T,
    {
        func(self.x, self.z)
    }

    pub fn swap_xz(self) -> Css<Z, X> {
        Css {
            x: self.z,
            z: self.x,
        }
    }

    pub fn as_ref(&self) -> Css<&X, &Z> {
        Css {
            x: &self.x,
            z: &self.z,
        }
    }

    pub fn as_mut(&mut self) -> Css<&mut X, &mut Z> {
        Css {
            x: &mut self.x,
            z: &mut self.z,
        }
    }
}



pub type CssOperator = Css<SparseBinVec>;

impl<'a> From<&'a PauliOperator> for CssOperator {
    fn from(operator: &'a PauliOperator) -> Self {
        Self {
            x: SparseBinVec::new(operator.len(), operator.x_part().into_raw_positions()),
            z: SparseBinVec::new(operator.len(), operator.z_part().into_raw_positions()),
        }
    }
}

pub type CssSyndrome<T = Vec<usize>> = Css<SparseBinVecBase<T>>;
pub type CssSyndromeView<'a> = Css<SparseBinSlice<'a>>;

impl<T> CssSyndrome<T>
where
    T: Deref<Target = [usize]>,
{
    pub fn is_trivial(&self) -> bool {
        self.both(|syndrome| syndrome.is_zero())
    }
}
