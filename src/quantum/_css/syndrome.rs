use sparse_bin_mat::SparseBinVec;

/// The binary representation of a syndrome for a CSS code.
///
/// The X part correponds to the syndrome measured by the X stabilizers
/// and the Z part to the syndrome measured by the Z stabilizers.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CssSyndrome {
    pub x: SparseBinVec,
    pub z: SparseBinVec,
}

impl CssSyndrome {
    /// Checks if the syndrome is the zero syndrome for both X and Z.
    pub fn is_trivial(&self) -> bool {
        self.x.is_zero() && self.z.is_zero()
    }
}
