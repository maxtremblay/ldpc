mod flip;
pub use flip::FlipDecoder;

mod css;
pub use css::CssDecoder;

mod belief_propagation;
pub use belief_propagation::BpDecoder;

use sparse_bin_mat::{SparseBinSlice, SparseBinVec};

pub trait LinearDecoder {
    fn decode(&self, message: SparseBinSlice) -> SparseBinVec;
}

pub trait SyndromeDecoder<Syndrome, Correction> {
    fn correction_for(&self, syndrome: Syndrome) -> Correction;
}

pub trait ClassicalSyndromeDecoder<'a>: SyndromeDecoder<SparseBinSlice<'a>, SparseBinVec> {}

