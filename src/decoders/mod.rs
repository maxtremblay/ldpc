mod flip;
pub use flip::FlipDecoder;

mod css;
pub use css::CssDecoder;

mod belief_propagation;
pub use belief_propagation::BpDecoder;

mod css_erasure;
pub use css_erasure::CssErasureDecoder;

use sparse_bin_mat::{SparseBinSlice, SparseBinVec};

pub trait LinearDecoder {
    fn decode(&self, message: SparseBinSlice) -> SparseBinVec;
}

pub trait SyndromeDecoder<Syndrome, Correction> {
    fn correction_for(&self, syndrome: Syndrome) -> Correction;
}

pub trait ClassicalSyndromeDecoder<'a>: SyndromeDecoder<SparseBinSlice<'a>, SparseBinVec> {}

pub trait ErasureDecoder {
    fn is_recoverable(&self, erasure: SparseBinSlice) -> bool;
}
