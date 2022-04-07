use sparse_bin_mat::{SparseBinMat, SparseBinSlice};

use super::ErasureDecoder;
use crate::codes::CssCode;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct CssErasureDecoder<'c> {
    code: &'c CssCode,
}

impl<'c> CssErasureDecoder<'c> {
    pub fn new(code: &'c CssCode) -> Self {
        Self { code }
    }
}

impl<'c> ErasureDecoder for CssErasureDecoder<'c> {
    fn is_recoverable(&self, erasure: SparseBinSlice) -> bool {
        recover(
            self.code.x_stabs_binary(),
            self.code.x_logicals_binary(),
            erasure.clone(),
        ) && recover(
            self.code.z_stabs_binary(),
            self.code.z_logicals_binary(),
            erasure,
        )
    }
}

fn recover(stabs: &SparseBinMat, logicals: &SparseBinMat, erasure: SparseBinSlice) -> bool {
    let stabs = stabs.keep_only_columns(erasure.as_slice()).unwrap();
    let logicals = logicals.keep_only_columns(erasure.as_slice()).unwrap();
    stabs.solve(&logicals).is_some()
}

#[cfg(test)]
mod test {
    use sparse_bin_mat::SparseBinVec;

    use super::*;

    #[test]
    fn erasure_failures_in_shor_code() {
        let code = CssCode::shor_code();
        let decoder = CssErasureDecoder::new(&code);

        let erasure = SparseBinVec::new(9, vec![0, 4, 8]);
        assert!(!decoder.is_recoverable(erasure.as_view()));

        let erasure = SparseBinVec::new(9, vec![0, 1, 2]);
        assert!(!decoder.is_recoverable(erasure.as_view()));
    }

    #[test]
    fn erasure_successes_in_shor_code() {
        let code = CssCode::shor_code();
        let decoder = CssErasureDecoder::new(&code);

        let erasure = SparseBinVec::new(9, vec![0, 1, 3, 4]);
        assert!(decoder.is_recoverable(erasure.as_view()));

        let erasure = SparseBinVec::new(9, vec![0, 6, 7]);
        assert!(decoder.is_recoverable(erasure.as_view()));
    }
}
