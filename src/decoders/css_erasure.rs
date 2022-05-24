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

    pub fn error_basis(&self, erasure: SparseBinSlice) -> SparseBinMat {
        let rows = erasure
            .non_trivial_positions()
            .map(|pos| vec![pos])
            .collect();
        SparseBinMat::new(self.code.len(), rows)
    }

    fn num_bad_x_errors(&self, errors: &SparseBinMat) -> usize {
        Self::num_bad_errors(
            errors,
            self.code.z_stabs_binary(),
            self.code.z_logicals_binary(),
        )
    }

    fn num_bad_z_errors(&self, errors: &SparseBinMat) -> usize {
        Self::num_bad_errors(
            errors,
            self.code.x_stabs_binary(),
            self.code.x_logicals_binary(),
        )
    }

    fn num_bad_errors(
        errors: &SparseBinMat,
        stabs: &SparseBinMat,
        logicals: &SparseBinMat,
    ) -> usize {
        let syndrome_rows = errors
            .rows()
            .map(|error| (stabs * &error).to_positions_vec())
            .collect();
        let syndrome_matrix = SparseBinMat::new(stabs.number_of_columns(), syndrome_rows);
        let logical_rows = errors
            .rows()
            .map(|error| (logicals * &error).to_positions_vec())
            .collect();
        let logical_matrix = SparseBinMat::new(logicals.number_of_columns(), logical_rows);
        let total_matrix = syndrome_matrix.horizontal_concat_with(&logical_matrix);
        total_matrix.rank() - syndrome_matrix.rank()
    }
}

impl<'c> ErasureDecoder for CssErasureDecoder<'c> {
    fn recovery_probability(&self, erasure: SparseBinSlice) -> f64 {
        let errors = self.error_basis(erasure);
        1.0 / 2.0_f64.powi((self.num_bad_x_errors(&errors) + self.num_bad_z_errors(&errors)) as i32)
    }
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
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 0);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 1);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 0.5);

        let erasure = SparseBinVec::new(9, vec![0, 1, 2]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 1);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 0);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 0.5);
    }

    #[test]
    fn erasure_successes_in_shor_code() {
        let code = CssCode::shor_code();
        let decoder = CssErasureDecoder::new(&code);

        let erasure = SparseBinVec::new(9, vec![0, 1, 3, 4]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 0);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 0);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 1.0);

        let erasure = SparseBinVec::new(9, vec![0, 6, 7]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 0);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 0);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 1.0);
    }

    #[test]
    fn empty_erasure_in_shor_code() {
        let code = CssCode::shor_code();
        let decoder = CssErasureDecoder::new(&code);

        let erasure = SparseBinVec::new(9, vec![]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 0);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 0);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 1.0);
    }

    #[test]
    fn erasure_failures_in_steane_code() {
        let code = CssCode::steane_code();
        let decoder = CssErasureDecoder::new(&code);

        let erasure = SparseBinVec::new(7, vec![0, 1, 2]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 1);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 1);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 0.25);

        let erasure = SparseBinVec::new(7, vec![0, 3, 4, 5]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 1);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 1);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 0.25);
    }

    #[test]
    fn erasure_failures_in_toric_code() {
        let code = CssCode::toric_code(3);
        let decoder = CssErasureDecoder::new(&code);

        let erasure = SparseBinVec::new(18, vec![0, 1, 2]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 0);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 1);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 0.5);

        let erasure = SparseBinVec::new(18, vec![9, 10, 11]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 1);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 0);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 0.5);

        let erasure = SparseBinVec::new(18, vec![1, 4, 6, 7, 8, 9, 12, 15]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 1);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 2);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 0.125);

        let erasure = SparseBinVec::new(18, vec![1, 4, 6, 7, 8, 9, 10, 11, 12, 15]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 2);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 2);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 0.0625);
    }

    #[test]
    fn erasure_successes_in_toric_code() {
        let code = CssCode::toric_code(3);
        println!(
            "x: {} \n {}",
            code.x_stabs_binary(),
            code.x_logicals_binary()
        );
        println!(
            "z: {} \n {}",
            code.z_stabs_binary(),
            code.z_logicals_binary()
        );
        let decoder = CssErasureDecoder::new(&code);

        let erasure = SparseBinVec::new(18, vec![0, 1]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 0);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 0);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 1.0);

        let erasure = SparseBinVec::new(18, vec![0, 1, 3, 4, 9, 10, 12, 13]);
        let error_basis = decoder.error_basis(erasure.as_view());
        assert_eq!(decoder.num_bad_x_errors(&error_basis), 0);
        assert_eq!(decoder.num_bad_z_errors(&error_basis), 0);
        assert_eq!(decoder.recovery_probability(erasure.as_view()), 1.0);
    }
}
