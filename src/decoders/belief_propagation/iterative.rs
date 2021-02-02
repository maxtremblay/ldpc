use super::BeliefPropagationBase;
use crate::LinearCode;
use sparse_bin_mat::{SparseBinVec, SparseBinVecBase};

#[derive(Debug)]
pub struct IterativeBeliefPropagation {
    base: BeliefPropagationBase,
    number_of_iterations: usize,
}

impl IterativeBeliefPropagation {
    pub fn with_code_error_probabilities_and_max_iterations(
        code: LinearCode,
        probabilities: &[f64],
        number_of_iterations: usize,
    ) -> Self {
        Self {
            base: BeliefPropagationBase::with_code_and_error_probabilities(code, probabilities),
            number_of_iterations,
        }
    }

    fn compute_correction_for<T>(&self, syndrome: &SparseBinVecBase<T>) -> SparseBinVec
    where
        T: std::ops::Deref<Target = [usize]>,
    {
        let mut state = self.base.initialize_decoding();
        for _ in 0..self.number_of_iterations {
            state = state.run_next_iteration(syndrome);
        }
        state.compute_correction()
    }
}
