use crate::linear_code::Edge;
use crate::{LinearCode, Probability};
use sparse_bin_mat::{SparseBinVec, SparseBinVecBase};

pub mod state;
pub use state::BeliefPropagationState;

mod messages;

#[derive(Debug)]
pub struct BeliefPropagationBase {
    code: LinearCode,
    initial_likelyhoods: Vec<f64>,
    empty_sparse_vector: SparseBinVec,
}

impl BeliefPropagationBase {
    pub fn with_code_and_error_probability(code: LinearCode, probability: Probability) -> Self {
        Self {
            code,
            initial_likelyhoods: Self::initialize_likelyhoods(
                code.block_size(),
                probability.value(),
            ),
            empty_sparse_vector: SparseBinVec::empty(),
        }
    }

    fn initialize_likelyhoods(length: usize, probability: f64) -> Vec<f64> {
        let likelyhood = ((1.0 - probability) / probability).ln();
        vec![likelyhood; length]
    }

    pub fn initialize_decoding(&self) -> BeliefPropagationState {
        BeliefPropagationState::from(self)
    }

    pub fn initial_likelyhood_of(&self, bit: usize) -> Option<f64> {
        self.initial_likelyhoods.get(bit).cloned()
    }

    pub fn block_size(&self) -> usize {
        self.code.block_size()
    }

    pub fn bits(&self) -> impl Iterator<Item = usize> {
        0..self.block_size()
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.code.edges()
    }

    pub fn edges_connected_to_bit(&self, bit: usize) -> Option<impl Iterator<Item = Edge> + '_> {
        self.code.checks_adjacent_to_bit(bit).map(move |checks| {
            checks
                .non_trivial_positions()
                .map(move |check| Edge { bit, check })
        })
    }

    pub fn edges_connected_to_z_stabilizer(&self, check: usize) -> impl Iterator<Item = Edge> + '_ {
        self.code
            .check(check)
            .unwrap_or(&self.empty_sparse_vector)
            .non_trivial_positions()
            .map(move |bit| Edge { bit, check })
    }

    pub fn syndrome_of(&self, operator: &SparseBinaryVector) -> SparseBinaryVector {
        self.code.syndrome_of(operator)
    }
}
