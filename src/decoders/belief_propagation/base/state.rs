use super::messages::{BitToCheckMessages, CheckToBitMessages};
use super::BeliefPropagationBase;
use itertools::Itertools;
use sparse_bin_mat::SparseBinVec;

#[derive(Debug, Clone)]
pub struct BeliefPropagationState<'decoder> {
    number_of_iterations: usize,
    bit_to_check_messages: BitToCheckMessages<'decoder>,
    check_to_bit_messages: CheckToBitMessages<'decoder>,
    decoder: &'decoder BeliefPropagationBase,
}

impl<'decoder> From<&'decoder BeliefPropagationBase> for BeliefPropagationState<'decoder> {
    fn from(decoder: &'decoder BeliefPropagationBase) -> Self {
        Self {
            number_of_iterations: 0,
            bit_to_check_messages: BitToCheckMessages::from(decoder),
            check_to_bit_messages: CheckToBitMessages::from(decoder),
            decoder,
        }
    }
}

impl<'decoder> BeliefPropagationState<'decoder> {
    pub fn run_next_iteration(&self, syndrome: &SparseBinVec) -> Self {
        let check_to_bit_messages = self.update_check_to_bit_messages(syndrome);
        let bit_to_check_messages = self.update_bit_to_check_messages(&check_to_bit_messages);
        Self {
            number_of_iterations: self.number_of_iterations + 1,
            bit_to_check_messages,
            check_to_bit_messages,
            decoder: self.decoder,
        }
    }

    fn update_bit_to_check_messages(
        &self,
        check_to_bit_messages: &CheckToBitMessages,
    ) -> BitToCheckMessages<'decoder> {
        self.bit_to_check_messages
            .update_from(check_to_bit_messages)
    }

    fn update_check_to_bit_messages(
        &self,
        syndrome: &SparseBinVec,
    ) -> CheckToBitMessages<'decoder> {
        self.check_to_bit_messages
            .update_from(&self.bit_to_check_messages, syndrome)
    }

    pub fn number_of_iterations(&self) -> usize {
        self.number_of_iterations
    }

    pub fn compute_correction_and_syndrome(&self) -> (SparseBinVec, SparseBinVec) {
        let correction = self.compute_correction();
        let syndrome = self.decoder.syndrome_of(&correction);
        (correction, syndrome)
    }

    pub fn compute_syndrome(&self) -> SparseBinVec {
        self.decoder.syndrome_of(&self.compute_correction())
    }

    pub fn compute_correction(&self) -> SparseBinVec {
        SparseBinVec::new(
            self.decoder.block_size(),
            self.decoder
                .bits()
                .positions(|bit| self.likelyhood_of_bit(bit) <= 0.0)
                .collect(),
        )
    }

    fn likelyhood_of_bit(&self, bit: usize) -> f64 {
        let initial_likelyhood = self.decoder.initial_likelyhood_of(bit).unwrap();
        let sum_of_messages = self
            .check_to_bit_messages
            .sum_at_edges(self.decoder.edges_connected_to_bit(bit));
        initial_likelyhood + sum_of_messages
    }
}
