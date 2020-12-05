use super::{LinearCode, SparseBinMat};
use bigs::{graph::Graph, Sampler};
use itertools::Itertools;
use rand::Rng;

/// A random regular ldpc code sampler.
///
/// See [`LinearCode::random_regular_code`](LinearCode::random_regular_code).
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy, Hash)]
pub struct RandomRegularCode {
    block_size: usize,
    number_of_checks: usize,
    bit_degree: usize,
    check_degree: usize,
}

impl RandomRegularCode {
    /// Fixes the block size of the code.
    ///
    /// Default is 0.
    pub fn block_size(&mut self, block_size: usize) -> &mut Self {
        self.block_size = block_size;
        self
    }

    /// Fixes the number of checks of the code.
    ///
    /// Default is 0.
    pub fn number_of_checks(&mut self, number_of_checks: usize) -> &mut Self {
        self.number_of_checks = number_of_checks;
        self
    }

    /// Fixes the number of checks connected to each bit of the code.
    ///
    /// Default is 0.
    pub fn bit_degree(&mut self, bit_degree: usize) -> &mut Self {
        self.bit_degree = bit_degree;
        self
    }

    /// Fixes the number of bits connected to each check of the code.
    ///
    /// Default is 0.
    pub fn check_degree(&mut self, check_degree: usize) -> &mut Self {
        self.check_degree = check_degree;
        self
    }

    /// Samples a random code with the given random number generator.
    pub fn sample_with<R: Rng>(&self, rng: &mut R) -> LinearCode {
        let graph = Sampler::builder()
            .number_of_variables(self.block_size)
            .number_of_constraints(self.number_of_checks)
            .variable_degree(self.bit_degree)
            .constraint_degree(self.check_degree)
            .build()
            .sample_with(rng);
        convert_graph_into_code(graph)
    }
}

fn convert_graph_into_code(graph: Graph) -> LinearCode {
    let checks = graph
        .constraints()
        .sorted_by_key(|check| check.label())
        .map(|check| check.neighbors().iter().cloned().collect())
        .collect();
    let parity_check_matrix = SparseBinMat::new(graph.number_of_variables(), checks);
    LinearCode::from_parity_check_matrix(parity_check_matrix)
}
