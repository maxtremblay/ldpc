use super::{LinearCode, SparseBinMat};
use bigs::{error::InvalidParameters, graph::Graph, Sampler};
use itertools::Itertools;
use rand::Rng;
use std::error::Error;
use std::fmt;

/// A random regular ldpc code sampler.
///
/// See [`LinearCode::random_regular_code`](LinearCode::random_regular_code).
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy, Hash)]
pub struct RandomRegularCode {
    num_bits: usize,
    num_checks: usize,
    bit_degree: usize,
    check_degree: usize,
}

impl RandomRegularCode {
    /// Fixes the length of the code.
    ///
    /// Default is 0.
    pub fn num_bits(&mut self, num_bits: usize) -> &mut Self {
        self.num_bits = num_bits;
        self
    }

    /// Fixes the number of checks of the code.
    ///
    /// Default is 0.
    pub fn num_checks(&mut self, num_checks: usize) -> &mut Self {
        self.num_checks = num_checks;
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

    /// Samples a random code with the given random number generator
    /// or returns an error if the `n * b != m * c` where
    /// `n` is the number of bits, `b` the bit's degree, `m` the number of checks
    /// and `c` the check's degree.
    pub fn sample_with<R: Rng>(&self, rng: &mut R) -> Result<LinearCode, SamplingError> {
        Sampler::builder()
            .number_of_variables(self.num_bits)
            .number_of_constraints(self.num_checks)
            .variable_degree(self.bit_degree)
            .constraint_degree(self.check_degree)
            .build()
            .map(|sampler| convert_graph_into_code(sampler.sample_with(rng)))
            .map_err(SamplingError::from_error)
    }
}

fn convert_graph_into_code(graph: Graph) -> LinearCode {
    let checks = graph
        .constraints()
        .sorted_by_key(|check| check.label())
        .map(|check| check.neighbors().iter().sorted().cloned().collect())
        .collect();
    let parity_check_matrix = SparseBinMat::new(graph.number_of_variables(), checks);
    LinearCode::from_parity_check_matrix(parity_check_matrix)
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct SamplingError {
    num_bits: usize,
    num_checks: usize,
    bit_degree: usize,
    check_degree: usize,
}

impl SamplingError {
    fn from_error(error: InvalidParameters) -> Self {
        Self {
            num_bits: error.number_of_variables,
            num_checks: error.number_of_constraints,
            bit_degree: error.variable_degree,
            check_degree: error.constraint_degree,
        }
    }
}

impl fmt::Display for SamplingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "can't generate a regular code with {} bits of degree {} and {} checks of degree {}",
            self.num_bits, self.bit_degree, self.num_checks, self.check_degree
        )
    }
}

impl Error for SamplingError {}
