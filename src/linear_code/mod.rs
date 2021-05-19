use crate::noise_model::NoiseModel;
use itertools::Itertools;
use rand::Rng;
use serde::{Deserialize, Serialize};
use sparse_bin_mat::{SparseBinMat, SparseBinSlice, SparseBinVec, SparseBinVecBase};

mod edges;
pub use edges::{Edge, Edges};

mod random;
pub use self::random::RandomRegularCode;

/// An implementation of linear codes optimized for LDPC codes.
///
/// A code can be define from either a parity check matrix `H`
/// or a generator matrix `G`.
/// These matrices have the property that `H G^T = 0`.
///
/// # Example
///
/// This is example shows 2 way to define the Hamming code.
///
/// ```
/// # use ldpc::{LinearCode, SparseBinMat};
/// let parity_check_matrix = SparseBinMat::new(
///     7,
///     vec![vec![0, 1, 2, 4], vec![0, 1, 3, 5], vec![0, 2, 3, 6]]
/// );
/// let generator_matrix = SparseBinMat::new(
///     7,
///     vec![vec![0, 4, 5, 6], vec![1, 4, 5], vec![2, 4, 6], vec![3, 5, 6]]
/// );
///
/// let code_from_parity = LinearCode::from_parity_check_matrix(parity_check_matrix);
/// let code_from_generator = LinearCode::from_generator_matrix(generator_matrix);
///
/// assert!(code_from_parity.has_same_codespace_as(&code_from_generator));
/// ```
///
/// # Comparison
///
/// Use the `==` if you want to know if 2 codes
/// have exactly the same parity check matrix and
/// generator matrix.
/// However, since there is freedom in the choice of
/// parity check matrix and generator matrix for the same code,
/// use [`has_the_same_codespace_as`](LinearCode::has_the_same_codespace_as) method
/// if you want to know if 2 codes define the same codespace even
/// if they may have different parity check matrix or generator matrix.
#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct LinearCode {
    parity_check_matrix: SparseBinMat,
    generator_matrix: SparseBinMat,
    bit_adjacencies: SparseBinMat,
}

impl LinearCode {
    /// Creates a new linear code from the given parity check matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::{LinearCode, SparseBinMat};
    /// // 3 bits repetition code.
    /// let matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2]]);
    /// let code = LinearCode::from_parity_check_matrix(matrix);
    ///
    /// assert_eq!(code.block_size(), 3);
    /// assert_eq!(code.dimension(), 1);
    /// assert_eq!(code.minimal_distance(), Some(3));
    /// ```
    pub fn from_parity_check_matrix(parity_check_matrix: SparseBinMat) -> Self {
        let generator_matrix = parity_check_matrix.nullspace();
        let bit_adjacencies = parity_check_matrix.transposed();
        Self {
            parity_check_matrix,
            generator_matrix,
            bit_adjacencies,
        }
    }

    /// Creates a new linear code from the given generator matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::{LinearCode, SparseBinMat};
    /// // 3 bits repetition code.
    /// let matrix = SparseBinMat::new(3, vec![vec![0, 1, 2]]);
    /// let code = LinearCode::from_generator_matrix(matrix);
    ///
    /// assert_eq!(code.block_size(), 3);
    /// assert_eq!(code.dimension(), 1);
    /// assert_eq!(code.minimal_distance(), Some(3));
    /// ```
    pub fn from_generator_matrix(generator_matrix: SparseBinMat) -> Self {
        let parity_check_matrix = generator_matrix.nullspace();
        let bit_adjacencies = parity_check_matrix.transposed();
        Self {
            parity_check_matrix,
            generator_matrix,
            bit_adjacencies,
        }
    }

    /// Returns a repetition code with the given block size.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::{LinearCode, SparseBinMat};
    /// let matrix = SparseBinMat::new(3, vec![vec![0, 1], vec![1, 2]]);
    /// let code = LinearCode::from_parity_check_matrix(matrix);
    ///
    /// assert!(code.has_same_codespace_as(&LinearCode::repetition_code(3)));
    /// ```
    pub fn repetition_code(block_size: usize) -> Self {
        let generator_matrix = SparseBinMat::new(block_size, vec![(0..block_size).collect()]);
        Self::from_generator_matrix(generator_matrix)
    }

    /// Returns the Hamming code.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::{LinearCode, SparseBinMat};
    /// let matrix = SparseBinMat::new(
    ///     7,
    ///     vec![vec![3, 4, 5, 6], vec![1, 2, 5, 6], vec![0, 2, 4, 6]],
    /// );
    /// let code = LinearCode::from_parity_check_matrix(matrix);
    ///
    /// assert!(code.has_same_codespace_as(&LinearCode::hamming_code()));
    /// ```
    pub fn hamming_code() -> Self {
        let parity_check_matrix = SparseBinMat::new(
            7,
            vec![vec![3, 4, 5, 6], vec![1, 2, 5, 6], vec![0, 2, 4, 6]],
        );
        Self::from_parity_check_matrix(parity_check_matrix)
    }

    /// Returns a code of length 0 encoding 0 bits and without checks.
    ///
    /// This is mostly useful as a place holder.
    pub fn empty() -> Self {
        let matrix = SparseBinMat::empty();
        Self::from_parity_check_matrix(matrix)
    }

    /// Returns a builder for random LDPC codes with
    /// regular parity check matrix.
    ///
    /// The [`sample_with`](RandomRegularCode::sample_with) method returns
    /// an error if the block size times the bit's degree is not equal
    /// to the number of checks times the bit check's degree.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::LinearCode;
    /// use rand::thread_rng;
    ///
    /// let code = LinearCode::random_regular_code()
    ///     .block_size(20)
    ///     .number_of_checks(15)
    ///     .bit_degree(3)
    ///     .check_degree(4)
    ///     .sample_with(&mut thread_rng())
    ///     .unwrap(); // 20 * 3 == 15 * 4
    ///
    /// assert_eq!(code.block_size(), 20);
    /// assert_eq!(code.number_of_checks(), 15);
    /// assert_eq!(code.parity_check_matrix().number_of_ones(), 60);
    /// ```
    pub fn random_regular_code() -> RandomRegularCode {
        RandomRegularCode::default()
    }

    /// Returns the parity check matrix of the code.
    pub fn parity_check_matrix(&self) -> &SparseBinMat {
        &self.parity_check_matrix
    }

    /// Returns the check at the given index or
    /// None if the index is out of bound.
    ///
    /// That is, this returns the row of the parity check matrix
    /// with the given index.
    pub fn check(&self, index: usize) -> Option<SparseBinSlice> {
        self.parity_check_matrix.row(index)
    }

    /// Returns the generator matrix of the code.
    pub fn generator_matrix(&self) -> &SparseBinMat {
        &self.generator_matrix
    }

    /// Returns the generator at the given index or
    /// None if the index is out of bound.
    ///
    /// That is, this returns the row of the generator matrix
    /// with the given index.
    pub fn generator(&self, index: usize) -> Option<SparseBinSlice> {
        self.generator_matrix.row(index)
    }

    /// Returns a matrix where the value in row i
    /// correspond to the check connected to bit i.
    pub fn bit_adjacencies(&self) -> &SparseBinMat {
        &self.bit_adjacencies
    }

    /// Returns the checks adjacents to the given bit or
    /// None if the bit is out of bound.
    pub fn checks_adjacent_to_bit(&self, bit: usize) -> Option<SparseBinSlice> {
        self.bit_adjacencies.row(bit)
    }

    /// Checks if two code define the same codespace.
    ///
    /// Two codes have the same codespace if all their codewords are the same.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::{LinearCode, SparseBinMat};
    /// // The Hamming code
    /// let parity_check_matrix = SparseBinMat::new(
    ///     7,
    ///     vec![vec![0, 1, 2, 4], vec![0, 1, 3, 5], vec![0, 2, 3, 6]]
    /// );
    /// let hamming_code = LinearCode::from_parity_check_matrix(parity_check_matrix);
    ///
    /// // Same but with the add the first check to the other two.
    /// let parity_check_matrix = SparseBinMat::new(
    ///     7,
    ///     vec![vec![0, 1, 2, 4], vec![2, 3, 4, 5], vec![1, 3, 4, 6]]
    /// );
    /// let other_hamming_code = LinearCode::from_parity_check_matrix(parity_check_matrix);
    ///
    /// assert!(hamming_code.has_same_codespace_as(&other_hamming_code));
    /// ```
    pub fn has_same_codespace_as(&self, other: &Self) -> bool {
        self.block_size() == other.block_size()
            && (&self.parity_check_matrix * &other.generator_matrix.transposed()).is_zero()
    }

    /// Returns the number of bits in the code.
    pub fn block_size(&self) -> usize {
        self.parity_check_matrix.number_of_columns()
    }

    /// Returns the number of rows of the parity check matrix
    /// of the code.
    pub fn number_of_checks(&self) -> usize {
        self.parity_check_matrix.number_of_rows()
    }

    /// Returns the number of rows of the generator matrix
    /// of the code.
    pub fn number_of_generators(&self) -> usize {
        self.generator_matrix.number_of_rows()
    }

    /// Returns the number of linearly independent codewords.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::{LinearCode, SparseBinMat};
    /// let parity_check_matrix = SparseBinMat::new(
    ///     7,
    ///     vec![vec![0, 1, 2, 4], vec![0, 1, 3, 5], vec![0, 2, 3, 6]]
    /// );
    /// let hamming_code = LinearCode::from_parity_check_matrix(parity_check_matrix);
    ///
    /// assert_eq!(hamming_code.dimension(), 4);
    /// ```
    pub fn dimension(&self) -> usize {
        self.generator_matrix.rank()
    }

    /// Returns the weight of the smallest non trivial codeword
    /// or None if the code have no codeword.
    ///
    /// # Warning
    ///
    /// The execution time of this method scale exponentially with the
    /// dimension of the code.
    pub fn minimal_distance(&self) -> Option<usize> {
        (1..=self.number_of_generators())
            .flat_map(|n| self.generator_matrix.rows().combinations(n))
            .filter_map(|generators| {
                let weight = generators
                    .into_iter()
                    .fold(SparseBinVec::zeros(self.block_size()), |sum, generator| {
                        &sum + &generator
                    })
                    .weight();
                if weight > 0 {
                    Some(weight)
                } else {
                    None
                }
            })
            .min()
    }

    /// Returns an iterator over all edges of the Tanner graph associated with
    /// the parity check matrix of the code.
    ///
    /// That is, this returns an iterator of over the coordinates (i, j) such
    /// that H_ij = 1 with H the parity check matrix.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::{LinearCode, SparseBinMat, SparseBinVec, Edge};
    /// let parity_check_matrix = SparseBinMat::new(
    ///     4,
    ///     vec![vec![0, 1], vec![0, 3], vec![1, 2]]
    /// );
    /// let code = LinearCode::from_parity_check_matrix(parity_check_matrix);
    /// let mut edges = code.edges();
    ///
    /// assert_eq!(edges.next(), Some(Edge { bit: 0, check: 0}));
    /// assert_eq!(edges.next(), Some(Edge { bit: 1, check: 0}));
    /// assert_eq!(edges.next(), Some(Edge { bit: 0, check: 1}));
    /// assert_eq!(edges.next(), Some(Edge { bit: 3, check: 1}));
    /// assert_eq!(edges.next(), Some(Edge { bit: 1, check: 2}));
    /// assert_eq!(edges.next(), Some(Edge { bit: 2, check: 2}));
    /// assert_eq!(edges.next(), None);
    /// ```
    pub fn edges(&self) -> Edges {
        Edges::new(self)
    }

    /// Returns the product of the parity check matrix with the given message
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::{LinearCode, SparseBinMat, SparseBinVec};
    /// let parity_check_matrix = SparseBinMat::new(
    ///     7,
    ///     vec![vec![0, 1, 2, 4], vec![0, 1, 3, 5], vec![0, 2, 3, 6]]
    /// );
    /// let hamming_code = LinearCode::from_parity_check_matrix(parity_check_matrix);
    ///
    /// let message = SparseBinVec::new(7, vec![0, 2, 4]);
    /// let syndrome = SparseBinVec::new(3, vec![0, 1]);
    ///
    /// assert_eq!(hamming_code.syndrome_of(&message.as_view()), syndrome);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if the message have a different length then code block size.
    pub fn syndrome_of<T>(&self, message: &SparseBinVecBase<T>) -> SparseBinVec
    where
        T: std::ops::Deref<Target = [usize]>,
    {
        if message.len() != self.block_size() {
            panic!(
                "message of length {} is invalid for code with block size {}",
                message.len(),
                self.block_size()
            );
        }
        &self.parity_check_matrix * message
    }

    /// Checks if a message has zero syndrome.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::{LinearCode, SparseBinMat, SparseBinVec};
    /// let parity_check_matrix = SparseBinMat::new(
    ///     7,
    ///     vec![vec![0, 1, 2, 4], vec![0, 1, 3, 5], vec![0, 2, 3, 6]]
    /// );
    /// let hamming_code = LinearCode::from_parity_check_matrix(parity_check_matrix);
    ///
    /// let error = SparseBinVec::new(7, vec![0, 2, 4]);
    /// let codeword = SparseBinVec::new(7, vec![2, 3, 4, 5]);
    ///
    /// assert_eq!(hamming_code.has_codeword(&error), false);
    /// assert_eq!(hamming_code.has_codeword(&codeword), true);
    /// ```
    ///
    /// # Panic
    ///
    /// Panics if the message have a different length then code block size.
    pub fn has_codeword<T>(&self, operator: &SparseBinVecBase<T>) -> bool
    where
        T: std::ops::Deref<Target = [usize]>,
    {
        self.syndrome_of(operator).is_zero()
    }

    /// Generates a random error with the given noise model.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::{SparseBinMat, LinearCode};
    /// use ldpc::noise_model::{BinarySymmetricChannel, Probability};
    /// use rand::thread_rng;
    ///
    /// let parity_check_matrix = SparseBinMat::new(
    ///     7,
    ///     vec![vec![0, 1, 2, 4], vec![0, 1, 3, 5], vec![0, 2, 3, 6]]
    /// );
    /// let code = LinearCode::from_parity_check_matrix(parity_check_matrix);
    ///
    /// let noise = BinarySymmetricChannel::with_probability(Probability::new(0.25));
    /// let error = code.random_error(&noise, &mut thread_rng());
    ///
    /// assert_eq!(error.len(), 7);
    /// ```
    pub fn random_error<N, R>(&self, noise_model: &N, rng: &mut R) -> SparseBinVec
    where
        N: NoiseModel<Error = SparseBinVec>,
        R: Rng,
    {
        noise_model.sample_error_of_length(self.block_size(), rng)
    }

    /// Returns the code as a json string.
    pub fn as_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(self)
    }
}
