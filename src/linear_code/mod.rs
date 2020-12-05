use itertools::Itertools;
pub use sparse_bin_mat::{SparseBinMat, SparseBinSlice, SparseBinVec};

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
/// assert!(code_from_parity.has_the_same_codespace_as(&code_from_generator));
/// ```
///
/// # Comparison
///
/// Use the `==` if you want to know if 2 codes
/// have exactly the same parity check matrix and
/// generator matrix.
/// However, since there is freedom in the choice of
/// parity check matrix and generator matrix for the same code,
/// use [`has_the_same_codespace_as`](LinearCode::has_the_same_codespace) method
/// if you want to know if 2 codes define the same codespace even
/// if they may have different parity check matrix or generator matrix.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct LinearCode {
    parity_check_matrix: SparseBinMat,
    generator_matrix: SparseBinMat,
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
    pub fn from_parity_check_matrix(matrix: SparseBinMat) -> Self {
        Self {
            generator_matrix: matrix.nullspace(),
            parity_check_matrix: matrix,
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
    pub fn from_generator_matrix(matrix: SparseBinMat) -> Self {
        Self {
            parity_check_matrix: matrix.nullspace(),
            generator_matrix: matrix,
        }
    }

    /// Returns a builder for random LDPC codes with
    /// regular parity check matrix.
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
    ///     .sample_with(&mut thread_rng());
    ///
    /// assert_eq!(code.block_size(), 20);
    /// assert_eq!(code.number_of_checks(), 15);
    /// assert_eq!(code.bit_degree(), 3);
    /// assert_eq!(code.check_degree(), 4);
    /// ```
    pub fn random_regular_code() -> RandomRegularCode {
        RandomRegularCode::default()
    }

    /// Returns the parity check matrix of the code.
    pub fn parity_check_matrix(&self) -> &SparseBinMat {
        &self.parity_check_matrix
    }

    /// Returns the generator matrix of the code.
    pub fn generator_matrix(&self) -> &SparseBinMat {
        &self.generator_matrix
    }

    /// Checks if two code define the same codespace.
    ///
    /// Two codes have the same codespace if all there codewords are the same.
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
    /// assert!(hamming_code.has_the_same_codespace_as(&other_hamming_code));
    /// ```
    pub fn has_the_same_codespace_as(&self, other: &Self) -> bool {
        (&self.parity_check_matrix * &other.generator_matrix.transposed()).is_zero()
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

    /// Returns the product of the parity check matrix with the given error.
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
    /// let syndrome = SparseBinVec::new(3, vec![0, 1]);
    ///
    /// assert_eq!(hamming_code.syndrome_of(&error.as_view()), syndrome);
    /// ```
    pub fn syndrome_of(&self, error: &SparseBinSlice) -> SparseBinVec {
        if error.len() != self.block_size() {
            panic!(
                "error of length {} is invalid for code of size {}",
                error.len(),
                self.block_size()
            );
        }
        self.parity_check_matrix.dot_with(error)
    }

    /// Checks if an operator has zero syndrome.
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
    /// assert_eq!(hamming_code.has_codeword(&error.as_view()), false);
    /// assert_eq!(hamming_code.has_codeword(&codeword.as_view()), true);
    /// ```
    pub fn has_codeword(&self, operator: &SparseBinSlice) -> bool {
        self.syndrome_of(operator).is_zero()
    }
}
