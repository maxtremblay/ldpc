use crate::{
    codes::LinearCode,
    css::{Css, CssOperator, CssSyndrome},
    noise::NoiseModel,
};
use pauli::{Pauli, PauliOperator};
use rand::Rng;
use serde::{Deserialize, Serialize};
use sparse_bin_mat::{SparseBinMat, SparseBinSlice};

mod logicals;
use logicals::from_linear_codes;

/// A quantum CSS code is defined from a pair of orthogonal linear codes.
/// The checks of the first code are used as a binary representation
/// of the X stabilizers while the checks of the second code are used
/// for the Z stabilizers.
///
/// The codewords of the first code are used to generate the X logical operators
/// while the codewords of the second code are used for the Z logical operators.
///
/// Any stabilizer generator or logical generator of a CSS code is
/// either composed of only Is and Xs or only Is and Zs.
#[derive(Debug, PartialEq, Eq, Clone, Hash, Serialize, Deserialize)]
pub struct CssCode {
    pub stabilizers: Css<SparseBinMat>,
    pub logicals: Css<SparseBinMat>,
}

impl CssCode {
    pub fn new(x_code: &LinearCode, z_code: &LinearCode) -> Self {
        Self::try_new(x_code, z_code).expect("[Error]")
    }

    pub fn try_new(x_code: &LinearCode, z_code: &LinearCode) -> Result<Self, CssError> {
        if x_code.len() != z_code.len() {
            return Err(CssError::DifferentXandZLength(x_code.len(), z_code.len()));
        } else if !(x_code.parity_check_matrix() * &z_code.parity_check_matrix().transposed())
            .is_zero()
        {
            return Err(CssError::NonOrthogonalCodes);
        }
        Ok(Self {
            stabilizers: Css {
                x: x_code.parity_check_matrix().clone(),
                z: z_code.parity_check_matrix().clone(),
            },
            logicals: from_linear_codes(x_code, z_code),
        })
    }

    /// Returns an instance of the Steane code which is construct
    /// from a pair of classical Hamming codes.
    pub fn steane_code() -> Self {
        let hamming_code = LinearCode::hamming_code();
        Self::new(&hamming_code, &hamming_code)
    }

    /// Returns an instance of the Shor code.
    pub fn shor_code() -> Self {
        Self {
            stabilizers: Css {
                x: SparseBinMat::new(9, vec![vec![0, 1, 2, 3, 4, 5], vec![3, 4, 5, 6, 7, 8]]),
                z: SparseBinMat::new(
                    9,
                    vec![
                        vec![0, 1],
                        vec![1, 2],
                        vec![3, 4],
                        vec![4, 5],
                        vec![6, 7],
                        vec![7, 8],
                    ],
                ),
            },
            logicals: Css {
                x: SparseBinMat::new(9, vec![vec![0, 1, 2]]),
                z: SparseBinMat::new(9, vec![vec![0, 3, 6]]),
            },
        }
    }

    /// Returns an instance of the toric code with given distance.
    pub fn toric_code(distance: usize) -> Self {
        let checks = (0..distance - 1)
            .map(|c| vec![c, c + 1])
            .chain(std::iter::once(vec![0, distance - 1]))
            .collect();
        let matrix = SparseBinMat::new(distance, checks);
        let code = LinearCode::from_parity_check_matrix(matrix);
        Self::hypergraph_product(&code, &code)
    }

    /// Returns the hypergraph product of two linear codes.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::codes::CssCode;
    /// # use ldpc::codes::LinearCode;
    /// let repetition_code = LinearCode::repetition_code(3);
    /// let surface_code = CssCode::hypergraph_product(&repetition_code, &repetition_code);
    ///
    /// use pauli::{PauliOperator, X, Z};
    ///
    /// let logical_x = PauliOperator::new(13, vec![0, 3, 6], vec![X, X, X]);
    /// assert!(surface_code.has_logical(&logical_x));
    ///
    /// let logical_z = PauliOperator::new(13, vec![0, 1, 2], vec![Z, Z, Z]);
    /// assert!(surface_code.has_logical(&logical_z));
    /// ```
    pub fn hypergraph_product(first_code: &LinearCode, second_code: &LinearCode) -> Self {
        let x_checks = Self::hypergraph_product_x_checks(first_code, second_code);
        let z_checks = Self::hypergraph_product_z_checks(first_code, second_code);
        Self::new(
            &LinearCode::from_parity_check_matrix(x_checks),
            &LinearCode::from_parity_check_matrix(z_checks),
        )
    }

    fn hypergraph_product_x_checks(
        first_code: &LinearCode,
        second_code: &LinearCode,
    ) -> SparseBinMat {
        SparseBinMat::identity(first_code.len())
            .kron_with(second_code.parity_check_matrix())
            .horizontal_concat_with(
                &first_code
                    .parity_check_matrix()
                    .transposed()
                    .kron_with(&SparseBinMat::identity(second_code.num_checks())),
            )
    }

    fn hypergraph_product_z_checks(
        first_code: &LinearCode,
        second_code: &LinearCode,
    ) -> SparseBinMat {
        first_code
            .parity_check_matrix()
            .kron_with(&SparseBinMat::identity(second_code.len()))
            .horizontal_concat_with(
                &SparseBinMat::identity(first_code.num_checks())
                    .kron_with(&second_code.parity_check_matrix().transposed()),
            )
    }

    /// Returns the number of physical qubits in the code.
    pub fn len(&self) -> usize {
        self.stabilizers.x.number_of_columns()
    }

    /// Checks if the code has zero physical qubits.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of x stabilizer generators.  
    pub fn num_x_stabs(&self) -> usize {
        self.stabilizers.x.number_of_rows()
    }

    /// Returns the number of z stabilizer generators.
    pub fn num_z_stabs(&self) -> usize {
        self.stabilizers.z.number_of_rows()
    }

    /// Returns the number of x logical generators.
    pub fn num_x_logicals(&self) -> usize {
        self.logicals.z.number_of_rows()
    }

    /// Returns the number of z logical generators.
    pub fn num_z_logicals(&self) -> usize {
        self.logicals.z.number_of_rows()
    }

    /// Returns both the X and Z parts of the syndrome of the given operator.
    ///
    /// The X part is the syndrome obtained from the X stabilizers and
    /// the Z part is the one from the Z stabilizers.
    /// Therefore, the X syndrome corresponds to the Z errors and vice-versa.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::codes::CssCode;
    /// # use ldpc::css::CssSyndrome;
    /// use pauli::{X, Z, PauliOperator};
    /// use sparse_bin_mat::SparseBinVec;
    ///
    /// let code = CssCode::shor_code();
    /// let error = PauliOperator::new(9, vec![1, 7], vec![X, Z]);
    /// let expected = CssSyndrome {
    ///     x: SparseBinVec::new(2, vec![1]),
    ///     z: SparseBinVec::new(6, vec![0, 1])
    /// };
    /// assert_eq!(code.syndrome_of(&error), expected);
    /// ```
    pub fn syndrome_of(&self, operator: &PauliOperator) -> CssSyndrome {
        self.stabilizers
            .as_ref()
            .pair(CssOperator::from(operator).swap_xz())
            .map(|(stabs, operator)| *stabs * operator)
    }

    /// Checks if an operator is a (potentially trivial) logical operator of the code.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::codes::CssCode;
    /// # use ldpc::css::CssSyndrome;
    /// use pauli::{X, Z, PauliOperator};
    /// use sparse_bin_mat::SparseBinVec;
    ///
    /// let code = CssCode::shor_code();
    ///
    /// let logical = PauliOperator::new(9, vec![0, 3, 6], vec![Z, Z, Z]);
    /// assert!(code.has_logical(&logical));
    ///
    /// let operator = PauliOperator::new(9, vec![0, 3, 6], vec![Z, X, Z]);
    /// assert!(!code.has_logical(&operator));
    /// ```
    pub fn has_logical(&self, operator: &PauliOperator) -> bool {
        self.syndrome_of(operator).is_trivial()
    }

    /// Checks if an operator is a stabilizer of the code.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::codes::CssCode;
    /// # use ldpc::css::CssSyndrome;
    /// use pauli::{X, Y, Z, PauliOperator};
    /// use sparse_bin_mat::SparseBinVec;
    ///
    /// let code = CssCode::shor_code();
    ///
    /// let stabilizer = PauliOperator::new(9, vec![0, 1, 2, 3, 4, 5], vec![X, X, X, X, Y, Y]);
    /// assert!(code.has_stabilizer(&stabilizer));
    ///
    /// let operator = PauliOperator::new(9, vec![0, 1, 2, 3, 4, 6], vec![X, X, X, X, Y, Z]);
    /// assert!(!code.has_stabilizer(&operator));
    /// ```
    pub fn has_stabilizer(&self, operator: &PauliOperator) -> bool {
        self.has_logical(operator)
            && self
                .logicals()
                .all(|logical| logical.commutes_with(operator))
    }

    /// Returns the binary matrix representing the X stabilizer
    /// generators in binary form.
    pub fn x_stabs_binary(&self) -> &SparseBinMat {
        &self.stabilizers.x
    }

    /// Returns the binary matrix representing the Z stabilizer
    /// generators in binary form.
    pub fn z_stabs_binary(&self) -> &SparseBinMat {
        &self.stabilizers.z
    }

    /// Returns the binary matrix representing the X logical
    /// generators in binary form.
    pub fn x_logicals_binary(&self) -> &SparseBinMat {
        &self.logicals.x
    }

    /// Returns the binary matrix representing the Z logical
    /// generators in binary form.
    pub fn z_logicals_binary(&self) -> &SparseBinMat {
        &self.logicals.z
    }

    /// Returns an iterator throught all stabilizer generators of the code.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::codes::CssCode;
    /// use pauli::{PauliOperator, X, Z};
    ///
    /// let code = CssCode::steane_code();
    /// let mut stabilizers = code.stabilizers();
    ///
    /// assert_eq!(stabilizers.next(), Some(PauliOperator::new(7, vec![3, 4, 5, 6], vec![X; 4])));
    /// assert_eq!(stabilizers.next(), Some(PauliOperator::new(7, vec![1, 2, 5, 6], vec![X; 4])));
    /// assert_eq!(stabilizers.next(), Some(PauliOperator::new(7, vec![0, 2, 4, 6], vec![X; 4])));
    ///
    /// assert_eq!(stabilizers.next(), Some(PauliOperator::new(7, vec![3, 4, 5, 6], vec![Z; 4])));
    /// assert_eq!(stabilizers.next(), Some(PauliOperator::new(7, vec![1, 2, 5, 6], vec![Z; 4])));
    /// assert_eq!(stabilizers.next(), Some(PauliOperator::new(7, vec![0, 2, 4, 6], vec![Z; 4])));
    ///
    /// assert!(stabilizers.next().is_none());
    /// ```
    pub fn stabilizers<'a>(&'a self) -> impl Iterator<Item = PauliOperator> + 'a {
        self.stabilizers
            .map_with_pauli(move |stabs, pauli| {
                stabs
                    .rows()
                    .map(move |stab| Self::operator_from_vec(pauli, stab))
            })
            .combine_with(Iterator::chain)
    }

    /// Returns an iterator throught all logical operator generators of the code.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::codes::CssCode;
    /// use pauli::{PauliOperator, X, Z};
    ///
    /// let code = CssCode::shor_code();
    /// let mut logicals = code.logicals();
    ///
    /// assert_eq!(logicals.next(), Some(PauliOperator::new(9, vec![0, 1, 2], vec![X; 3])));
    /// assert_eq!(logicals.next(), Some(PauliOperator::new(9, vec![0, 3, 6], vec![Z; 3])));
    /// ```
    pub fn logicals<'a>(&'a self) -> impl Iterator<Item = PauliOperator> + 'a {
        self.logicals
            .map_with_pauli(move |stabs, pauli| {
                stabs
                    .rows()
                    .map(move |stab| Self::operator_from_vec(pauli, stab))
            })
            .combine_with(Iterator::chain)
    }

    fn operator_from_vec(pauli: Pauli, vector: SparseBinSlice) -> PauliOperator {
        PauliOperator::new(
            vector.len(),
            vector.non_trivial_positions().collect(),
            vec![pauli; vector.weight()],
        )
    }

    /// Generates a random error with the given noise model.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::codes::CssCode;
    /// use ldpc::noise::{DepolarizingNoise, Probability};
    /// use rand::thread_rng;
    ///
    /// let code = CssCode::steane_code();
    ///
    /// let noise = DepolarizingNoise::with_probability(Probability::new(0.25));
    /// let error = code.random_error(&noise, &mut thread_rng());
    ///
    /// assert_eq!(error.len(), 7);
    /// ```
    pub fn random_error<N, R>(&self, noise_model: &N, rng: &mut R) -> PauliOperator
    where
        N: NoiseModel<Error = PauliOperator>,
        R: Rng,
    {
        noise_model.sample_error_of_length(self.len(), rng)
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum CssError {
    DifferentXandZLength(usize, usize),
    NonOrthogonalCodes,
}

impl std::fmt::Display for CssError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DifferentXandZLength(x_length, z_length) => {
                write!(f, "different x and z lengths: {} & {}", x_length, z_length)
            }
            Self::NonOrthogonalCodes => write!(f, "codes are not orthogonal"),
        }
    }
}

impl std::error::Error for CssError {}
