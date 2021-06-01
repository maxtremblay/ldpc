use crate::LinearCode;
use pauli::PauliOperator;
use sparse_bin_mat::{SparseBinMat, SparseBinVec};

mod logicals;
use logicals::from_linear_codes;

mod syndrome;
pub use syndrome::CssSyndrome;

/// A quantum CSS code is defined from a pair of orthogonal linear codes.
///
/// The checks of the first code are used as a binary representation
/// of the X stabilizers while the checks of the second code are used
/// for the Z stabilizers.
///
/// The codewords of the first code are used to generate the X logical operators
/// while the codewords of the second code are used for the Z logical operators.
///
/// Any stabilizer generator or logical generator of a CSS code is
/// either composed of only Is and Xs or only Is and Zs.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct CssCode {
    x_stabilizers: SparseBinMat,
    z_stabilizers: SparseBinMat,
    x_logicals: SparseBinMat,
    z_logicals: SparseBinMat,
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
        let (x_logicals, z_logicals) = from_linear_codes(x_code, z_code);
        Ok(Self {
            x_stabilizers: x_code.parity_check_matrix().clone(),
            z_stabilizers: z_code.parity_check_matrix().clone(),
            x_logicals,
            z_logicals,
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
        let x_code = LinearCode::from_parity_check_matrix(SparseBinMat::new(
            9,
            vec![vec![0, 1, 2, 3, 4, 5], vec![3, 4, 5, 6, 7, 8]],
        ));
        let z_code = LinearCode::from_parity_check_matrix(SparseBinMat::new(
            9,
            vec![
                vec![0, 1],
                vec![1, 2],
                vec![3, 4],
                vec![4, 5],
                vec![6, 7],
                vec![7, 8],
            ],
        ));
        Self::new(&x_code, &z_code)
    }

    /// Returns the number of physical qubits in the code.
    pub fn len(&self) -> usize {
        self.x_stabilizers.number_of_columns()
    }

    /// Checks if the code has zero physical qubits.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of x stabilizer generators.
    pub fn num_x_stabs(&self) -> usize {
        self.x_stabilizers.number_of_rows()
    }

    /// Returns the number of z stabilizer generators.
    pub fn num_z_stabs(&self) -> usize {
        self.z_stabilizers.number_of_rows()
    }

    /// Returns the number of x logical generators.
    pub fn num_x_logicals(&self) -> usize {
        self.x_logicals.number_of_rows()
    }

    /// Returns the number of z logical generators.
    pub fn num_z_logicals(&self) -> usize {
        self.z_logicals.number_of_rows()
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
    /// # use ldpc::quantum::{CssCode, CssSyndrome};
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
        CssSyndrome {
            x: &self.x_stabilizers
                * &SparseBinVec::new(operator.len(), operator.z_part().into_raw_positions()),
            z: &self.z_stabilizers
                * &SparseBinVec::new(operator.len(), operator.x_part().into_raw_positions()),
        }
    }

    /// Checks if an operator commutes with all stabilizers.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::quantum::{CssCode, CssSyndrome};
    /// use pauli::{X, Z, PauliOperator};
    /// use sparse_bin_mat::SparseBinVec;
    ///
    /// let code = CssCode::shor_code();
    /// let logical_x = PauliOperator::new(9, vec![0, 3, 6], vec![Z, Z, Z]);
    /// assert!(code.has_logical(&logical_x));
    /// ```
    pub fn has_logical(&self, operator: &PauliOperator) -> bool {
        self.syndrome_of(operator).is_trivial()
    }


    /// Returns the binary matrix representing the X stabilizer
    /// generators in binary form.
    pub fn x_stabs_binary(&self) -> &SparseBinMat {
        &self.x_stabilizers
    }

    /// Returns the binary matrix representing the Z stabilizer
    /// generators in binary form.
    pub fn z_stabs_binary(&self) -> &SparseBinMat {
        &self.z_stabilizers
    }

    /// Returns an iterator throught all stabilizer generators of the code.
    ///
    /// # Example
    ///
    /// ```
    /// # use ldpc::quantum::CssCode;
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
        use pauli::{X, Z};
        self.x_stabilizers
            .rows()
            .map(move |stab| {
                PauliOperator::new(
                    self.len(),
                    stab.non_trivial_positions().collect(),
                    vec![X; stab.weight()],
                )
            })
            .chain(self.z_stabilizers.rows().map(move |stab| {
                PauliOperator::new(
                    self.len(),
                    stab.non_trivial_positions().collect(),
                    vec![Z; stab.weight()],
                )
            }))
    }

    /// Returns the binary matrix representing the X logical
    /// generators in binary form.
    pub fn x_logicals_binary(&self) -> &SparseBinMat {
        &self.x_logicals
    }

    /// Returns the binary matrix representing the Z logical
    /// generators in binary form.
    pub fn z_logicals_binary(&self) -> &SparseBinMat {
        &self.z_logicals
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn codes_should_have_same_length() {
        let small = LinearCode::repetition_code(3);
        let large = LinearCode::repetition_code(30);
        let css = CssCode::try_new(&small, &large);
        assert_eq!(css, Err(CssError::DifferentXandZLength(3, 30)));
    }

    #[test]
    fn codes_should_be_orthogonal() {
        let code = LinearCode::repetition_code(3);
        let css = CssCode::try_new(&code, &code);
        assert_eq!(css, Err(CssError::NonOrthogonalCodes));
    }

    #[test]
    fn syndrome_steane_code() {
        use pauli::{X, Y, Z};
        let code = CssCode::steane_code();
        let error = PauliOperator::new(7, vec![0, 3, 6], vec![X, Y, Z]);
        let expected = CssSyndrome {
            x: SparseBinVec::new(3, vec![1, 2]),
            z: SparseBinVec::new(3, vec![0, 2]),
        };
        assert_eq!(code.syndrome_of(&error), expected);
    }

    #[test]
    fn syndrome_shor_code() {
        use pauli::{X, Y, Z};
        let code = CssCode::shor_code();
        let error = PauliOperator::new(9, vec![0, 3, 6], vec![X, Y, Z]);
        let expected = CssSyndrome {
            x: SparseBinVec::new(2, vec![0]),
            z: SparseBinVec::new(6, vec![0, 2]),
        };
        assert_eq!(code.syndrome_of(&error), expected);
    }
}
