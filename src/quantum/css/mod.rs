use crate::LinearCode;
use sparse_bin_mat::SparseBinMat;

mod logicals;
use logicals::from_linear_codes;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct CssCode {
    x_stabilizers: SparseBinMat,
    z_stabilizers: SparseBinMat,
    x_logicals: SparseBinMat,
    z_logicals: SparseBinMat,
}

impl CssCode {
    pub fn from_x_and_z_linear_codes(x_code: &LinearCode, z_code: &LinearCode) -> Self {
        Self::try_from_x_and_z_linear_codes(x_code, z_code).expect("[Error]")
    }

    pub fn try_from_x_and_z_linear_codes(
        x_code: &LinearCode,
        z_code: &LinearCode,
    ) -> Result<Self, CssError> {
        if x_code.len() != z_code.len() {
            return Err(CssError::DifferentXandZLength(x_code.len(), z_code.len()));
        } else if !(x_code.parity_check_matrix() * &z_code.parity_check_matrix().transposed()).is_zero() {
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
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum CssError {
    DifferentXandZLength(usize, usize),
    NonOrthogonalCodes,
}

impl std::fmt::Display for CssError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DifferentXandZLength(x_length, z_length) => write!(
                f,
                "codes have different lengths: {} & {}",
                x_length, z_length
            ),
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
        let css = CssCode::try_from_x_and_z_linear_codes(&small, &large);
        assert_eq!(css, Err(CssError::DifferentXandZLength(3, 30)));
    }

    #[test]
    fn codes_should_be_orthogonal() {
        let code = LinearCode::repetition_code(3);
        let css = CssCode::try_from_x_and_z_linear_codes(&code, &code);
        assert_eq!(css, Err(CssError::NonOrthogonalCodes));

    }
}
