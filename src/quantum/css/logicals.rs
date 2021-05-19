use crate::LinearCode;
use sparse_bin_mat::{SparseBinMat, SparseBinVec};

// This implement a variation of the method
// introduced in of https://arxiv.org/abs/0903.5256
// to compute logical operator generators of a CSS
// code from the linear codes.
pub(super) fn from_linear_codes(
    x_code: &LinearCode,
    z_code: &LinearCode,
) -> (SparseBinMat, SparseBinMat) {
    Logicals::new(x_code, z_code).compute()
}

struct Logicals {
    raw_x_generators: Vec<SparseBinVec>,
    raw_z_generators: Vec<SparseBinVec>,
    x_logicals: Vec<Vec<usize>>,
    z_logicals: Vec<Vec<usize>>,
    length: usize,
}

impl Logicals {
    fn new(x_code: &LinearCode, z_code: &LinearCode) -> Self {
        Self {
            raw_x_generators: Self::to_generator_vec(z_code),
            raw_z_generators: Self::to_generator_vec(x_code),
            x_logicals: Vec::new(),
            z_logicals: Vec::new(),
            length: x_code.len(),
        }
    }

    fn to_generator_vec(code: &LinearCode) -> Vec<SparseBinVec> {
        code.generator_matrix()
            .rows()
            .map(|row| row.to_owned())
            .collect()
    }

    fn compute(mut self) -> (SparseBinMat, SparseBinMat) {
        while let Some(x_generator) = self.raw_x_generators.pop() {
            if let Some(z_generator) = self.find_anticommuting_z_generator(&x_generator) {
                self.update_remaining_generators(&x_generator, &z_generator);
                self.push_logicals(x_generator, z_generator);
            }
        }
        (
            SparseBinMat::new(self.length, self.x_logicals),
            SparseBinMat::new(self.length, self.z_logicals),
        )
    }

    fn anticommute(x_generator: &SparseBinVec, z_generator: &SparseBinVec) -> bool {
        x_generator.dot_with(z_generator).unwrap() == 1
    }

    fn find_anticommuting_z_generator(
        &mut self,
        x_generator: &SparseBinVec,
    ) -> Option<SparseBinVec> {
        self.raw_z_generators
            .iter()
            .position(|z_generator| Self::anticommute(x_generator, z_generator))
            .map(|position| self.raw_z_generators.swap_remove(position))
    }

    fn update_remaining_generators(
        &mut self,
        x_generator: &SparseBinVec,
        z_generator: &SparseBinVec,
    ) {
        self.raw_z_generators
            .iter_mut()
            .filter(|gen| Self::anticommute(x_generator, gen))
            .for_each(|gen| *gen = z_generator + gen);
        self.raw_x_generators
            .iter_mut()
            .filter(|gen| Self::anticommute(gen, z_generator))
            .for_each(|gen| *gen = x_generator + gen);
    }

    fn push_logicals(&mut self, x_generator: SparseBinVec, z_generator: SparseBinVec) {
        self.x_logicals.push(x_generator.to_positions_vec());
        self.z_logicals.push(z_generator.to_positions_vec());
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn steane_code() {
        let hamming = LinearCode::hamming_code();
        let (x_logicals, z_logicals) = from_linear_codes(&hamming, &hamming);
        assert_logicals_commute_with_stabilizers(&x_logicals, hamming.parity_check_matrix());
        assert_logicals_commute_with_stabilizers(&z_logicals, hamming.parity_check_matrix());
        assert_anticommuting_logical_pairs(&x_logicals, &z_logicals);
    }

    #[test]
    fn shor_code() {
        let x_code = LinearCode::from_parity_check_matrix(SparseBinMat::new(
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
        let z_code = LinearCode::from_parity_check_matrix(SparseBinMat::new(
            9,
            vec![vec![0, 1, 2, 3, 4, 5], vec![3, 4, 5, 6, 7, 8]],
        ));
        let (x_logicals, z_logicals) = from_linear_codes(&x_code, &z_code);
        assert_logicals_commute_with_stabilizers(&x_logicals, z_code.parity_check_matrix());
        assert_logicals_commute_with_stabilizers(&z_logicals, x_code.parity_check_matrix());
        assert_anticommuting_logical_pairs(&x_logicals, &z_logicals);
    }

    // TODO: Add a test for some random codes when there is an implementation.
    // Most likely, this is going to be using the hypergraph product.

    fn assert_logicals_commute_with_stabilizers(
        logicals: &SparseBinMat,
        stabilizers: &SparseBinMat,
    ) -> bool {
        (logicals * &stabilizers.transposed()).is_zero()
    }

    fn assert_anticommuting_logical_pairs(
        x_logicals: &SparseBinMat,
        z_logicals: &SparseBinMat,
    ) -> bool {
        (x_logicals * &z_logicals.transposed())
            == SparseBinMat::identity(x_logicals.number_of_rows())
    }
}
