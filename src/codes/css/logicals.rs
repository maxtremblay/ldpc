use crate::css::Css;
use crate::codes::LinearCode;
use sparse_bin_mat::{BinNum, SparseBinMat, SparseBinVec};

// This implement a variation of the method
// introduced in https://arxiv.org/abs/0903.5256
// to compute logical operator generators of a CSS
// code from the linear codes.
pub(super) fn from_linear_codes(x_code: &LinearCode, z_code: &LinearCode) -> Css<SparseBinMat> {
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
            .map(|row| row.to_vec())
            .collect()
    }

    fn compute(mut self) -> Css<SparseBinMat> {
        while let Some(x_generator) = self.raw_x_generators.pop() {
            if let Some(z_generator) = self.find_anticommuting_z_generator(&x_generator) {
                self.update_remaining_generators(&x_generator, &z_generator);
                self.push_logicals(x_generator, z_generator);
            }
        }
        Css {
            x: SparseBinMat::new(self.length, self.x_logicals),
            z: SparseBinMat::new(self.length, self.z_logicals),
        }
    }

    fn anticommute(x_generator: &SparseBinVec, z_generator: &SparseBinVec) -> bool {
        x_generator.dot_with(z_generator).unwrap() == BinNum::one()
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
    use rand::thread_rng;

    #[test]
    fn steane_code() {
        let hamming = LinearCode::hamming_code();
        let logicals = from_linear_codes(&hamming, &hamming);
        assert_commutations(
            logicals,
            Css {
                x: hamming.parity_check_matrix(),
                z: hamming.parity_check_matrix(),
            },
        )
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
        let logicals = from_linear_codes(&x_code, &z_code);
        assert_commutations(logicals, Css {x: x_code.parity_check_matrix(), z: z_code.parity_check_matrix()});
    }

    #[test]
    fn random_hypergraph_product() {
        let code = LinearCode::random_regular_code()
            .num_bits(25)
            .num_checks(15)
            .bit_degree(3)
            .check_degree(5)
            .sample_with(&mut thread_rng())
            .unwrap();
        let logicals = from_linear_codes(&code, &code);
        assert_commutations(logicals, Css {x: code.parity_check_matrix(), z: code.parity_check_matrix()});
    }

    fn assert_commutations(logicals: Css<SparseBinMat>, par_matrices: Css<&SparseBinMat>) {
        assert_logicals_commute_with_stabilizers(&logicals.x, par_matrices.x);
        assert_logicals_commute_with_stabilizers(&logicals.z, par_matrices.z);
        assert_anticommuting_logical_pairs(&logicals.x, &logicals.z);
    }

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
