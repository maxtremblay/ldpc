use super::{LinearDecoder, SyndromeDecoder};
use crate::noise::Probability;
use itertools::Itertools;
use sparse_bin_mat::{SparseBinMat, SparseBinSlice, SparseBinVec};
use sprs::{CsMat, TriMat};

#[derive(Debug, Clone)]
pub struct BpDecoder {
    parity_mat: SparseBinMat,
    transposed_mat: SparseBinMat,
    likelyhoods: Vec<f64>,
    num_iterations: usize,
}

impl LinearDecoder for BpDecoder {
    fn decode(&self, message: SparseBinSlice) -> SparseBinVec {
        let syndrome = &self.parity_mat * &message;
        let correction = self.correction_for(syndrome.as_view());
        &message + &correction
    }
}

impl<'a> SyndromeDecoder<SparseBinSlice<'a>, SparseBinVec> for BpDecoder {
    fn correction_for(&self, syndrome: SparseBinSlice) -> SparseBinVec {
        self.initialize_from(syndrome.as_view())
            .update_until(|state| {
                &(&self.parity_mat * &state.decode()).as_view() == &syndrome
                    || state.num_iterations == self.num_iterations
            })
            .decode()
    }
}

impl BpDecoder {
    pub fn new(parity_mat: &SparseBinMat, probability: Probability, num_iterations: usize) -> Self {
        let probability = probability.value();
        let likelyhoods = std::iter::repeat(((1.0 - probability) / probability).ln())
            .take(parity_mat.number_of_columns())
            .collect();
        Self {
            parity_mat: parity_mat.clone(),
            transposed_mat: parity_mat.transposed(),
            likelyhoods,
            num_iterations,
        }
    }

    fn initialize_from<'a>(&'a self, syndrome: SparseBinSlice<'a>) -> BpState<'a> {
        BpState {
            messages: Messages {
                bits: self.initialize_bits(),
                checks: self.initialize_checks(),
            },
            syndrome,
            likelyhoods: &self.likelyhoods,
            num_iterations: 0,
        }
    }

    fn initialize_bits(&self) -> CsMat<f64> {
        let mut messages = TriMat::new((self.num_checks(), self.num_bits()));
        for (check, bit) in self.parity_mat.non_trivial_elements() {
            messages.add_triplet(check, bit, self.likelyhoods[bit]);
        }
        messages.to_csr()
    }

    fn initialize_checks(&self) -> CsMat<f64> {
        let mut messages = TriMat::new((self.num_checks(), self.num_bits()));
        for (check, bits) in self.parity_mat.rows().enumerate() {
            for bit in bits.non_trivial_positions() {
                messages.add_triplet(check, bit, 0.0);
            }
        }
        messages.to_csc()
    }

    pub fn num_bits(&self) -> usize {
        self.parity_mat.number_of_columns()
    }

    pub fn num_checks(&self) -> usize {
        self.parity_mat.number_of_rows()
    }

    pub fn has_zero_syndrome(&self, vector: SparseBinSlice) -> bool {
        (&self.parity_mat * &vector).is_zero()
    }
}

#[derive(Debug, Clone, PartialEq)]
struct BpState<'a> {
    syndrome: SparseBinSlice<'a>,
    likelyhoods: &'a [f64],
    messages: Messages,
    num_iterations: usize,
}

impl<'a> BpState<'a> {
    fn decode(&self) -> SparseBinVec {
        let mut likelyhoods = self.likelyhoods.to_owned();
        for (bit, cols) in self.messages.checks.outer_iterator().enumerate() {
            for (_, value) in cols.iter() {
                likelyhoods[bit] += value;
            }
        }
        println!("likelyhoods: {:?}", likelyhoods);
        SparseBinVec::new(
            likelyhoods.len(),
            likelyhoods
                .iter()
                .positions(|likelyhood| *likelyhood < 0.0)
                .collect(),
        )
    }

    fn update_while<F>(mut self, condition: F) -> Self
    where
        F: Fn(&BpState) -> bool,
    {
        while condition(&self) {
            self = self.update_once();
        }
        self
    }

    fn update_until<F>(self, condition: F) -> Self
    where
        F: Fn(&BpState) -> bool,
    {
        self.update_while(|state| !condition(state))
    }

    fn update_once(mut self) -> Self {
        self.num_iterations += 1;
        self.messages = self
            .messages
            .update_checks(self.syndrome.clone())
            .update_bits(self.likelyhoods);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Messages {
    bits: CsMat<f64>,
    checks: CsMat<f64>,
}

impl Messages {
    fn update_checks(mut self, syndrome: SparseBinSlice) -> Self {
        let products = self
            .bits
            .outer_iterator()
            .map(|rows| rows.iter().map(|(_, v)| (v / 2.0).tanh()).product())
            .collect::<Vec<f64>>();
        for (bit, mut checks) in self.checks.outer_iterator_mut().enumerate() {
            for (check, value) in checks.iter_mut() {
                let inner = products[check] / (self.bits.get(check, bit).unwrap() / 2.0).tanh();
                *value = 2.0 * inner.atanh();
                if syndrome.get(check).unwrap().is_one() {
                    *value *= -1.0;
                }
            }
        }
        self
    }

    fn update_bits(mut self, likelyhoods: &[f64]) -> Self {
        let sums = self
            .checks
            .outer_iterator()
            .map(|cols| cols.iter().map(|(_, v)| *v).sum::<f64>())
            .collect_vec();
        for (check, mut bits) in self.bits.outer_iterator_mut().enumerate() {
            for (bit, value) in bits.iter_mut() {
                *value = sums[bit] - self.checks.get(check, bit).unwrap() + likelyhoods[bit]
            }
        }
        self
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::codes::LinearCode;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn no_error_for_hamming_code() {
        let code = LinearCode::hamming_code();
        let decoder = BpDecoder::new(code.parity_check_matrix(), Probability::new(0.1), 10);
        let error = SparseBinVec::new(7, Vec::new());
        assert_eq!(decoder.decode(error.as_view()), SparseBinVec::zeros(7));
    }

    #[test]
    fn flipping_first_bit_for_hamming_code() {
        let code = LinearCode::hamming_code();
        let decoder = BpDecoder::new(code.parity_check_matrix(), Probability::new(0.1), 10);
        let codeword = SparseBinVec::new(7, vec![0, 1, 2]);
        let error = SparseBinVec::new(7, vec![0]);
        let corrupted = &codeword + &error;
        let decoded = decoder.decode(corrupted.as_view());
        assert_eq!(decoded, codeword);
    }

    #[test]
    fn flipping_third_bit_for_hamming_code() {
        let code = LinearCode::hamming_code();
        let decoder = BpDecoder::new(code.parity_check_matrix(), Probability::new(0.1), 10);
        let codeword = SparseBinVec::new(7, vec![3, 4, 5, 6]);
        let error = SparseBinVec::new(7, vec![2]);
        let corrupted = &codeword + &error;
        let decoded = decoder.decode(corrupted.as_view());
        assert_eq!(decoded, codeword);
    }

    #[test]
    fn flipping_first_and_third_bit_for_hamming_code() {
        let code = LinearCode::hamming_code();
        let decoder = BpDecoder::new(code.parity_check_matrix(), Probability::new(0.1), 10);
        let codeword = SparseBinVec::new(7, vec![0, 2, 4, 6]);
        let error = SparseBinVec::new(7, vec![0, 2]);
        let corrupted = &codeword + &error;
        let decoded = decoder.decode(corrupted.as_view());
        let expected = SparseBinVec::new(7, vec![1, 4, 6]);
        assert_eq!(decoded, expected);
    }

    fn random_code() -> LinearCode {
        LinearCode::random_regular_code()
            .num_bits(16)
            .num_checks(12)
            .bit_degree(3)
            .check_degree(4)
            .sample_with(&mut StdRng::seed_from_u64(123))
            .unwrap()
    }

    #[test]
    fn no_error_for_random_code() {
        let code = random_code();
        let decoder = BpDecoder::new(code.parity_check_matrix(), Probability::new(0.1), 10);
        let error = SparseBinVec::new(16, Vec::new());
        assert_eq!(decoder.decode(error.as_view()), SparseBinVec::zeros(16));
    }

    #[test]
    fn flipping_first_bit_for_random_code() {
        let code = random_code();
        let decoder = BpDecoder::new(code.parity_check_matrix(), Probability::new(0.1), 10);
        let codeword = code.generator_matrix().row(0).unwrap();
        let error = SparseBinVec::new(code.len(), vec![0]);
        let corrupted = &codeword + &error;
        let decoded = decoder.decode(corrupted.as_view());
        assert_eq!(decoded.as_view(), codeword);
    }

    #[test]
    fn flipping_third_bit_for_random_code() {
        let code = random_code();
        let decoder = BpDecoder::new(code.parity_check_matrix(), Probability::new(0.1), 10);
        let codeword = code.generator_matrix().row(0).unwrap();
        let error = SparseBinVec::new(code.len(), vec![2]);
        let corrupted = &codeword + &error;
        let decoded = decoder.decode(corrupted.as_view());
        assert_eq!(decoded.as_view(), codeword);
    }

    #[test]
    fn flipping_two_bits_for_random_code() {
        let code = random_code();
        let decoder = BpDecoder::new(code.parity_check_matrix(), Probability::new(0.1), 10);
        let codeword = code.generator_matrix().row(0).unwrap();
        let error = SparseBinVec::new(code.len(), vec![0, 10]);
        let corrupted = &codeword + &error;
        let decoded = decoder.decode(corrupted.as_view());
        assert_eq!(decoded.as_view(), codeword);
    }
}
