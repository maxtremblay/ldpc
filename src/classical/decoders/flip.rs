use crate::classical::LinearCode;
use sparse_bin_mat::{SparseBinVec, SparseBinVecBase};
use std::borrow::Borrow;
use std::fmt;

#[derive(Debug, Clone)]
pub struct FlipDecoder<Code> {
    code: Code,
}

impl<Code> FlipDecoder<Code> {
    pub fn new(code: Code) -> Self {
        Self { code }
    }
}

impl<Code> FlipDecoder<Code>
where
    Code: Borrow<LinearCode>,
{
    pub fn decode<T>(&self, message: &SparseBinVecBase<T>) -> SparseBinVec
    where
        T: std::ops::Deref<Target = [usize]>,
    {
        let mut syndrome = self.code().syndrome_of(message);
        let mut output = SparseBinVec::new(message.len(), message.as_slice().to_vec());
        while let Some(bit) = self.find_flippable(&syndrome) {
            let update = SparseBinVec::new(self.code().len(), vec![bit]);
            syndrome = &syndrome + &self.code().syndrome_of(&update);
            output = &output + &update;
        }
        output
    }

    fn find_flippable(&self, syndrome: &SparseBinVec) -> Option<usize> {
        self.code().bit_adjacencies().rows().position(|checks| {
            let number_unsatisfied = checks
                .non_trivial_positions()
                .filter(|check| syndrome.is_one_at(*check).unwrap_or(false))
                .count();
            number_unsatisfied > checks.weight() / 2
        })
    }

    fn code(&self) -> &LinearCode {
        self.code.borrow()
    }
}

impl<T> fmt::Display for FlipDecoder<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Flip decoder")
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn no_error_for_hamming_code() {
        let code = LinearCode::hamming_code();
        let decoder = FlipDecoder::new(code);
        let error = SparseBinVec::new(7, Vec::new());
        println!("OK");
        assert_eq!(decoder.decode(&error), SparseBinVec::zeros(7));
    }

    #[test]
    fn flipping_first_bit_for_hamming_code() {
        let code = LinearCode::hamming_code();
        let decoder = FlipDecoder::new(code);
        let codeword = SparseBinVec::new(7, vec![0, 1, 2]);
        let error = SparseBinVec::new(7, vec![0]);
        let corrupted = &codeword + &error;
        assert_eq!(decoder.decode(&corrupted), codeword);
    }

    #[test]
    fn flipping_third_bit_for_hamming_code() {
        let code = LinearCode::hamming_code();
        let decoder = FlipDecoder::new(&code);
        let codeword = SparseBinVec::new(7, vec![3, 4, 5, 6]);
        let error = SparseBinVec::new(7, vec![2]);
        let corrupted = &codeword + &error;
        let expected = SparseBinVec::new(7, (0..7).collect());
        assert_eq!(decoder.decode(&corrupted), expected);
    }

    #[test]
    fn flipping_first_and_third_bit_for_hamming_code() {
        let code = LinearCode::hamming_code();
        let decoder = FlipDecoder::new(&code);
        let codeword = SparseBinVec::new(7, vec![0, 2, 4, 6]);
        let error = SparseBinVec::new(7, vec![0, 2]);
        let corrupted = &codeword + &error;
        let expected = SparseBinVec::new(7, vec![1, 4, 6]);
        assert_eq!(decoder.decode(&corrupted), expected);
    }
}
