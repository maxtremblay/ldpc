use super::LinearCode;
use sparse_bin_mat::NonTrivialElements;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub struct Edge {
    pub bit: usize,
    pub check: usize,
}

#[derive(Debug, Clone)]
pub struct Edges<'code> {
    elements: NonTrivialElements<'code>,
}

impl<'code> Edges<'code> {
    pub(super) fn new(code: &'code LinearCode) -> Self {
        Self {
            elements: code.parity_check_matrix().non_trivial_elements(),
        }
    }
}

impl<'code> Iterator for Edges<'code> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        self.elements.next().map(|(check, bit)| Edge { bit, check })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn edges_of_hamming_code() {
        let code = LinearCode::hamming_code();
        let mut edges = code.edges();

        assert_eq!(edges.next(), Some(Edge { bit: 3, check: 0 }));
        assert_eq!(edges.next(), Some(Edge { bit: 4, check: 0 }));
        assert_eq!(edges.next(), Some(Edge { bit: 5, check: 0 }));
        assert_eq!(edges.next(), Some(Edge { bit: 6, check: 0 }));

        assert_eq!(edges.next(), Some(Edge { bit: 1, check: 1 }));
        assert_eq!(edges.next(), Some(Edge { bit: 2, check: 1 }));
        assert_eq!(edges.next(), Some(Edge { bit: 5, check: 1 }));
        assert_eq!(edges.next(), Some(Edge { bit: 6, check: 1 }));

        assert_eq!(edges.next(), Some(Edge { bit: 0, check: 2 }));
        assert_eq!(edges.next(), Some(Edge { bit: 2, check: 2 }));
        assert_eq!(edges.next(), Some(Edge { bit: 4, check: 2 }));
        assert_eq!(edges.next(), Some(Edge { bit: 6, check: 2 }));

        assert_eq!(edges.next(), None);
    }
}
