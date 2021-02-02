use super::BeliefPropagationBase;
use crate::linear_code::Edge;
use sparse_bin_mat::SparseBinVec;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(super) struct BitToCheckMessages<'decoder> {
    decoder: &'decoder BeliefPropagationBase,
    messages: HashMap<Edge, f64>,
}

impl<'decoder> From<&'decoder BeliefPropagationBase> for BitToCheckMessages<'decoder> {
    fn from(decoder: &'decoder BeliefPropagationBase) -> Self {
        Self {
            decoder,
            messages: decoder
                .edges()
                .map(|edge| (edge, decoder.initial_likelyhood_of(edge.bit).unwrap()))
                .collect(),
        }
    }
}

impl<'decoder> BitToCheckMessages<'decoder> {
    pub(super) fn update_from(&self, check_to_bit_messages: &CheckToBitMessages) -> Self {
        let mut updated_messages = HashMap::with_capacity(self.messages.len());
        self.decoder.edges().for_each(|edge| {
            updated_messages.insert(
                edge,
                self.updated_message_of_edge_from(edge, check_to_bit_messages),
            );
        });
        Self {
            decoder: self.decoder,
            messages: updated_messages,
        }
    }

    fn updated_message_of_edge_from(
        &self,
        edge: Edge,
        check_to_bit_messages: &CheckToBitMessages,
    ) -> f64 {
        self.decoder.initial_likelyhood_of(edge.bit).unwrap()
            + check_to_bit_messages.sum_at_edges(self.other_edges_connected_to_bit(edge))
    }

    fn other_edges_connected_to_bit(&self, edge: Edge) -> impl Iterator<Item = Edge> + '_ {
        self.decoder
            .edges_connected_to_bit(edge.bit)
            .unwrap()
            .filter(move |other_edge| edge.check != other_edge.check)
    }

    fn product_with_tanh_at_edges<E>(&self, edges: E) -> f64
    where
        E: IntoIterator<Item = Edge>,
    {
        edges
            .into_iter()
            .filter_map(|edge| self.messages.get(&edge))
            .map(|message| (message / 2.0).tanh())
            .product()
    }
}

#[derive(Debug, Clone)]
pub(super) struct CheckToBitMessages<'decoder> {
    decoder: &'decoder BeliefPropagationBase,
    messages: HashMap<Edge, f64>,
}

impl<'decoder> From<&'decoder BeliefPropagationBase> for CheckToBitMessages<'decoder> {
    fn from(decoder: &'decoder BeliefPropagationBase) -> Self {
        Self {
            decoder,
            messages: decoder.edges().map(|edge| (edge, 0.0)).collect(),
        }
    }
}

impl<'decoder> CheckToBitMessages<'decoder> {
    pub(super) fn update_from(
        &self,
        bit_to_check_messages: &BitToCheckMessages,
        syndrome: &SparseBinaryVector,
    ) -> Self {
        let mut updated_messages = HashMap::with_capacity(self.messages.len());
        self.decoder.edges().for_each(|edge| {
            updated_messages.insert(
                edge,
                self.updated_message_of_edge_from(edge, bit_to_check_messages, syndrome),
            );
        });
        Self {
            decoder: self.decoder,
            messages: updated_messages,
        }
    }

    fn updated_message_of_edge_from(
        &self,
        edge: Edge,
        bit_to_check_messages: &BitToCheckMessages,
        syndrome: &SparseBinaryVector,
    ) -> f64 {
        let message = 2.0
            * bit_to_check_messages
                .product_with_tanh_at_edges(self.other_edges_connected_to_check(edge))
                .atanh();
        if syndrome.get(edge.check).unwrap().is_one() {
            -1.0 * message
        } else {
            message
        }
    }

    fn other_edges_connected_to_check(&self, edge: Edge) -> impl Iterator<Item = Edge> + '_ {
        self.decoder
            .edges_connected_to_z_stabilizer(edge.check)
            .filter(move |other_edge| edge.bit != other_edge.bit)
    }

    pub(super) fn sum_at_edges<E>(&self, edges: E) -> f64
    where
        E: IntoIterator<Item = Edge>,
    {
        edges
            .into_iter()
            .filter_map(|edge| self.messages.get(&edge))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codes::BinaryCode;
    use assert_approx_eq::assert_approx_eq;
    use binary::SparseBinaryMatrix;

    #[test]
    fn messages_for_single_error_on_repetition_code() {
        let code = repetition_code();
        let decoder =
            BeliefPropagationBase::with_code_and_error_probabilities(code, &probabilities());

        let mut bit_to_check = BitToCheckMessages::from(&decoder);
        let mut check_to_bit = CheckToBitMessages::from(&decoder);

        check_to_bit = check_to_bit.update_from(&bit_to_check, &syndrome());
        assert_check_to_bit_messages_are_approx_eq(
            &check_to_bit,
            &CheckToBitMessages {
                decoder: &decoder,
                messages: expected_check_to_bit_messages(),
            },
        );

        bit_to_check = bit_to_check.update_from(&check_to_bit);
        assert_bit_to_check_messages_are_approx_eq(
            &bit_to_check,
            &BitToCheckMessages {
                decoder: &decoder,
                messages: expected_bit_to_check_messages(),
            },
        );
    }

    fn expected_check_to_bit_messages() -> HashMap<Edge, f64> {
        let likelyhoods = initial_likelyhoods();
        let mut messages = HashMap::new();
        messages.insert(Edge { bit: 0, check: 0 }, likelyhoods[1]);
        messages.insert(Edge { bit: 1, check: 0 }, likelyhoods[0]);
        messages.insert(Edge { bit: 1, check: 1 }, -1.0 * likelyhoods[2]);
        messages.insert(Edge { bit: 2, check: 1 }, -1.0 * likelyhoods[1]);
        messages
    }

    fn assert_check_to_bit_messages_are_approx_eq(
        first_messages: &CheckToBitMessages,
        second_messages: &CheckToBitMessages,
    ) {
        for edge in edges() {
            let first_message = first_messages.messages.get(&edge).unwrap();
            let second_message = second_messages.messages.get(&edge).unwrap();
            assert_approx_eq!(first_message, second_message)
        }
    }

    fn expected_bit_to_check_messages() -> HashMap<Edge, f64> {
        let likelyhoods = initial_likelyhoods();
        let mut messages = HashMap::new();
        messages.insert(Edge { bit: 0, check: 0 }, likelyhoods[0]);
        messages.insert(Edge { bit: 1, check: 0 }, likelyhoods[1] - likelyhoods[2]);
        messages.insert(Edge { bit: 1, check: 1 }, likelyhoods[1] + likelyhoods[0]);
        messages.insert(Edge { bit: 2, check: 1 }, likelyhoods[2]);
        messages
    }

    fn assert_bit_to_check_messages_are_approx_eq(
        first_messages: &BitToCheckMessages,
        second_messages: &BitToCheckMessages,
    ) {
        for edge in edges() {
            let first_message = first_messages.messages.get(&edge).unwrap();
            let second_message = second_messages.messages.get(&edge).unwrap();
            assert_approx_eq!(first_message, second_message)
        }
    }

    fn repetition_code() -> BinaryCode {
        let checks = vec![
            SparseBinaryVector::with_length_and_one_positions(3, vec![0, 1]),
            SparseBinaryVector::with_length_and_one_positions(3, vec![1, 2]),
        ];
        SparseBinaryMatrix::with_rows(checks).into()
    }

    fn probabilities() -> Vec<f64> {
        vec![0.1, 0.2, 0.3]
    }

    fn initial_likelyhoods() -> Vec<f64> {
        probabilities()
            .into_iter()
            .map(|prob| ((1.0 - prob) / prob).ln())
            .collect()
    }

    fn syndrome() -> SparseBinaryVector {
        SparseBinaryVector::with_length_and_one_positions(2, vec![1])
    }

    fn edges() -> Vec<Edge> {
        vec![
            Edge { bit: 0, check: 0 },
            Edge { bit: 1, check: 0 },
            Edge { bit: 1, check: 1 },
            Edge { bit: 2, check: 1 },
        ]
    }
}
