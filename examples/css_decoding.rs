use ldpc::codes::{CssCode, LinearCode};
use ldpc::decoders::{BpDecoder, CssDecoder, SyndromeDecoder};
use ldpc::noise::{DepolarizingNoise, Probability};
use pauli::PauliOperator;
use rand::thread_rng;

fn main() {
    let max_iterations = 100;
    let probability = Probability::new(0.01);
    let code = code();
    let decoder = decoder(&code, probability, max_iterations);
    let noise = DepolarizingNoise::with_probability(probability);
    let mut failures = 0;
    for _ in 0..1000 {
        let error = code.random_error(&noise, &mut thread_rng());
        let syndrome = code.syndrome_of(&error);
        let correction: PauliOperator = decoder.correction_for(syndrome.as_view()).into();
        if !code.has_stabilizer(&(&error * &correction)) {
            failures += 1;
        }
    }
    println!("{}", failures);
}

fn code() -> CssCode {
    let code = LinearCode::random_regular_code()
        .num_bits(20)
        .num_checks(15)
        .bit_degree(3)
        .check_degree(4)
        .sample_with(&mut thread_rng())
        .unwrap();
    CssCode::hypergraph_product(&code, &code)
}

fn decoder(
    code: &CssCode,
    probability: Probability,
    max_iterations: usize,
) -> CssDecoder<BpDecoder> {
    CssDecoder {
        x: BpDecoder::new(&code.stabilizers.x, probability, max_iterations),
        z: BpDecoder::new(&code.stabilizers.z, probability, max_iterations),
    }
}
