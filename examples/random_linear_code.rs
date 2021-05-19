use ldpc::noise_model::{BinarySymmetricChannel, Probability};
use ldpc::LinearCode;
use rand::thread_rng;

fn main() {
    let code = LinearCode::random_regular_code()
        .number_of_bits(4)
        .number_of_checks(3)
        .bit_degree(3)
        .check_degree(4)
        .sample_with(&mut thread_rng())
        .unwrap();

    let noise = BinarySymmetricChannel::with_probability(Probability::new(0.2));
    let error = code.random_error(&noise, &mut thread_rng());
    println!("{}", error);
}
