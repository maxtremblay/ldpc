use ldpc::LinearCode;
use rand::thread_rng;

fn main() {
    let code = LinearCode::random_regular_code()
        .block_size(20)
        .number_of_checks(12)
        .bit_degree(3)
        .check_degree(5)
        .sample_with(&mut thread_rng());
    println!("{:?}", code);
}
