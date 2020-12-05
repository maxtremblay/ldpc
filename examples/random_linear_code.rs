use ldpc::LinearCode;
use rand::thread_rng;

fn main() {
    let code = LinearCode::random_regular_code()
        .block_size(4)
        .number_of_checks(3)
        .bit_degree(3)
        .check_degree(4)
        .sample_with(&mut thread_rng());
    println!("{:?}", code);
    println!("Block size: {}", code.block_size());
    println!("Dimension: {}", code.dimension());
    println!("Minimal distance: {:?}", code.minimal_distance());
}
