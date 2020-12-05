# LDPC

A toolbox for classical (and soon quantum) LDPC codes.
For now, only classical linear codes are implemented.
There is also a generic implementation of noise model
that can be used to generate random error for codes.

## Example

```rust
use ldpc::LinearCode;
use ldpc::noise_model::BinarySymmetricChannel;
use rand::thread_rng;

// This sample a random regular LDPC code.
let code = LinearCode::random_regular_code()
    .block_size(40)
    .number_of_checks(20)
    .bit_degree(3)
    .check_degree(6)
    .sample_with(&mut thread_rng());

let noise = BinarySymmetricChannel::with_probability(0.1);

// The error is a sparse binary vector where each 1 represent a bit flip.
let error = code.random_error(&noise, &mut thread_rng());
```

