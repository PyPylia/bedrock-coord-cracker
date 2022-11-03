use cuda_std::*;
use gpu_rand::{DefaultRand, xoroshiro::rand_core::RngCore};
use alloc;

fn is_bedrock(seed: i64, x: i32, z: i32, y: i64, d: f32) -> bool {
    let new_seed = (x * 3129871i32) as i64 ^ z as i64 * 116129781 ^ y;

    ((((((
        new_seed * new_seed * 42317861 + new_seed * 11
        ) >> 16
        ) ^ seed ^ 25214903917
        ) & 281474976710655
        ) * 25214903917 + 11 & 281474976710655
        ) >> 48 - 24
    ) as f32 * 5.9604645E-8 < d
}

fn filter(seed: i64, x: i32, z: i32) -> bool {
    is_bedrock(seed, x + 13, z + 8, 1, 0.8) &&
    is_bedrock(seed, x + 14, z + 8, 1, 0.8) &&
    is_bedrock(seed, x + 15, z + 8, 1, 0.8) &&
    !is_bedrock(seed, x + 15, z + 13, 4, 0.2) &&
    !is_bedrock(seed, x + 15, z + 14, 4, 0.2) &&
    !is_bedrock(seed, x + 15, z + 11, 4, 0.2) &&
    !is_bedrock(seed, x + 15, z + 8, 4, 0.2) &&
    !is_bedrock(seed, x + 15, z + 9, 4, 0.2) &&
    !is_bedrock(seed, x + 15, z + 4, 4, 0.2) &&
    !is_bedrock(seed, x + 15, z + 5, 4, 0.2) &&
    !is_bedrock(seed, x + 15, z + 6, 4, 0.2) &&
    !is_bedrock(seed, x + 14, z + 11, 4, 0.2) &&
    !is_bedrock(seed, x + 14, z + 12, 4, 0.2) &&
    !is_bedrock(seed, x + 14, z + 13, 4, 0.2) &&
    !is_bedrock(seed, x + 14, z + 14, 4, 0.2) &&
    !is_bedrock(seed, x + 14, z + 8, 4, 0.2) &&
    !is_bedrock(seed, x + 14, z + 9, 4, 0.2) &&
    !is_bedrock(seed, x + 14, z + 5, 4, 0.2) &&
    !is_bedrock(seed, x + 14, z + 6, 4, 0.2) &&
    !is_bedrock(seed, x + 13, z + 4, 4, 0.2) &&
    !is_bedrock(seed, x + 13, z + 5, 4, 0.2) &&
    !is_bedrock(seed, x + 13, z + 6, 4, 0.2) &&
    !is_bedrock(seed, x + 13, z + 8, 4, 0.2) &&
    !is_bedrock(seed, x + 13, z + 9, 4, 0.2) &&
    !is_bedrock(seed, x + 13, z + 11, 4, 0.2) &&
    !is_bedrock(seed, x + 13, z + 12, 4, 0.2) &&
    !is_bedrock(seed, x + 13, z + 13, 4, 0.2) &&
    !is_bedrock(seed, x + 13, z + 15, 4, 0.2) &&
    is_bedrock(seed, x + 13, z + 7, 2, 0.6) &&
    !is_bedrock(seed, x + 13, z + 8, 2, 0.6) &&
    is_bedrock(seed, x + 13, z + 9, 2, 0.6) &&
    is_bedrock(seed, x + 14, z + 7, 2, 0.6) &&
    !is_bedrock(seed, x + 14, z + 8, 2, 0.6) &&
    is_bedrock(seed, x + 15, z + 7, 2, 0.6) &&
    !is_bedrock(seed, x + 15, z + 8, 2, 0.6) &&
    is_bedrock(seed, x + 15, z + 9, 2, 0.6) &&
    is_bedrock(seed, x + 13, z + 4, 3, 0.4) &&
    !is_bedrock(seed, x + 13, z + 5, 3, 0.4) &&
    !is_bedrock(seed, x + 13, z + 6, 3, 0.4) &&
    is_bedrock(seed, x + 13, z + 7, 3, 0.4) &&
    !is_bedrock(seed, x + 13, z + 8, 3, 0.4) &&
    !is_bedrock(seed, x + 13, z + 9, 3, 0.4) &&
    !is_bedrock(seed, x + 14, z + 5, 3, 0.4) &&
    !is_bedrock(seed, x + 14, z + 6, 3, 0.4) &&
    is_bedrock(seed, x + 14, z + 7, 3, 0.4) &&
    !is_bedrock(seed, x + 14, z + 8, 3, 0.4) &&
    !is_bedrock(seed, x + 14, z + 9, 3, 0.4) &&
    !is_bedrock(seed, x + 14, z + 11, 3, 0.4) &&
    !is_bedrock(seed, x + 14, z + 12, 3, 0.4) &&
    !is_bedrock(seed, x + 14, z + 13, 3, 0.4) &&
    !is_bedrock(seed, x + 15, z + 4, 3, 0.4) &&
    is_bedrock(seed, x + 15, z + 5, 3, 0.4) &&
    is_bedrock(seed, x + 15, z + 6, 3, 0.4) &&
    is_bedrock(seed, x + 15, z + 7, 3, 0.4) &&
    !is_bedrock(seed, x + 15, z + 8, 3, 0.4) &&
    is_bedrock(seed, x + 15, z + 9, 3, 0.4) &&
    is_bedrock(seed, x + 15, z + 11, 3, 0.4) &&
    is_bedrock(seed, x + 15, z + 12, 3, 0.4) &&
    is_bedrock(seed, x + 15, z + 13, 3, 0.4) &&
    !is_bedrock(seed, x + 15, z + 14, 3, 0.4) &&
    is_bedrock(seed, x + 13, z + 10, 4, 0.2) &&
    is_bedrock(seed, x + 13, z + 7, 4, 0.2) &&
    is_bedrock(seed, x + 13, z + 14, 4, 0.2) &&
    is_bedrock(seed, x + 14, z + 4, 4, 0.2) &&
    is_bedrock(seed, x + 14, z + 7, 4, 0.2) &&
    is_bedrock(seed, x + 14, z + 10, 4, 0.2) &&
    is_bedrock(seed, x + 14, z + 15, 4, 0.2) &&
    is_bedrock(seed, x + 15, z + 7, 4, 0.2) &&
    is_bedrock(seed, x + 15, z + 10, 4, 0.2) &&
    is_bedrock(seed, x + 15, z + 12, 4, 0.2) &&
    is_bedrock(seed, x + 15, z + 15, 4, 0.2)
}

#[kernel]
pub unsafe fn main_legacy(seed: i64, rand_gens: *mut DefaultRand) {
    let index = thread::index();
    let mut this_gen = &mut *rand_gens.add(index as usize);

    let mut inc: u64 = 0;

    let mut x;
    let mut z;

    loop {
        x = this_gen.next_u32() as i32 >> 0x0D << 4;
        z = this_gen.next_u32() as i32 >> 0x0D << 4;

        if filter(seed, x, z) {
            println!(
                "Found candidate position!   Coords: ({:8}, {:8})     [ID: {:5}, Counter: {:9}]",
                x, z, index, inc
            );
        }

        if x > 4194250 && z > 4194250 { println!("Probably like {:12} chunks checked rn.", inc * 22 * 640); }

        inc += 1;
    }
}