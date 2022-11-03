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
    todo!()
}

#[kernel]
pub unsafe fn main_legacy(seed: i64, rand_gens: *mut DefaultRand) {
    let index = thread::index();
    let mut this_gen = &mut *rand_gens.add(index as usize);

    let mut inc: u32 = 0;
    let mut buf: i64 = 0;

    let mut x = 0;
    let mut z = 0;

    loop {
        buf = this_gen.next_u64() as i64;

        x = (buf &  0xFFFFFFFF) as i32 >> 0x08;
        z = (buf >> 0x28      ) as i32;

        if filter(seed, x, z) {
            println!(
                "Found candidate position!   Coords: ({:8}, {:8})     [ID: {:5}, Counter: {:9}]",
                x, z, index, inc
            );
        }

        inc += 1;
    }

    // DefaultRand::seed_from_u64(thread::index());
    // let idx = thread::index_1d() as usize;
    // if idx < a.len() {
    //     let elem = &mut *c.add(idx);
    //     *elem = a[idx] + b[idx];
    // }
}