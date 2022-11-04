#[allow(unused_imports)]
use alloc;
use cuda_std::{*, println};

const WORLD_SEED:   i64 =  64149200;

const SEARCH_START: i32 = -234_375;
const SEARCH_END:   i32 =  234_375;
const SEARCH_SHIFT: u8  =  4;

const GEN_ID: u64 = 2042456806;
const RAND_A: u64 = 25214903917;
const RAND_B: u64 = 281474976710655;

const SEARCH_SIZE: u64 =
      SEARCH_START.abs() as u64 +
      SEARCH_END  .abs() as u64;

const SEARCH_BREAK: i32 = SEARCH_END << SEARCH_SHIFT;

#[inline(always)]
const fn next_bits(val: u64) -> u64 {
    RAND_A.wrapping_mul(val) + 11 & RAND_B
}

#[inline(always)]
const fn new_random(val: u64) -> u64 {
    next_bits((val ^ RAND_A) & RAND_B)
}

#[inline(always)]
const fn next_long(val: u64) -> u64 {
    (val >> 16 << 32) + ((next_bits(val) >> 16) as u32) as u64
}

const fn hash_check(seed: u64) -> u64 {
    (   (   (   (
                    seed * seed * 42317861 + seed * 11
                ) >> 0x10
            ) ^ const {
                next_long(
                    new_random(
                        next_long(
                            new_random(WORLD_SEED as u64)
                        ) ^ GEN_ID
                    )
                ) ^ RAND_A
            }
        ) & RAND_B
    ) * RAND_A + 11 & RAND_B
}

macro_rules! is_bedrock {
    ($x:expr, $y:expr, $z:expr) => {
        hash_check(
            ($x * 3129871i32) as u64 ^ $z as u64 * 116129781 ^ $y as u64
        ) < const { (5 - $y) as u64 * (0x333333 << 24) }
    };
}

fn filter(x: i32, z: i32) -> bool {
    !is_bedrock!(x,     2, z    ) && // AMOGUS
     is_bedrock!(x + 1, 2, z    ) && // AMOGUS
     is_bedrock!(x + 2, 2, z    ) && // AMOGUS
     is_bedrock!(x + 3, 2, z    ) && // AMOGUS
     is_bedrock!(x,     2, z + 1) && // AMOGUS
     is_bedrock!(x + 1, 2, z + 1) && // AMOGUS
    !is_bedrock!(x + 2, 2, z + 1) && // AMOGUS
    !is_bedrock!(x + 3, 2, z + 1) && // AMOGUS
     is_bedrock!(x,     2, z + 2) && // AMOGUS
     is_bedrock!(x + 1, 2, z + 2) && // AMOGUS
     is_bedrock!(x + 2, 2, z + 2) && // AMOGUS
     is_bedrock!(x + 3, 2, z + 2) && // AMOGUS
    !is_bedrock!(x,     2, z + 3) && // AMOGUS
     is_bedrock!(x + 1, 2, z + 3) && // AMOGUS
     is_bedrock!(x + 2, 2, z + 3) && // AMOGUS
     is_bedrock!(x + 3, 2, z + 3) && // AMOGUS
    !is_bedrock!(x,     2, z + 4) && // AMOGUS
     is_bedrock!(x + 1, 2, z + 4) && // AMOGUS
    !is_bedrock!(x + 2, 2, z + 4) && // AMOGUS
     is_bedrock!(x + 3, 2, z + 4)    // AMOGUS
}

#[kernel]
pub unsafe fn main_legacy(thread_count: u32) {
    let step_size = thread_count as u64;
    let index = thread::index();

    let mut inc: u64 = index as u64;
    let mut x: i32;
    let mut z: i32;

    let mut last_x = SEARCH_START << SEARCH_SHIFT;

    loop {
        x = ((inc / SEARCH_SIZE) as i32 + SEARCH_START) << SEARCH_SHIFT;
        z = ((inc % SEARCH_SIZE) as i32 + SEARCH_START) << SEARCH_SHIFT;

        if x > SEARCH_BREAK {
            break;
        }

        if filter(x, z) {
            println!(
                "Found candidate position!   Coords: ({:8}, {:8})     [ID: {:5}]",
                x, z, index
            );
        }

        if index == 0 && (x - last_x) > 200000 {
            println!("Currently: ({:8}, {:8}) | {:12}", x, z, inc);
            last_x = x;
        }

        inc += step_size;
    }
}