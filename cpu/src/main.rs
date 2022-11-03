mod random;

use cust::{prelude::*, memory::GpuBuffer};
use gpu_rand::DefaultRand;
use rand_core::{SeedableRng, RngCore};
use random::CheckedRandom;

const WORLD_SEED: i64 = 64149200;
const GENERATOR_HASH: u64 = 0x79BD6AE6;

const CUDA_SEED: u64 = 0x899D9B0927D947C6;
// Java: "minecraft:bedrock_floor".hashCode() == 2042456806

static PTX: &str = include_str!("../../ptx/bedrock-coord-cracker.ptx");

fn main() -> anyhow::Result<()> {
    
    let seed: i64 = CheckedRandom::seed_from_u64(
        GENERATOR_HASH ^ CheckedRandom::seed_from_u64(
            WORLD_SEED as u64
        ).next_u64()
    ).next_u64() as i64;

    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let func = module.get_function("main_legacy")?;
    let (grid_size, block_size) = func.suggested_launch_configuration(0, 0.into())?;

    println!(
        "Using {} blocks and {} threads per block\n",
        grid_size, block_size
    );

    let rand_gens = DeviceBuffer::from_slice(
        DefaultRand::initialize_states(
            CUDA_SEED,
            grid_size as usize * block_size as usize
        ).as_slice()
    )?;

    unsafe { launch!(
        func<<<grid_size, block_size, 0, stream>>>(seed, rand_gens.as_device_ptr())
    )?; }

    stream.synchronize()?;

    Ok(())
}