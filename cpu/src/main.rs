use cust::prelude::*;

static PTX: &str = include_str!("../../ptx/bedrock-coord-cracker.ptx");

fn main() -> anyhow::Result<()> {
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let func = module.get_function("main_legacy")?;
    let (grid_size, block_size) = func.suggested_launch_configuration(0, 0.into())?;

    println!(
        "Using {} blocks and {} threads per block\n",
        grid_size, block_size
    );

    unsafe { launch!(
        func<<<grid_size, block_size, 0, stream>>>(grid_size * block_size)
    )?; }

    stream.synchronize()?;

    Ok(())
}