use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../gpu")
        .copy_to("../ptx/bedrock-coord-cracker.ptx")
        .build()
        .unwrap();
}