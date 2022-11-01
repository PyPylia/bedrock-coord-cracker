use std::marker::PhantomData;

trait BaseRandom {
    fn new(seed: i64) -> Self;
    fn next(&mut self, bits: u8) -> i32;
    fn split(seed: i64, x: i64, y: i64, z: i64) -> Self;

    fn next_float(&mut self) -> f32 {
        self.next(24) as f32 * 5.9604645E-8
    }

    fn next_long(&mut self) -> i64 {
        let val1 = self.next(32);
        ((val1 as i64) << 32) + self.next(32) as i64
	}
}

struct CheckedRandom { seed: i64 }
impl BaseRandom for CheckedRandom {
    fn new(seed: i64) -> Self {
        Self { seed: (seed ^ 25214903917) & 281474976710655 }
    }

    fn next(&mut self, bits: u8) -> i32 {
        self.seed = (self.seed * 25214903917 + 11) & 281474976710655;
        (self.seed >> 48 - bits) as i32
    }

    fn split(seed: i64, x: i64, y: i64, z: i64) -> Self {
        let new_seed = (x * 3129871) ^ z * 116129781 ^ y;
        Self::new(((new_seed * new_seed * 42317861 + new_seed * 11) >> 16) ^ seed)
    }
}

struct Generator<T: BaseRandom> {
    seed: i64,
    bottom: i64,
    _phantom: PhantomData<T>
}

impl<T: BaseRandom> Generator<T> {
    fn new(world_seed: i64, bottom: i64) -> Self {
        // Java: "minecraft:bedrock_floor".hashCode() == 2042456806
        let seed = T::new(2042456806 ^ T::new(world_seed).next_long()).next_long();
        Self { seed: seed, bottom: bottom, _phantom: PhantomData }
    }

    fn is_bedrock(&self, x: i64, y: i64, z: i64) -> bool {
        let d = 1.0 + (y - self.bottom) as f32 / 5.0 * -1.0;
        T::split(self.seed, x, y, z).next_float() < d
    }
}

fn main() {
    let generator = Generator::<CheckedRandom>::new(64149200, 0);
    for x in -10..10 {
        let mut line = String::new();
        for z in -10..10 {
            line = line + if generator.is_bedrock(x, 4, z) {"██"} else {"  "};
        }

        println!("{}", line);
    }
}