use rand_core::{SeedableRng, RngCore, impls, Error};

pub struct CheckedRandom(pub [u8; 8]);

impl SeedableRng for CheckedRandom {
    type Seed = [u8; 8];

    fn seed_from_u64(seed: u64) -> Self {
        CheckedRandom(((seed ^ 25214903917) & 281474976710655).to_le_bytes())
    }

    fn from_seed(seed: [u8; 8]) -> Self {
        Self::seed_from_u64(u64::from_le_bytes(seed))
    }
}

impl RngCore for CheckedRandom {
    fn next_u32(&mut self) -> u32 {
        let seed = u64::from_le_bytes(self.0) * 25214903917 + 11 & 281474976710655;
        self.0 = seed.to_le_bytes();
        (seed >> 48 - 32) as u32
    }

    fn next_u64(&mut self) -> u64 {
        let val1 = self.next_u32();
        ((val1 as u64) << 32) + self.next_u32() as u64
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        impls::fill_bytes_via_next(self, dest)
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        Ok(self.fill_bytes(dest))
    }
}