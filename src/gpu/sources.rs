use ec_gpu_gen::gen_base_code;
use ec_gpu_gen::Limb;

static FFT_SRC: &str = include_str!("cl/fft.cl");

fn fft(field: &str) -> String {
    String::from(FFT_SRC).replace("FIELD", field)
}

pub fn kernel<L: Limb>() -> String {
    [
        gen_base_code::<dusk_bls12_381::BlsScalar, dusk_bls12_381::Fp, L>(),
        fft("Fr"),
    ]
    .join("\n\n")
}
