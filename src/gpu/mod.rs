pub(crate) use self::error::*;
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::fft::*;
#[cfg(any(feature = "cuda", feature = "opencl"))]
pub use self::locks::*;

mod program;

#[cfg(any(feature = "cuda", feature = "opencl"))]
mod locks;

mod error;
#[cfg(any(feature = "cuda", feature = "opencl"))]
mod fft;
#[cfg(any(feature = "cuda", feature = "opencl"))]
mod sources;

#[cfg(any(feature = "cuda", feature = "opencl"))]
pub fn kernel<'a>() -> Option<LockedFFTKernel<'a>> {
    Some(LockedFFTKernel::new(false))
}

#[cfg(not(all(feature = "cuda", feature = "opencl")))]
pub fn kernel<'a>() -> Option<LockedFFTKernel<'a>> {
    None
}
