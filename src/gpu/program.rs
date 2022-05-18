use std::env;

#[cfg(feature = "opencl")]
use ec_gpu_gen::Limb64;
#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;
#[cfg(feature = "opencl")]
use rust_gpu_tools::opencl;
use rust_gpu_tools::{Device, Framework, GPUError as GpuToolsError, Program};

use log::info;

#[cfg(not(all(feature = "cuda", feature = "opencl")))]
use crate::gpu::error::GPUError;
use crate::gpu::error::GPUResult;
#[cfg(feature = "opencl")]
use crate::gpu::sources;


/// Selects a CUDA or OpenCL on the `PLONK_GPU_FRAMEWORK` environment variable
/// and the compile-time features.
///
/// You cannot select CUDA if the library was compiled without support for it.
#[allow(clippy::unnecessary_wraps)] // No error can be returned if `cuda` and `opencl `are enabled.
fn select_framework(default_framework: Framework) -> GPUResult<Framework> {
    match env::var("PLONK_GPU_FRAMEWORK") {
        Ok(env) => match env.as_ref() {
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Framework::Cuda)
                }

                #[cfg(not(feature = "cuda"))]
                Err(GPUError::Simple("CUDA framework is not supported, please compile with the `cuda` feature enabled."))
            }
            "opencl" => {
                #[cfg(feature = "opencl")]
                {
                    Ok(Framework::Opencl)
                }

                #[cfg(not(feature = "opencl"))]
                Err(GPUError::Simple("OpenCL framework is not supported, please compile with the `opencl` feature enabled."))
            }
            _ => Ok(default_framework),
        },
        Err(_) => Ok(default_framework),
    }
}

/// Returns the program for the preferred [`rust_gpu_tools::device::Framework`].
///
/// If the device supports CUDA, then CUDA is used, else OpenCL. You can force a
/// selection with the environment variable `PLONK_GPU_FRAMEWORK`, which can
/// be set either to `cuda` or `opencl`.
pub fn program(device: &Device) -> GPUResult<Program>
{
    let framework = select_framework(device.framework())?;
    program_use_framework(device, framework)
}

/// Returns the program for the specified [`rust_gpu_tools::device::Framework`].
pub fn program_use_framework(
    device: &Device,
    framework: Framework,
) -> GPUResult<Program>
{
    match framework {
        #[cfg(feature = "cuda")]
        Framework::Cuda => {
            info!("Using kernel on CUDA.");
            let kernel = include_bytes!(env!("CUDA_PLONK_FATBIN"));
            let cuda_device =
                device.cuda_device().ok_or(GpuToolsError::DeviceNotFound)?;
            let program = cuda::Program::from_bytes(cuda_device, kernel)?;
            Ok(Program::Cuda(program))
        }
        #[cfg(feature = "opencl")]
        Framework::Opencl => {
            info!("Using kernel on OpenCL.");
            let src = sources::kernel::<Limb64>();
            let opencl_device = device
                .opencl_device()
                .ok_or(GpuToolsError::DeviceNotFound)?;
            let program = opencl::Program::from_opencl(opencl_device, &src)?;
            Ok(Program::Opencl(program))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils;

    use super::*;

    #[test]
    fn test_plonk_gpu_framework() {
        #[cfg(all(feature = "cuda", feature = "opencl"))]
        {
            // Use CUDA if set
            test_utils::with_env_vars(
                vec![("PLONK_GPU_FRAMEWORK", Some("cuda"))],
                || {
                    assert_eq!(
                        select_framework(Framework::Opencl).unwrap(),
                        Framework::Cuda
                    );
                },
            );

            // Use OpenCL if set
            test_utils::with_env_vars(
                vec![("PLONK_GPU_FRAMEWORK", Some("opencl"))],
                || {
                    assert_eq!(
                        select_framework(Framework::Cuda).unwrap(),
                        Framework::Opencl
                    );
                },
            );

            // Use default framework if arbitrary value is set
            test_utils::with_env_vars(
                vec![("PLONK_GPU_FRAMEWORK", None)],
                || {
                    assert_eq!(
                        select_framework(Framework::Cuda).unwrap(),
                        Framework::Cuda
                    );
                },
            );
            test_utils::with_env_vars(
                vec![("PLONK_GPU_FRAMEWORK", None)],
                || {
                    assert_eq!(
                        select_framework(Framework::Opencl).unwrap(),
                        Framework::Opencl
                    );
                },
            );
        }

        #[cfg(feature = "cuda")]
        {
            // Use default value if no value is set
            test_utils::with_env_vars(
                vec![("PLONK_GPU_FRAMEWORK", None)],
                || {
                    assert_eq!(
                        select_framework(Framework::Cuda).unwrap(),
                        Framework::Cuda
                    );
                },
            );

            // Use default framework if arbitrary value is set
            test_utils::with_env_vars(
                vec![("PLONK_GPU_FRAMEWORK", Some("notcudaoropencl"))],
                || {
                    assert_eq!(
                        select_framework(Framework::Cuda).unwrap(),
                        Framework::Cuda
                    );
                },
            );
        }

        #[cfg(feature = "opencl")]
        {
            // Use default value if no value is set
            test_utils::with_env_vars(
                vec![("PLONK_GPU_FRAMEWORK", None)],
                || {
                    assert_eq!(
                        select_framework(Framework::Opencl).unwrap(),
                        Framework::Opencl
                    );
                },
            );

            // Use default framework if arbitrary value is set
            test_utils::with_env_vars(
                vec![("PLONK_GPU_FRAMEWORK", Some("notcudaoropencl"))],
                || {
                    assert_eq!(
                        select_framework(Framework::Opencl).unwrap(),
                        Framework::Opencl
                    );
                },
            );
        }
    }
}
