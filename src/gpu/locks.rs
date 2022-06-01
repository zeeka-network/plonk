use std::fs::File;
use std::path::PathBuf;

use fs2::FileExt;
use log::{debug, info, warn};

use crate::gpu::error::{GPUError, GPUResult};
use crate::gpu::fft::{create_fft_kernel, FFTKernel};

const GPU_LOCK_NAME: &str = "plonk.gpu.lock";
const PRIORITY_LOCK_NAME: &str = "plonk.priority.lock";
fn tmp_path(filename: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(filename);
    p
}

/// `GPULock` prevents two kernel objects to be instantiated simultaneously.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct GPULock(File);
impl GPULock {
    pub fn lock() -> GPUResult<GPULock> {
        let gpu_lock_file = tmp_path(GPU_LOCK_NAME);
        debug!("Acquiring GPU lock at {:?} ...", &gpu_lock_file);
        let f = File::create(&gpu_lock_file).unwrap_or_else(|_| {
            panic!("Cannot create GPU lock file at {:?}", &gpu_lock_file)
        });
        f.lock_exclusive().map_err(|e| {
            debug!("try to lock GPU failed. error: {}", e);
            GPUError::Simple("GPU lock was busy. ")
        })?;
        debug!("GPU lock acquired!");
        Ok(GPULock(f))
    }
}
impl Drop for GPULock {
    fn drop(&mut self) {
        self.0.unlock().unwrap();
        debug!("GPU lock released!");
    }
}

/// `PrioriyLock` is like a flag. When acquired, it means a high-priority
/// process needs to acquire the GPU really soon. Acquiring the `PriorityLock`
/// is like signaling all other processes to release their `GPULock`s.
/// Only one process can have the `PriorityLock` at a time.
#[derive(Debug)]
pub struct PriorityLock(File);
impl PriorityLock {
    pub fn lock() -> PriorityLock {
        let priority_lock_file = tmp_path(PRIORITY_LOCK_NAME);
        debug!("Acquiring priority lock at {:?} ...", &priority_lock_file);
        let f = File::create(&priority_lock_file).unwrap_or_else(|_| {
            panic!(
                "Cannot create priority lock file at {:?}",
                &priority_lock_file
            )
        });
        f.lock_exclusive().unwrap();
        debug!("Priority lock acquired!");
        PriorityLock(f)
    }

    pub fn wait(priority: bool) {
        if !priority {
            if let Err(err) = File::create(tmp_path(PRIORITY_LOCK_NAME))
                .unwrap()
                .try_lock_exclusive()
            {
                warn!("failed to create priority log: {:?}", err);
            }
        }
    }

    pub fn should_break(priority: bool) -> bool {
        if priority {
            return false;
        }
        if let Err(err) = File::create(tmp_path(PRIORITY_LOCK_NAME))
            .unwrap()
            .try_lock_shared()
        {
            // Check that the error is actually a locking one
            if err.raw_os_error() == fs2::lock_contended_error().raw_os_error()
            {
                return true;
            } else {
                warn!("failed to check lock: {:?}", err);
            }
        }
        false
    }
}

impl Drop for PriorityLock {
    fn drop(&mut self) {
        self.0.unlock().unwrap();
        debug!("Priority lock released!");
    }
}

macro_rules! locked_kernel {
    ($class:ident, $kern:ident, $func:ident, $name:expr) => {
        #[allow(clippy::upper_case_acronyms)]
        pub struct $class<'a>
        {
            priority: bool,
            kernel_and_lock: Option<($kern<'a>, GPULock)>,
        }

        impl<'a> $class<'a>
        {
            pub fn new(priority: bool) -> $class<'a> {
                $class {
                    priority,
                    kernel_and_lock: None,
                }
            }

            fn init(&mut self) -> GPUResult<()> {
                if self.kernel_and_lock.is_none() {
                    PriorityLock::wait(self.priority);
                    info!("GPU is available for {}!", $name);
                    let lock = GPULock::lock().map_err(|e| {
                        debug!("try to acquiring GPU again because of {}", e);
                        e
                    })?;
                    if let Some(kernel) = $func(self.priority) {
                        self.kernel_and_lock = Some((kernel, lock));
                    }
                }
                Ok(())
            }

            fn free(&mut self) {
                if let Some(_kernel) = self.kernel_and_lock.take() {
                    warn!(
                        "GPU acquired by a high priority process! Freeing up {} kernels...",
                        $name
                    );
                }
            }

            pub fn with<F, R>(&mut self, mut f: F) -> GPUResult<R>
            where
                F: FnMut(&mut $kern) -> GPUResult<R>,
            {
                if std::env::var("PLONK_NO_GPU").is_ok() {
                    return Err(GPUError::GPUDisabled);
                }
                loop {
                    // `init()` is a possibly blocking call that waits until the GPU is available.
                    if self.init().is_err() {
                        continue;
                    }
                    if let Some((ref mut k, ref _gpu_lock)) = self.kernel_and_lock {
                        let gpu_ret = f(k);
                        match gpu_ret {
                            // Re-trying to run on the GPU is the core of this loop, all other
                            // cases abort the loop.
                            Err(GPUError::GPUTaken) => {
                                self.free();
                            }
                            Err(e) => {
                                warn!("GPU {} failed! Falling back to CPU... Error: {}", $name, e);
                                return Err(e);
                            }
                            Ok(v) => {
                                return Ok(v);
                            }
                        }
                    } else {
                        return Err(GPUError::KernelUninitialized);
                    }
                }
            }
        }
    };
}

locked_kernel!(LockedFFTKernel, FFTKernel, create_fft_kernel, "FFT");
