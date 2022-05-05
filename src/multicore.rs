extern crate crossbeam;
extern crate futures;
extern crate num_cpus;

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crossbeam::thread::Scope;
use futures::channel::oneshot::{channel, Receiver};
use futures::executor::{block_on, ThreadPool};
use futures::future::lazy;

#[derive(Clone)]
pub struct Workers {
    pub(crate) cpus: usize,
    pool: ThreadPool,
}

impl Workers {
    pub(crate) fn new() -> Workers {
        let cpus = num_cpus::get_physical();
        Workers {
            cpus,
            pool: ThreadPool::builder().pool_size(cpus).create().expect("should create a thread pool for futures execution"),
        }
    }

    pub fn log_num_cpus(&self) -> u32 {
        log2_floor(self.cpus)
    }

    pub fn compute<F, T, E>(
        &self, f: F,
    ) -> Waiter<T, E>
        where F: FnOnce() -> Result<T, E> + Send + 'static,
              T: Send + 'static,
              E: Send + 'static
    {
        let (sender, receiver) = channel();
        let lazy_future = lazy(move |_| {
            let res = f();

            if !sender.is_canceled() {
                let _ = sender.send(res);
            }
        });

        let waiter = Waiter {
            receiver
        };

        self.pool.spawn_ok(lazy_future);

        waiter
    }

    pub fn scope<'a, F, R>(
        &self,
        elements: usize,
        f: F,
    ) -> R
        where F: FnOnce(&Scope<'a>, usize) -> R
    {
        let chunk_size = self.get_chunk_size(elements);

        crossbeam::scope(|scope| {
            f(scope, chunk_size)
        }).expect("underlying must run. ")
    }

    pub fn get_chunk_size(
        &self,
        elements: usize,
    ) -> usize {
        let chunk_size = if elements <= self.cpus {
            1
        } else {
            Self::chunk_size_for_num_spawned_threads(elements, self.cpus)
        };

        chunk_size
    }

    pub fn chunk_size_for_num_spawned_threads(elements: usize, num_threads: usize) -> usize {
        assert!(elements >= num_threads, "received {} elements to spawn {} threads", elements, num_threads);
        if elements % num_threads == 0 {
            elements / num_threads
        } else {
            elements / num_threads + 1
        }
    }
}


pub struct Waiter<T, E> {
    receiver: Receiver<Result<T, E>>,
}

impl<T: Send + 'static, E: Send + 'static> Future for Waiter<T, E> {
    type Output = Result<T, E>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output>
    {
        let rec = unsafe { self.map_unchecked_mut(|s| &mut s.receiver) };
        match rec.poll(cx) {
            Poll::Ready(v) => {
                if let Ok(v) = v {
                    return Poll::Ready(v);
                } else {
                    panic!("Worker future can not have canceled sender");
                }
            }
            Poll::Pending => {
                return Poll::Pending;
            }
        }
    }
}

impl<T: Send + 'static, E: Send + 'static> Waiter<T, E> {
    pub fn wait(self) -> <Self as Future>::Output {
        block_on(self)
    }
}

fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

#[test]
fn test_log2_floor() {
    assert_eq!(log2_floor(1), 0);
    assert_eq!(log2_floor(2), 1);
    assert_eq!(log2_floor(3), 1);
    assert_eq!(log2_floor(4), 2);
    assert_eq!(log2_floor(5), 2);
    assert_eq!(log2_floor(6), 2);
    assert_eq!(log2_floor(7), 2);
    assert_eq!(log2_floor(8), 3);
}
