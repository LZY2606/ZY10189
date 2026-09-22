//! Model-checks the generation claim protocol that `load_observed` and
//! `compare_exchange_observed` are built upon.
//!
//! A plain `cargo test` runs the model once with real atomics and threads. Running
//!
//! ```text
//! RUSTFLAGS="--cfg loom" cargo test --test loom_observed
//! ```
//!
//! explores the interleavings systematically with loom.
//!
//! The model mirrors the protocol
//! used by `ArcSwapAny`: writers claim the generation (even -> odd) before replacing the
//! pointer and release it (odd -> even + 2) right after; a conditional update commits by
//! compare-exchanging the generation and may then rely on the pointer not having moved
//! since the matching observation.

// The `loom` cfg is set through RUSTFLAGS when model-checking; it is not a cargo feature.
#![allow(unexpected_cfgs)]

#[cfg(loom)]
use loom::sync::atomic::{AtomicUsize, Ordering};
#[cfg(loom)]
use loom::sync::Arc;
#[cfg(loom)]
use loom::thread;

#[cfg(not(loom))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(not(loom))]
use std::sync::Arc;
#[cfg(not(loom))]
use std::thread;

/// Spinning is fine on real hardware, but unbounded retries explode the state space
/// loom has to explore, so the model gives up after a few attempts (a give-up is
/// indistinguishable from "the operation was scheduled a bit later" for our purposes).
#[cfg(loom)]
const MAX_SPINS: usize = 3;
#[cfg(not(loom))]
const MAX_SPINS: usize = 1_000_000;

/// A minimal model of the pointer + generation pair inside `ArcSwapAny`.
struct Model {
    gen: AtomicUsize,
    ptr: AtomicUsize,
}

impl Model {
    /// Mirrors `ArcSwapAny::swap`: claim, replace, release.
    fn write(&self, value: usize) {
        let mut spins = 0;
        let even = loop {
            let gen = self.gen.load(Ordering::SeqCst);
            if gen % 2 == 0
                && self
                    .gen
                    .compare_exchange_weak(gen, gen + 1, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            {
                break gen;
            }
            spins += 1;
            if spins >= MAX_SPINS {
                // Couldn't claim right now; on real hardware we'd just spin a bit longer.
                return;
            }
            #[cfg(loom)]
            loom::thread::yield_now();
        };
        self.ptr.store(value, Ordering::SeqCst);
        self.gen.store(even + 2, Ordering::SeqCst);
    }

    /// Mirrors `ArcSwapAny::load_observed`: a seqlock-style consistent snapshot.
    fn observe(&self) -> Option<(usize, usize)> {
        for _ in 0..MAX_SPINS {
            let before = self.gen.load(Ordering::SeqCst);
            if before % 2 != 0 {
                #[cfg(loom)]
                loom::thread::yield_now();
                continue;
            }
            let ptr = self.ptr.load(Ordering::SeqCst);
            let after = self.gen.load(Ordering::SeqCst);
            if before == after {
                return Some((before, ptr));
            }
        }
        None
    }

    /// Mirrors `ArcSwapAny::compare_exchange_observed`. Returns `true` if it committed.
    fn commit(&self, expected: (usize, usize), value: usize) -> bool {
        let (gen, ptr) = expected;
        if self
            .gen
            .compare_exchange(gen, gen + 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return false;
        }
        // The core invariant the whole design rests on: once the generation matched and
        // is claimed by us, the pointer must still be the observed one ‒ no A-B-A in
        // between can go unnoticed.
        assert_eq!(
            ptr,
            self.ptr.load(Ordering::SeqCst),
            "generation matched but the pointer moved underneath us"
        );
        self.ptr.store(value, Ordering::SeqCst);
        self.gen.store(gen + 2, Ordering::SeqCst);
        true
    }
}

fn model_body() {
    let model = Arc::new(Model {
        gen: AtomicUsize::new(0),
        ptr: AtomicUsize::new(0),
    });

    let mut handles = Vec::new();

    // A plain writer, replacing the pointer (possibly several times, eg. A -> B -> A).
    handles.push({
        let model = Arc::clone(&model);
        thread::spawn(move || {
            model.write(1);
            model.write(0);
        })
    });

    // Two conditional updaters, racing each other and the writer.
    for value in [10, 20] {
        let model = Arc::clone(&model);
        handles.push(thread::spawn(move || {
            if let Some(observation) = model.observe() {
                let _ = model.commit(observation, value);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Whatever happened, no claim is left stuck (the generation is even again).
    assert_eq!(0, model.gen.load(Ordering::SeqCst) % 2);
}

#[cfg(loom)]
#[test]
fn gen_protocol_loom() {
    // Keep the explored state space tractable; the interesting races (writer vs.
    // committer, committer vs. committer) all happen within a couple of preemptions.
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(model_body);
}

#[cfg(not(loom))]
#[test]
fn gen_protocol_std() {
    model_body();
}
