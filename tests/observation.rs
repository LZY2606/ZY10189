//! Integration tests for the generation-token observation API.
//!
//! These exercise only the public surface and use barrier-pinned thread schedules (see
//! [`SpinBarrier`]) so that interleavings are deterministic: no sleeps, no clock, no reliance on
//! allocator behaviour and no external resources.

#![cfg(not(feature = "experimental-thread-local"))]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arc_swap::{ArcSwap, ArcSwapOption};

/// A deterministic one-shot phase rendezvous: workers count up on `arrived`, the orchestrator
/// releases all of them with one store on `release`.
#[derive(Clone)]
struct Phase {
    inner: Arc<PhaseInner>,
}

struct PhaseInner {
    arrived: AtomicUsize,
    release: AtomicUsize,
}

impl Phase {
    fn new() -> Self {
        Self {
            inner: Arc::new(PhaseInner {
                arrived: AtomicUsize::new(0),
                release: AtomicUsize::new(0),
            }),
        }
    }

    fn arrive_and_wait(&self) {
        self.inner.arrived.fetch_add(1, Ordering::AcqRel);
        while self.inner.release.load(Ordering::Acquire) == 0 {
            std::hint::spin_loop();
        }
    }

    fn wait_all_then_release(&self, parties: usize) {
        while self.inner.arrived.load(Ordering::Acquire) < parties {
            std::hint::spin_loop();
        }
        self.inner.release.store(1, Ordering::Release);
    }
}

/// Fixed schedule of named phases, shareable by cloning.
#[derive(Clone)]
struct Schedule {
    phases: Arc<Vec<Phase>>,
}

impl Schedule {
    fn new(count: usize) -> Self {
        Self {
            phases: Arc::new((0..count).map(|_| Phase::new()).collect()),
        }
    }
    fn sync(&self, phase: usize) {
        self.phases[phase].arrive_and_wait();
    }
    fn release(&self, phase: usize, parties: usize) {
        self.phases[phase].wait_all_then_release(parties);
    }
}

/// A reader's token taken before an A-B-A toggle must never commit, even when the numeric value
/// (and, conceptually, the address) returns. The schedule pins every step.
#[test]
fn deterministic_aba_public_surface() {
    crossbeam_utils::thread::scope(|scope| {
        // 2 worker parties per phase (reader + writer).
        let schedule = Schedule::new(3);
        let shared = Arc::new(ArcSwap::from_pointee(1u32));

        let reader_schedule = schedule.clone();
        let reader_shared = Arc::clone(&shared);
        scope.spawn(move |_| {
            // Phase 0: capture the observation of A.
            let (guard, token) = reader_shared.load_observed();
            assert_eq!(1, **guard);
            reader_schedule.sync(0);

            // Phase 2: writer has performed A -> B -> A.
            reader_schedule.sync(2);
            let result = reader_shared.compare_exchange_observed(token, Arc::new(42));
            let current = result.expect_err("token from before A-B-A must not commit");
            assert_eq!(
                10,
                **reader_shared.load(),
                "writer restores the numeric value, not the generation",
            );
            assert_ne!(token, current);
            assert_eq!(
                token.generation() + 2,
                current.generation(),
                "two publications happened between capture and verification",
            );
            drop(guard);

            // A fresh token commits once.
            let fresh = reader_shared.load_observed().1;
            reader_shared
                .compare_exchange_observed(fresh, Arc::new(7))
                .expect("fresh token commits");
            assert_eq!(7, **reader_shared.load());
        });

        let writer_schedule = schedule.clone();
        let writer_shared = Arc::clone(&shared);
        scope.spawn(move |_| {
            writer_schedule.sync(0);
            // Phase 1/2 transitions, then let the reader proceed.
            writer_shared.store(Arc::new(2)); // B
            writer_shared.store(Arc::new(10)); // back to a "matching" value, fresh allocation
            writer_schedule.sync(2);
        });

        // Drive the schedule from the orchestrating thread.
        schedule.release(0, 2);
        // The workers only meet at phases 0 and 2 in this scenario.
        schedule.release(2, 2);
    })
    .unwrap();
}

/// Same token can't commit twice when many threads race; exactly one wins per round.
#[test]
fn exactly_one_winner_per_round() {
    const ROUNDS: usize = 25;
    const RACERS: usize = 8;
    let shared = Arc::new(ArcSwap::from_pointee(0usize));

    for round in 0..ROUNDS {
        let token = shared.load_observed().1;
        let winners = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(AtomicUsize::new(0));

        crossbeam_utils::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..RACERS {
                let shared = Arc::clone(&shared);
                let winners = Arc::clone(&winners);
                let start = Arc::clone(&start);
                handles.push(scope.spawn(move |_| {
                    while start.load(Ordering::Acquire) == 0 {
                        std::hint::spin_loop();
                    }
                    if shared
                        .compare_exchange_observed(token, Arc::new(round + 1))
                        .is_ok()
                    {
                        winners.fetch_add(1, Ordering::AcqRel);
                    }
                }));
            }
            // Deterministic fan-out: release all racers from one store.
            start.store(1, Ordering::Release);
            for h in handles {
                h.join().unwrap();
            }
        })
        .unwrap();

        assert_eq!(
            1,
            winners.load(Ordering::Acquire),
            "round {round}: exactly one racer must win",
        );
        assert_eq!(round + 1, **shared.load());
    }
}

/// `None` is a generation-bearing publication through the public API, and None <-> Some toggles
/// invalidate tokens each time.
#[test]
fn option_none_generations_public() {
    let shared: ArcSwapOption<u32> = ArcSwapOption::empty();
    let none_token = shared.load_observed().1;
    assert_eq!(0, none_token.generation());

    shared.store(Some(Arc::new(1)));
    shared.store(None);

    assert!(
        shared
            .compare_exchange_observed(none_token, Some(Arc::new(2)))
            .is_err(),
        "stale None token must not commit after None -> Some -> None",
    );
    let fresh_none = shared.load_observed().1;
    assert!(
        fresh_none.same_address(&std::ptr::null()),
        "None observation records the null address",
    );
    shared
        .compare_exchange_observed(fresh_none, Some(Arc::new(3)))
        .expect("current None token commits");
    assert_eq!(3, **shared.load().as_ref().unwrap());
}

/// Observations can cross threads and be compared by value (same address + same generation).
#[test]
fn observation_send_and_eq() {
    let shared = Arc::new(ArcSwap::from_pointee(1u32));
    let token = shared.load_observed().1;

    let handle = std::thread::spawn(move || {
        let also = token;
        assert_eq!(token, also);
        token
    });
    let returned = handle.join().unwrap();
    assert_eq!(returned, shared.load_observed().1);
}

/// A generation difference with an equal address is observable as "same address, newer
/// generation" — the documented way to classify an ABA refusal.
#[test]
fn refusal_classifies_aba() {
    let shared = ArcSwap::from_pointee(1u32);
    let value = Arc::new(2u32);
    shared.store(Arc::clone(&value));
    let token = shared.load_observed().1;

    // Advance only the generation while keeping the same address, the token-level shape of an
    // address-reuse ABA (the test never depends on a real reuse).
    shared.compare_and_swap(&value, Arc::clone(&value));
    let err = shared
        .compare_exchange_observed(token, Arc::new(3))
        .expect_err("newer generation must refuse");
    assert!(
        token.same_address(&err),
        "ABA: address matches, {} -> {}",
        token.generation(),
        err.generation(),
    );
    assert!(err.generation() > token.generation());
    assert_eq!(2, **shared.load());
}
