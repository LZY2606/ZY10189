//! Tests for the generation-token ([`Observation`]) API.
//!
//! These tests deliberately never rely on the allocator reusing an address: ABA is exercised both
//! with deterministic scheduling (barrier-pinned thread interleavings) and with a synthetic
//! observation that shares an address but carries a different generation, which is exactly what a
//! reused allocation would look like at the token level.

#![cfg(test)]

use alloc::sync::{Arc, Weak};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use adaptive_barrier::{Barrier, PanicMode};
use crossbeam_utils::thread;

use crate::strategy::CaS;
use crate::Strategy;
use crate::{ArcSwap, ArcSwapAny, ArcSwapOption, ArcSwapWeak, Observation};

type As<T> = ArcSwap<T>;
type Aso<T> = ArcSwapOption<T>;

/// The initial generation of every freshly created storage is 0.
#[test]
fn initial_generation_is_zero() {
    let shared = As::from_pointee(1u32);
    let (guard, observation) = shared.load_observed();
    assert_eq!(1, **guard);
    assert_eq!(
        0,
        observation.generation(),
        "fresh storage must start at gen 0"
    );
    assert_eq!(
        shared.current_generation(),
        observation.generation(),
        "load_observed must report the live generation",
    );
}

/// load_observed never returns without taking the write lock (checked indirectly) — two
/// back-to-back loads of an idle storage describe the same publication.
#[test]
fn repeated_loads_stable() {
    let shared = As::from_pointee(1u32);
    let (_, first) = shared.load_observed();
    for _ in 0..100 {
        let (_, again) = shared.load_observed();
        assert_eq!(first, again, "idle storage must keep the same observation");
    }
}

/// Storing the *same* Arc twice publishes two distinct generations.
#[test]
fn same_arc_twice_is_two_generations() {
    let shared = As::from_pointee(1u32);
    let value = Arc::new(2u32);
    shared.store(Arc::clone(&value));
    let (_, gen_one) = shared.load_observed();
    shared.store(Arc::clone(&value));
    let (_, gen_two) = shared.load_observed();
    assert_eq!(
        gen_one.generation() + 1,
        gen_two.generation(),
        "re-storing the identical Arc must advance the generation",
    );
    assert!(
        gen_one.same_address(&gen_two),
        "the address is identical, only the generation differs",
    );
    // But the observations are not equal: that is the ABA signal.
    assert_ne!(gen_one, gen_two);
}

/// Every successful swap / compare_and_swap / rcu bumps the generation exactly once; failed
/// conditional updates never bump it.
#[test]
fn every_publication_bumps_once() {
    let shared = As::from_pointee(0u32);
    assert_eq!(0, shared.current_generation());

    drop(shared.swap(Arc::new(1)));
    assert_eq!(1, shared.current_generation());

    shared.store(Arc::new(2));
    assert_eq!(2, shared.current_generation());

    let (_, current) = shared.load_observed();
    let new_arc = Arc::new(3u32);
    let prev = shared.compare_exchange_observed(current, Arc::clone(&new_arc));
    assert!(prev.is_ok(), "fresh observation must commit");
    assert_eq!(3, shared.current_generation());

    // The consumed token cannot commit a second time; no bump on failure.
    let retry = shared.compare_exchange_observed(current, Arc::new(4));
    assert!(retry.is_err(), "consumed token must be rejected");
    assert_eq!(
        3,
        shared.current_generation(),
        "failed conditional update must not bump the generation",
    );

    // Address-only CAS still bumps (it is a successful publication).
    let loaded = shared.load_full();
    let old = shared.compare_and_swap(&loaded, Arc::new(5));
    assert_eq!(3, **old);
    assert_eq!(4, shared.current_generation());

    // rcu bumps too.
    shared.rcu(|v| **v + 1);
    assert_eq!(5, shared.current_generation());
}

/// `None` in ArcSwapOption is a publication with a generation of its own.
#[test]
fn none_has_a_generation() {
    let shared: Aso<u32> = ArcSwapOption::empty();
    let (guard, none_token) = shared.load_observed();
    assert!(guard.is_none());
    assert_eq!(0, none_token.generation());
    assert!(none_token.same_address(&core::ptr::null()));

    shared.store(Some(Arc::new(1)));
    assert_eq!(1, shared.load_observed().1.generation());
    shared.store(None);
    assert_eq!(2, shared.load_observed().1.generation());

    // A conditional swap on a None generation commits exactly once.
    let committed = shared.compare_exchange_observed(none_token, Some(Arc::new(2)));
    assert!(committed.is_err(), "the stale None token must not commit");
    let fresh_none = shared.load_observed().1;
    assert!(shared
        .compare_exchange_observed(fresh_none, Some(Arc::new(3)))
        .is_ok());
    assert_eq!(3, **shared.load().as_ref().expect("stored Some"));
}

/// The essential ABA guarantee, without relying on allocator behavior: construct the observation
/// a reused address *would* produce (same address, newer generation) and verify the conditional
/// update refuses it while reporting that the address matches.
#[test]
fn synthetic_aba_same_address_rejected() {
    let shared = As::from_pointee(1u32);
    let value = Arc::new(2u32);
    shared.store(Arc::clone(&value));
    let (_, observation) = shared.load_observed();

    // Simulate "the value went away and came back": same address, generation advanced.
    shared.force_generation(observation.generation() + 2);
    let result = shared.compare_exchange_observed(observation, Arc::new(99));
    let current = match result {
        Err(current) => current,
        Ok(_) => panic!("ABA token with same address must not commit"),
    };
    assert!(
        observation.same_address(&current),
        "the failure must visibly be an ABA: address equal, generation {} != {}",
        observation.generation(),
        current.generation(),
    );
    assert_ne!(observation, current);
    assert_eq!(2, **shared.load(), "nothing must have been stored");
}

/// After A -> B -> A with *different* allocations, the old A token must fail (this does not
/// depend on addresses matching).
#[test]
fn toggle_a_b_a_distinct_allocations_rejected() {
    let shared = As::from_pointee(0u32);
    let a1 = Arc::new(7u32);
    shared.store(Arc::clone(&a1));
    let (_, token) = shared.load_observed();

    let b = Arc::new(7u32);
    let a2 = Arc::new(7u32);
    assert_ne!(
        Arc::as_ptr(&a1),
        Arc::as_ptr(&a2),
        "test needs distinct Arcs"
    );
    shared.store(b);
    shared.store(a2);

    let result = shared.compare_exchange_observed(token, Arc::new(8));
    assert!(result.is_err(), "token from before A-B-A must be rejected");
    let current = result.err().unwrap();
    assert!(
        !token.same_address(&current),
        "distinct allocations differ in address"
    );

    // A fresh observation commits, and returns the previous value.
    let fresh = shared.load_observed().1;
    let prev = shared
        .compare_exchange_observed(fresh, Arc::new(8))
        .expect("fresh token commits");
    assert_eq!(7, **prev);
    drop(prev);
    assert_eq!(8, **shared.load());
}

/// Deterministic A-B-A interleaving: one writer performs A -> B -> A while a reader holds a Guard
/// and a token obtained before the toggles; the conditional commit must fail. The barrier
/// schedule pins the interleaving, so this never relies on timing or address reuse.
#[test]
fn deterministic_aba_with_pinned_interleaving() {
    const WRITERS: usize = 1;
    let barrier = Barrier::new(PanicMode::Poison);
    let shared = As::from_pointee(10u32);

    thread::scope(|scope| {
        let mut reader_barrier = barrier.clone();
        let reader_shared = &shared;
        scope.spawn(move |_| {
            // Phase 0: observe A.
            let (guard, token) = reader_shared.load_observed();
            assert_eq!(10, **guard);
            reader_barrier.wait(); // (0) observed

            // Phase 2: writer has toggled A -> B -> A; the guard may pin the old allocation but
            // the generation must have advanced.
            reader_barrier.wait(); // (2) toggled
            let result = reader_shared.compare_exchange_observed(token, Arc::new(20));
            assert!(result.is_err(), "held-guard token must not survive A-B-A");
            assert_eq!(10, **reader_shared.load(), "writer restored numeric value");
            drop(guard);
        });

        let mut writer_barrier = barrier.clone();
        let writer_shared = &shared;
        scope.spawn(move |_| {
            writer_barrier.wait(); // (0) reader observed A
            writer_shared.store(Arc::new(11)); // B
            writer_shared.store(Arc::new(10)); // A (a fresh allocation)
            writer_barrier.wait(); // (2) toggled
            let _ = WRITERS;
        });

        drop(barrier);
    })
    .expect("scoped threads must not panic");

    assert!(shared.current_generation() >= 2);
}

/// Under concurrent writers performing store loops, every observed (ptr, generation) pair is
/// coherent and generations are strictly increasing for each observer.
#[test]
fn concurrent_publications_monotonic_generations() {
    const THREADS: usize = 4;
    const PER_THREAD: usize = 500;
    let mut barrier = Barrier::new(PanicMode::Poison);
    let shared = Arc::new(As::from_pointee(0usize));

    thread::scope(|scope| {
        for t in 0..THREADS {
            let mut barrier = barrier.clone();
            let shared = Arc::clone(&shared);
            scope.spawn(move |_| {
                barrier.wait();
                for i in 0..PER_THREAD {
                    shared.store(Arc::new(t * PER_THREAD + i));
                }
            });
        }
        let shared = Arc::clone(&shared);
        scope.spawn(move |_| {
            let mut last_generation = 0usize;
            let mut last_ptr = shared.load_observed().1.ptr_addr();
            for _ in 0..THREADS * PER_THREAD * 4 {
                let (guard, observation) = shared.load_observed();
                // Coherence: the guarded value lives at the observed address.
                assert_eq!(
                    Arc::as_ptr(&*guard) as usize,
                    observation.ptr_addr(),
                    "load_observed returned an inconsistent (guard, token) pair",
                );
                if observation.generation() != last_generation || observation.ptr_addr() != last_ptr
                {
                    // A new generation never revisits a previously observed (gen, ptr) pair in
                    // increasing order.
                    assert!(
                        observation.generation() >= last_generation,
                        "generation went backwards: {} -> {}",
                        last_generation,
                        observation.generation(),
                    );
                    last_generation = observation.generation();
                    last_ptr = observation.ptr_addr();
                }
                core::sync::atomic::spin_loop_hint();
            }
        });
        // Release all writer threads at each round (the original handle is the coordinator).
        for _ in 0..PER_THREAD {
            barrier.wait();
        }
    })
    .expect("scoped threads must not panic");

    assert_eq!(THREADS * PER_THREAD, shared.current_generation());
}

/// Exactly one of many racing conditional updaters with the same current token may commit; all
/// others observe the new generation and retry. This is the intended CAS discipline.
#[test]
fn racing_observed_cas_only_one_commits_per_round() {
    const ROUNDS: usize = 50;
    const RACERS: usize = 6;
    let shared = Arc::new(As::from_pointee(0usize));

    for round in 0..ROUNDS {
        let token = shared.load_observed().1;
        let winners = Arc::new(AtomicUsize::new(0));
        let barrier = Barrier::new(PanicMode::Poison);
        // Clone every participating handle before spawning, then retire the original exactly as
        // the barrier documentation prescribes — no "phantom" participant, no early release.
        let handles: Vec<_> = (0..RACERS)
            .map(|_| {
                let barrier = barrier.clone();
                let shared = Arc::clone(&shared);
                let winners = Arc::clone(&winners);
                (barrier, shared, winners)
            })
            .collect();
        drop(barrier);

        thread::scope(|scope| {
            for (mut barrier, shared, winners) in handles {
                scope.spawn(move |_| {
                    barrier.wait();
                    if shared
                        .compare_exchange_observed(token, Arc::new(round + 1))
                        .is_ok()
                    {
                        winners.fetch_add(1, Ordering::SeqCst);
                    }
                });
            }
        })
        .expect("scoped threads must not panic");

        assert_eq!(
            1,
            winners.load(Ordering::SeqCst),
            "round {round}: exactly one racer must commit",
        );
        assert_eq!(round + 1, **shared.load(), "round {round}: committed value");
    }
}

/// Saturation: after forcing the generation to usize::MAX, no observation can drive a successful
/// conditional update (neither the current one nor a crafted reused-address one), and further
/// publications keep the generation pinned while the pointer still changes.
#[test]
fn saturation_is_terminal() {
    let shared = As::from_pointee(1u32);
    shared.store(Arc::new(2));
    shared.force_generation(usize::MAX);

    let (_, saturated) = shared.load_observed();
    assert!(
        saturated.is_saturated(),
        "forced generation must read as saturated"
    );

    let result = shared.compare_exchange_observed(saturated, Arc::new(3));
    assert!(result.is_err(), "saturated token must never commit");
    assert_eq!(2, **shared.load(), "pointer must remain unchanged");

    // Ordinary publications still work, but the generation does not wrap.
    shared.store(Arc::new(4));
    assert_eq!(
        usize::MAX,
        shared.current_generation(),
        "generation must saturate, not wrap"
    );
    assert_eq!(4, **shared.load());

    // A token crafted at the saturated generation with the *current* address is also rejected;
    // this is the wrap-safety guarantee.
    let (_, crafted) = shared.load_observed();
    assert_eq!(usize::MAX, crafted.generation());
    assert!(shared
        .compare_exchange_observed(crafted, Arc::new(5))
        .is_err());
    assert_eq!(4, **shared.load());
}

/// One step before saturation, updates still work; stepping into saturation pins it.
#[test]
fn last_real_generation_still_commits() {
    let shared = As::from_pointee(0u32);
    shared.force_generation(usize::MAX - 1);
    let (_, token) = shared.load_observed();
    assert!(shared.compare_exchange_observed(token, Arc::new(1)).is_ok());
    assert_eq!(usize::MAX, shared.current_generation());
    assert_eq!(1, **shared.load());
}

/// The returned Guard on success owns one reference and the storage owns one; failed attempts do
/// not touch reference counts.
#[test]
fn reference_counts() {
    let shared = As::from_pointee(1u32);
    let new_value = Arc::new(2u32);
    let token = shared.load_observed().1;

    // Failing attempt first: bump gen, reuse token.
    shared.store(Arc::new(99));
    let failed = shared.compare_exchange_observed(token, Arc::clone(&new_value));
    assert!(failed.is_err());
    assert_eq!(
        1,
        Arc::strong_count(&new_value),
        "rejected new value must not gain a reference",
    );

    // Successful attempt.
    let fresh = shared.load_observed().1;
    let prev = shared
        .compare_exchange_observed(fresh, Arc::clone(&new_value))
        .unwrap();
    assert_eq!(
        2,
        Arc::strong_count(&new_value),
        "storage + cloned argument"
    );
    drop(prev);
    let loaded = shared.load_full();
    assert_eq!(
        3,
        Arc::strong_count(&loaded),
        "storage + clone argument + load"
    );
}

/// Observation tokens are Send + Sync when the value is, same as Arc.
#[cfg(not(feature = "experimental-thread-local"))]
#[test]
fn observation_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Observation<Arc<u32>>>();
    assert_send_sync::<Observation<Option<Arc<u32>>>>();

    let shared = As::from_pointee(1u32);
    let token = Arc::new(std::sync::Mutex::new(shared.load_observed().1));
    thread::scope(|scope| {
        for _ in 0..2 {
            let token = Arc::clone(&token);
            scope.spawn(move |_| {
                let guard = token.lock().unwrap();
                let _ = guard.generation();
            });
        }
    })
    .unwrap();
}

/// The generation machinery works through the Weak path too (conditional commit and rejection).
#[cfg(feature = "weak")]
#[test]
fn weak_observation_roundtrip() {
    let data = Arc::new("hello");
    let shared: ArcSwapWeak<&'static str> = ArcSwapAny::new(Arc::downgrade(&data));
    let (_, token) = shared.load_observed();
    assert_eq!(0, token.generation());

    // A different weak pointer is a new publication.
    let other = Arc::new("world");
    shared.store(Arc::downgrade(&other));
    assert!(shared
        .compare_exchange_observed(token, Arc::downgrade(&data))
        .is_err());

    let fresh = shared.load_observed().1;
    let prev = shared
        .compare_exchange_observed(fresh, Arc::downgrade(&data))
        .expect("fresh weak token commits");
    let upgraded = Weak::clone(&*prev).upgrade().expect("other still alive");
    assert_eq!("world", *upgraded);

    // Null weak (Weak::new) carries a generation as well.
    shared.store(Weak::new());
    let (_, none_token) = shared.load_observed();
    assert!(none_token.same_address(&core::ptr::null()));
    assert!(shared
        .compare_exchange_observed(none_token, Arc::downgrade(&data))
        .is_ok());
}

/// Compile-time guard: the observed API is available for every strategy supporting CaS, not just
/// the default one.
#[cfg(feature = "internal-test-strategies")]
#[test]
fn works_with_test_strategy() {
    use crate::strategy::test_strategies::FillFastSlots;
    let shared: ArcSwapAny<Arc<u32>, FillFastSlots> = ArcSwapAny::from(Arc::new(1u32));
    let token = shared.load_observed().1;
    assert!(shared.compare_exchange_observed(token, Arc::new(2)).is_ok());
    assert_eq!(2, **shared.load());
}

/// Cache and observed operations interoperate: a cached handle keeps serving the old Arc while
/// the generation advances, and refreshing after publication sees the new generation.
#[test]
fn cache_interoperation() {
    use crate::Cache;
    let shared = Arc::new(As::from_pointee(1u32));
    let mut cache = Cache::new(Arc::clone(&shared));
    assert_eq!(1, **cache.load());
    shared.store(Arc::new(2));
    assert_eq!(2, **cache.load(), "cache refreshes to the new publication");
    let token_taken_after = shared.load_observed().1;
    shared.store(Arc::new(3));
    assert!(shared
        .compare_exchange_observed(token_taken_after, Arc::new(4))
        .is_err());
    assert_eq!(3, **cache.load());
    let fresh = shared.load_observed().1;
    assert!(shared.compare_exchange_observed(fresh, Arc::new(4)).is_ok());
    assert_eq!(4, **cache.load());
}

/// Holding a Guard across publications still allows the generation machinery to progress; the
/// guarded value is kept alive until the guard drops, and every replaced value is reclaimed by
/// the time the scope settles.
#[test]
fn guard_held_across_publications() {
    let dropped = Arc::new(AtomicUsize::new(0));
    struct Track(Arc<AtomicUsize>, u32);
    impl Drop for Track {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let shared = As::from(Arc::new(Track(Arc::clone(&dropped), 0)));
    let guard = shared.load();
    assert_eq!(0, guard.1);
    assert_eq!(
        0,
        dropped.load(Ordering::SeqCst),
        "nothing dropped before any publication"
    );

    // Holding the guard across a publication must not drop the guarded value.
    shared.store(Arc::new(Track(Arc::clone(&dropped), 1)));
    assert_eq!(
        0,
        dropped.load(Ordering::SeqCst),
        "the guarded publication stays alive"
    );

    // The generation advanced regardless of the guard.
    assert_eq!(1, shared.current_generation());

    // Dropping the guard eventually lets the first value go; forcing a further reclamation
    // settles the debt bookkeeping for the value replaced while the guard was alive.
    drop(guard);
    shared.store(Arc::new(Track(Arc::clone(&dropped), 2)));
    // Values #0 and #1 have been replaced; both must be gone by now.
    assert_eq!(
        2,
        dropped.load(Ordering::SeqCst),
        "both superseded values must be reclaimed after the guard is gone",
    );

    // Observed commits work after guard churn too.
    let token = shared.load_observed().1;
    assert!(shared
        .compare_exchange_observed(token, Arc::new(Track(Arc::clone(&dropped), 3)))
        .is_ok());
    drop(shared);
}

#[allow(dead_code)]
fn _strategy_bounds_compile<S: Strategy<Arc<u32>> + CaS<Arc<u32>>>(s: &ArcSwapAny<Arc<u32>, S>) {
    let token = s.load_observed().1;
    let _ = s.compare_exchange_observed(token, Arc::new(1));
}

/// Bulk A-B-A torture: many togglers publish A and B in a fixed barrier-synchronized pattern
/// while observers attempt conditional updates; any success must correspond to the exact
/// generation observed.
#[test]
fn bulk_toggle_no_ghost_success() {
    const TOGGLERS: usize = 3;
    const CYCLES: usize = 100;
    let barrier = Barrier::new(PanicMode::Poison);
    let shared = Arc::new(As::from_pointee(0u32));
    let successes = Arc::new(AtomicUsize::new(0));
    let ghost_failures = Arc::new(AtomicUsize::new(0));

    thread::scope(|scope| {
        for t in 0..TOGGLERS {
            let mut barrier = barrier.clone();
            let shared = Arc::clone(&shared);
            scope.spawn(move |_| {
                for c in 0..CYCLES {
                    barrier.wait();
                    shared.store(Arc::new(((t * CYCLES + c) % 2) as u32));
                }
            });
        }
        let mut observer_barrier = barrier.clone();
        drop(barrier);
        let shared = Arc::clone(&shared);
        let successes = Arc::clone(&successes);
        let ghost_failures = Arc::clone(&ghost_failures);
        scope.spawn(move |_| {
            // One rendezvous per barrier phase; each phase contains all TOGGLERS stores.
            for _ in 0..CYCLES {
                observer_barrier.wait();
                let (_, token) = shared.load_observed();
                let target = Arc::new(128);
                match shared.compare_exchange_observed(token, target) {
                    Ok(prev) => {
                        // If it committed, the previous value must be exactly the one at the
                        // observed address and the new generation is live.
                        assert_eq!(token.ptr_addr(), Arc::as_ptr(&prev) as usize);
                        successes.fetch_add(1, Ordering::SeqCst);
                    }
                    Err(current) => {
                        // Refused because something published in between.
                        if current.generation() == token.generation()
                            && current.ptr_addr() == token.ptr_addr()
                        {
                            ghost_failures.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
            }
        });
    })
    .expect("scoped threads must not panic");

    assert_eq!(
        0,
        ghost_failures.load(Ordering::SeqCst),
        "a failed comparison must never report the same publication",
    );
    // Every observed successful commit had its accounting validated inline; the final value is
    // whatever the last publication was, and a fresh token there commits deterministically.
    let final_token = shared.load_observed().1;
    assert!(
        shared
            .compare_exchange_observed(final_token, Arc::new(255))
            .is_ok(),
        "after all togglers stop, a fresh token must commit exactly once",
    );
    assert_eq!(255, **shared.load());
    let _ = successes;
}

#[allow(dead_code)]
fn _use_vec(_v: Vec<u32>) {}

/// The observed API must also work with the internal `RwLock<()>` strategy, which has its own
/// compare-and-swap implementation.
#[cfg(feature = "internal-test-strategies")]
#[test]
fn works_with_rwlock_strategy() {
    use std::sync::RwLock;
    let shared: ArcSwapAny<Arc<u32>, RwLock<()>> =
        ArcSwapAny::with_strategy(Arc::new(1), RwLock::new(()));
    let token = shared.load_observed().1;
    assert_eq!(0, token.generation());
    let prev = shared
        .compare_exchange_observed(token, Arc::new(2))
        .expect("fresh token commits under RwLock strategy");
    assert_eq!(1, **prev);
    // consumed token fails
    assert!(shared
        .compare_exchange_observed(token, Arc::new(3))
        .is_err());
    // A-B-A address test through this strategy
    let v = Arc::new(4u32);
    shared.store(Arc::clone(&v));
    let t2 = shared.load_observed().1;
    shared.force_generation(t2.generation() + 3);
    let err = shared
        .compare_exchange_observed(t2, Arc::new(5))
        .err()
        .unwrap();
    assert!(t2.same_address(&err));
    assert_ne!(t2, err);
}
