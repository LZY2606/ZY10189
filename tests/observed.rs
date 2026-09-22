//! Tests for the generation-token API (`load_observed` / `compare_exchange_observed`).
//!
//! The multi-threaded tests in here use barriers to force the exact interleavings they
//! exercise, so they are deterministic: they don't depend on the allocator reusing
//! addresses, on timing, or on the filesystem.

use std::sync::Arc;

use adaptive_barrier::{Barrier, PanicMode};
use crossbeam_utils::thread;

use arc_swap::{ArcSwap, ArcSwapOption};

/// Basic success/failure behaviour of the conditional update.
#[test]
fn observed_commit_basic() {
    let shared = ArcSwap::from_pointee(0);
    let (guard, obs) = shared.load_observed();
    assert_eq!(0, **guard);

    // A successful commit returns the observation of the new generation, ready for chaining.
    let obs2 = shared
        .compare_exchange_observed(&obs, Arc::new(1))
        .expect("commit of a current observation should succeed");
    assert_ne!(
        obs, obs2,
        "a successful commit must produce a new generation"
    );
    assert!(obs.generation() < obs2.generation());
    assert_eq!(1, **shared.load());

    // The old token is stale now and the commit fails without touching the value.
    let err = shared
        .compare_exchange_observed(&obs, Arc::new(2))
        .expect_err("commit with a stale token must fail");
    assert_eq!(
        1,
        ***err.current(),
        "failed commit must not change the value"
    );

    // The failure carries the *current* observation, matching a fresh load.
    let (_, fresh) = shared.load_observed();
    assert_eq!(
        fresh,
        err.observation(),
        "a failed commit must return the current observation"
    );

    // And it hands the `new` value back to the caller.
    let (_guard, _obs, new) = err.into_parts();
    assert_eq!(2, *new, "the failed commit must return the new value");

    // Chaining: the observation returned by the successful commit is current.
    shared
        .compare_exchange_observed(&obs2, Arc::new(3))
        .expect("chained commit with the returned observation should succeed");
    assert_eq!(3, **shared.load());
}

/// Storing the very same `Arc` twice in a row still produces a new generation each time.
#[test]
fn double_store_same_arc_new_generation() {
    let a = Arc::new(42);
    let shared = ArcSwap::from(Arc::clone(&a));
    let (_, obs0) = shared.load_observed();

    shared.store(Arc::clone(&a));
    let (_, obs1) = shared.load_observed();
    assert_ne!(
        obs0, obs1,
        "storing the same Arc must still advance the generation"
    );

    shared.store(Arc::clone(&a));
    let (_, obs2) = shared.load_observed();
    assert_ne!(obs1, obs2, "every store advances the generation");
    assert!(obs0.generation() < obs1.generation());
    assert!(obs1.generation() < obs2.generation());

    // The pointer never changed, yet the token from before the stores is stale.
    shared
        .compare_exchange_observed(&obs0, Arc::new(0))
        .expect_err("token from before identical stores must not commit");
}

/// A writer switching the value A -> B -> A must invalidate tokens from before the switch,
/// even though the pointer ends up identical. The interleaving is forced by barriers, so
/// the test does not rely on the allocator reusing addresses.
#[test]
fn aba_switch_invalidates_token() {
    let a = Arc::new(String::from("A"));
    let b = Arc::new(String::from("B"));
    let shared = ArcSwap::from(Arc::clone(&a));
    let mut barrier = Barrier::new(PanicMode::Poison);

    thread::scope(|scope| {
        scope.spawn({
            let mut barrier = barrier.clone();
            let shared = &shared;
            let a = &a;
            let b = &b;
            move |_| {
                // Wait until the observer took its token...
                barrier.wait();
                // ...then switch A -> B -> A, storing the *same* Arc back.
                shared.store(Arc::clone(b));
                shared.store(Arc::clone(a));
                barrier.wait();
            }
        });

        let (guard, obs) = shared.load_observed();
        assert_eq!("A", guard.as_str());
        // Let the writer do the A -> B -> A dance.
        barrier.wait();
        barrier.wait();

        // The pointer is the very same Arc again...
        assert!(
            Arc::ptr_eq(&shared.load_full(), &a),
            "expected the original Arc to be back in place"
        );
        // ...but the generation token detects the switch.
        let err = shared
            .compare_exchange_observed(&obs, Arc::new(String::from("C")))
            .expect_err("token from before the A-B-A switch must not commit");
        assert_eq!("A", err.current().as_str());
        assert_eq!("C", err.into_parts().2.as_str());

        // For contrast, the pointer-based compare_and_swap cannot tell the difference and
        // succeeds. This documents why the token API exists.
        let prev = shared.compare_and_swap(&a, Arc::new(String::from("D")));
        assert!(
            Arc::ptr_eq(&prev, &a),
            "pointer comparison is fooled by the A-B-A switch"
        );
        assert_eq!("D", shared.load_full().as_str());
    })
    .unwrap();
}

/// Holding a `Guard` (a debt slot) across observed commits keeps working and keeps the old
/// value alive and readable.
#[test]
fn guard_held_across_commits() {
    let shared = ArcSwap::from_pointee(0);
    let (guard, obs) = shared.load_observed();
    let mut barrier = Barrier::new(PanicMode::Poison);

    thread::scope(|scope| {
        scope.spawn({
            let mut barrier = barrier.clone();
            let shared = &shared;
            move |_| {
                let mut obs = obs;
                barrier.wait();
                for i in 1..=10 {
                    obs = shared
                        .compare_exchange_observed(&obs, Arc::new(i))
                        .expect("chained commits should succeed");
                }
                barrier.wait();
            }
        });

        barrier.wait();
        barrier.wait();
        // The commits are done, but our guard still protects the original value.
        assert_eq!(0, **guard, "held guard must keep its value alive");
        assert_eq!(10, **shared.load());
    })
    .unwrap();
}

/// `None` in an `ArcSwapOption` has its own generations, just like any other value.
#[test]
fn none_has_generation() {
    let shared = ArcSwapOption::<usize>::empty();
    let (guard, obs_none) = shared.load_observed();
    assert!(guard.is_none());

    // Storing None again still advances the generation.
    shared.store(None);
    shared
        .compare_exchange_observed(&obs_none, Some(Arc::new(1)))
        .expect_err("token of the previous None generation must be stale");

    // A fresh observation of None commits fine.
    let (guard, obs_none2) = shared.load_observed();
    assert!(guard.is_none());
    let obs_some = shared
        .compare_exchange_observed(&obs_none2, Some(Arc::new(42)))
        .expect("committing over a current None observation should succeed");
    assert_eq!(42, **shared.load().as_ref().unwrap());

    // And back to None, chained on the returned observation.
    shared
        .compare_exchange_observed(&obs_some, None)
        .expect("chained commit back to None should succeed");
    assert!(shared.load().is_none());
}

/// Multiple threads doing read-modify-write through the observed API must not lose
/// updates.
#[test]
fn concurrent_observed_updates() {
    const THREADS: usize = 4;
    #[cfg(miri)]
    const ITERATIONS: usize = 5;
    #[cfg(not(miri))]
    const ITERATIONS: usize = 50;

    let shared = ArcSwap::from_pointee(0usize);
    thread::scope(|scope| {
        for _ in 0..THREADS {
            scope.spawn(|_| {
                for _ in 0..ITERATIONS {
                    let (mut guard, mut obs) = shared.load_observed();
                    loop {
                        let new = Arc::new(**guard + 1);
                        match shared.compare_exchange_observed(&obs, new) {
                            Ok(_) => break,
                            Err(err) => {
                                let (current, current_obs, _) = err.into_parts();
                                guard = current;
                                obs = current_obs;
                            }
                        }
                    }
                }
            });
        }
    })
    .unwrap();
    assert_eq!(
        THREADS * ITERATIONS,
        **shared.load(),
        "no update must be lost"
    );
}

/// Interaction with the classic API: successful writes invalidate tokens, while a failed
/// pointer-based `compare_and_swap` does not needlessly invalidate them.
#[test]
fn classic_api_interaction() {
    let shared = ArcSwap::from_pointee(0);
    let (_, obs) = shared.load_observed();

    // A failed compare_and_swap (wrong `current`) doesn't store anything...
    let not_current = Arc::new(999);
    let prev = shared.compare_and_swap(&not_current, Arc::new(1));
    assert_eq!(0, **prev);
    // ...so the token is still valid.
    let obs = shared
        .compare_exchange_observed(&obs, Arc::new(2))
        .expect("failed compare_and_swap must not invalidate tokens");

    // A successful compare_and_swap does invalidate.
    let current = shared.load();
    let prev = shared.compare_and_swap(&current, Arc::new(3));
    assert_eq!(2, **prev);
    drop(current);
    shared
        .compare_exchange_observed(&obs, Arc::new(4))
        .expect_err("successful compare_and_swap must invalidate tokens");
    assert_eq!(3, **shared.load());

    // And so does rcu (which is built on compare_and_swap).
    let (_, obs) = shared.load_observed();
    shared.rcu(|v| **v + 1);
    shared
        .compare_exchange_observed(&obs, Arc::new(5))
        .expect_err("rcu must invalidate tokens");
    assert_eq!(4, **shared.load());
}

/// The `Cache` handle (and the `Access` trait) keeps working and revalidating while
/// observed commits happen.
#[test]
fn cache_observes_commits() {
    use arc_swap::cache::Access;
    use arc_swap::Cache;

    let shared = Arc::new(ArcSwap::from_pointee(0));
    let mut cache = Cache::new(Arc::clone(&shared));
    assert_eq!(0, **cache.load());

    let (_, obs) = shared.load_observed();
    shared
        .compare_exchange_observed(&obs, Arc::new(1))
        .expect("commit should succeed");
    assert_eq!(
        1,
        **cache.load(),
        "cache must revalidate after an observed commit"
    );

    // The generic Access abstraction sees the same.
    fn read_via_access<A: Access<i32>>(access: &mut A) -> i32 {
        *access.load()
    }
    assert_eq!(1, read_via_access(&mut cache));
}

/// The observed API works on the `Weak` storage too.
#[cfg(feature = "weak")]
#[test]
fn weak_observed() {
    use std::sync::Weak;

    use arc_swap::ArcSwapAny;

    let data = Arc::new(String::from("hello"));
    let shared: ArcSwapAny<Weak<String>> = ArcSwapAny::new(Arc::downgrade(&data));
    let (guard, obs) = shared.load_observed();
    assert_eq!("hello", guard.upgrade().unwrap().as_str());

    let data2 = Arc::new(String::from("world"));
    shared
        .compare_exchange_observed(&obs, Arc::downgrade(&data2))
        .expect("commit on weak storage should succeed");
    assert_eq!("world", shared.load().upgrade().unwrap().as_str());

    shared
        .compare_exchange_observed(&obs, Weak::new())
        .expect_err("stale token on weak storage must fail");
}

/// The observed API keeps working on instances that went through serde (de)serialization;
/// the generation itself is not serialized, a deserialized instance starts a fresh
/// generation sequence.
#[cfg(feature = "serde")]
#[test]
fn serde_observed() {
    use serde_derive::{Deserialize, Serialize};
    use serde_test::{assert_tokens, Token};

    // A wrapper giving ArcSwap equality, mirroring the crate's own serde tests.
    #[derive(Debug, Serialize, Deserialize)]
    #[serde(transparent)]
    struct Wrap(arc_swap::ArcSwap<i32>);
    impl PartialEq for Wrap {
        fn eq(&self, other: &Self) -> bool {
            self.0.load().eq(&other.0.load())
        }
    }

    let shared = Wrap(ArcSwap::from_pointee(42));
    assert_tokens(&shared, &[Token::I32(42)]);

    let (_, obs) = shared.0.load_observed();
    shared
        .0
        .compare_exchange_observed(&obs, Arc::new(43))
        .expect("commit should succeed");
    assert_tokens(&shared, &[Token::I32(43)]);
}
