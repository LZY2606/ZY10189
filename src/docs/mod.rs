//! Additional documentation.
//!
//! Here we have some more general topics that might be good to know that just don't fit to the
//! crate level intro.
//!
//! Also, there were some previous blog posts about the crate which you might find interesting.
//!
//! # Atomic orderings
//!
//! Each operation on the [`ArcSwapAny`] with [`DefaultStrategy`] type callable concurrently (eg.
//! [`load`], but not [`into_inner`]) contains at least one [`SeqCst`] atomic read-write operation,
//! therefore even operations on different instances have a defined global order of operations.
//!
//! # Generation observations (ABA-safe conditional updates)
//!
//! [`compare_and_swap`] identifies the current value by the address of the [`Arc`]. That is
//! ambiguous in two situations: storing the very same `Arc` instance twice, and the address of a
//! dropped value being reused by a fresh allocation. In both cases the address "comes back" — the
//! classic ABA problem — and an address-only conditional update can't tell that the publication
//! is a different one.
//!
//! [`load_observed`] returns, besides the usual [`Guard`], an [`Observation`] token that pairs
//! the address with a per-instance *generation*. Every successful publication ([`store`],
//! [`swap`], [`compare_and_swap`], [`compare_exchange_observed`] and [`rcu`]) increments the
//! generation, starting at `0`. Storing the same `Arc` twice produces two generations; `None` in
//! [`ArcSwapOption`] is a publication with a generation of its own.
//!
//! [`compare_exchange_observed`] commits only if both the address and the generation still match
//! the token. On failure it returns the observation found at verification time, so the caller can
//! compare it with the stale token to distinguish a plain change (different address) from an ABA
//! transition (same address, newer generation).
//!
//! The token does **not** keep the observed value alive — it is just an address and a number, and
//! is `Copy` + `Send`/`Sync` under the usual bounds. The generation is the full machine word and
//! saturates at `usize::MAX` instead of wrapping; an observation at the saturated generation can
//! never commit. This means an arbitrarily old token can never become valid again, at the cost of
//! a finite (on 64-bit platforms, `usize::MAX - 1` ≈ 1.8\u{2024}10\u{00b9} publications per
//! instance) boundary. This deliberately does not claim to have eliminated ABA beyond that
//! boundary.
//!
//! Ordinary readers ([`load`]/[`load_full`]) never touch the generation and never take locks or
//! allocate additional memory per access. Writers additionally serialize the publication step
//! (pointer swap + generation bump) on a small per-instance spin-lock, released before debt
//! reclamation; this is the only overhead paid by users that don't call the observed methods. All
//! generation accesses use [`SeqCst`].
//!
//! [`ArcSwapOption`]: crate::ArcSwapOption
//! [`Observation`]: crate::Observation
//! [`Guard`]: crate::Guard
//! [`compare_and_swap`]: crate::ArcSwapAny::compare_and_swap
//! [`compare_exchange_observed`]: crate::ArcSwapAny::compare_exchange_observed
//! [`load_observed`]: crate::ArcSwapAny::load_observed
//! [`load`]: crate::ArcSwapAny::load
//! [`load_full`]: crate::ArcSwapAny::load_full
//! [`store`]: crate::ArcSwapAny::store
//! [`swap`]: crate::ArcSwapAny::swap
//! [`rcu`]: crate::ArcSwapAny::rcu
//!
//! # Features
//!
//! The `weak` feature adds the ability to use arc-swap with the [`Weak`] pointer too,
//! through the [`ArcSwapWeak`] type. The needed std support is stabilized in rust version 1.45 (as
//! of now in beta).
//!
//! The `experimental-strategies` enables few more strategies that can be used. Note that these
//! **are not** part of the API stability guarantees and they may be changed, renamed or removed at
//! any time.
//!
//! The `experimental-thread-local` feature can be used to build arc-swap for `no_std` targets, by
//! replacing occurences of [`std::thread_local!`] with the `#[thread_local]` directive. This
//! requires a nightly Rust compiler as it makes use of the experimental
//! [`thread_local`](https://doc.rust-lang.org/unstable-book/language-features/thread-local.html)
//! feature. Using this features, thread-local variables are compiled using LLVM built-ins, which
//! have [several underlying modes of
//! operation](https://doc.rust-lang.org/beta/unstable-book/compiler-flags/tls-model.html).  To add
//! support for thread-local variables on a platform that does not have OS or linker support, the
//! easiest way is to use `-Ztls-model=emulated` and to implement `__emutls_get_address` by hand,
//! as in [this
//! example](https://opensource.apple.com/source/clang/clang-800.0.38/src/projects/compiler-rt/lib/builtins/emutls.c.auto.html)
//! from Clang.
//!
//! # Minimal compiler version
//!
//! The `1` versions will compile on all compilers supporting the 2018 edition. Note that this
//! applies only if no additional feature flags are enabled and does not apply to compiling or
//! running tests.
//!
//! [`ArcSwapAny`]: crate::ArcSwapAny
//! [`ArcSwapWeak`]: crate::ArcSwapWeak
//! [`Arc`]: std::sync::Arc
//! [`Observation`]: crate::Observation
//! [`Guard`]: crate::Guard
//! [`load`]: crate::ArcSwapAny::load
//! [`load_full`]: crate::ArcSwapAny::load_full
//! [`into_inner`]: crate::ArcSwapAny::into_inner
//! [`DefaultStrategy`]: crate::DefaultStrategy
//! [`SeqCst`]: std::sync::atomic::Ordering::SeqCst
//! [`Weak`]: std::sync::Weak

pub mod internal;
pub mod limitations;
pub mod patterns;
pub mod performance;
