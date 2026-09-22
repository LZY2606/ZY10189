//! Limitations and common pitfalls.
//!
//! # Sized types
//!
//! This currently works only for `Sized` types. Unsized types have „fat pointers“, which are twice
//! as large as the normal ones. The [`AtomicPtr`] doesn't support them. One could use something
//! like `AtomicU128` for them. The catch is this doesn't exist and the difference would make it
//! really hard to implement the debt storage/stripped down hazard pointers.
//!
//! A workaround is to use double indirection:
//!
//! ```rust
//! # use arc_swap::ArcSwap;
//! // This doesn't work:
//! // let data: ArcSwap<[u8]> = ArcSwap::new(Arc::from([1, 2, 3]));
//!
//! // But this does:
//! let data: ArcSwap<Box<[u8]>> = ArcSwap::from_pointee(Box::new([1, 2, 3]));
//! # drop(data);
//! ```
//!
//! It also may be possible to use `ArcSwap` with the [`triomphe::ThinArc`] (that crate needs
//! enabling a feature flag to cooperate with `ArcSwap`).
//!
//! # Too many [`Guard`]s
//!
//! There's only limited number of "fast" slots for borrowing from [`ArcSwap`] for each single
//! thread (currently 8, but this might change in future versions). If these run out, the algorithm
//! falls back to slower path.
//!
//! If too many [`Guard`]s are kept around, the performance might be poor. These are not intended
//! to be stored in data structures or used across async yield points.
//!
//! [`ArcSwap`]: crate::ArcSwap
//! [`Guard`]: crate::Guard
//! [`AtomicPtr`]: std::sync::atomic::AtomicPtr
//!
//! # No `Clone` implementation
//!
//! Previous version implemented [`Clone`], but it turned out to be very confusing to people, since
//! it created fully independent [`ArcSwap`]. Users expected the instances to be tied to each
//! other, that store in one would change the result of future load of the other.
//!
//! To emulate the original behaviour, one can do something like this:
//!
//! ```rust
//! # use arc_swap::ArcSwap;
//! # let old = ArcSwap::from_pointee(42);
//! let new = ArcSwap::new(old.load_full());
//! # let _ = new;
//! ```
//!
//! # Pointer comparison and the ABA problem
//!
//! The [`compare_and_swap`] method decides by comparing the *pointer*. If the stored value can
//! go `A -> B -> A` between the load and the compare-and-swap (including the case where the old
//! allocation was freed and its address got reused for a different value), the comparison can't
//! tell that anything happened. Most of the time that is fine ‒ if the pointer is the same, the
//! value is the same, so the outcome is equivalent.
//!
//! Sometimes, though, the caller needs to know that *its* observed generation is still the
//! current one (eg. because the value carries meaning beyond the pointer identity, or because
//! an update must be applied exactly once per observed state). For that, there's the
//! [`load_observed`] / [`compare_exchange_observed`] pair: the former hands out an
//! [`Observation`] token alongside the value, the latter commits only if that exact generation
//! is still current. Every write operation (even storing the same [`Arc`] again) advances the
//! generation, so the token can't be fooled by `A -> B -> A` switches. The generation counter
//! is `usize`-wide and wraps around after `usize::MAX / 2 + 1` writes; tokens older than that
//! may theoretically match again, which is the documented boundary of the guarantee ‒ see
//! [`Observation`] for details.
//!
//! [`triomphe::ThinArc`]: https://docs.rs/triomphe/latest/triomphe/struct.ThinArc.html
//! [`Arc`]: std::sync::Arc
//! [`compare_and_swap`]: crate::ArcSwapAny::compare_and_swap
//! [`compare_exchange_observed`]: crate::ArcSwapAny::compare_exchange_observed
//! [`load_observed`]: crate::ArcSwapAny::load_observed
//! [`Observation`]: crate::Observation
