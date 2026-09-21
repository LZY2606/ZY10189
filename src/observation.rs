//! Generation tokens for conditional updates.
//!
//! The ordinary [`compare_and_swap`](crate::ArcSwapAny::compare_and_swap) identifies the value by
//! the *address* of its [`Arc`](std::sync::Arc). If the same address happens to be reused after
//! the previous value was destroyed (or the value simply toggles back and forth), an address-only
//! comparison can't tell "the same allocation, a different publication" from the value the caller
//! has actually observed. This is the classic ABA problem.
//!
//! The [`Observation`] type pairs the address with a [`generation`](Observation::generation)
//! counter that advances every time a *new* value is published into the same
//! [`ArcSwapAny`](crate::ArcSwapAny). [`load_observed`](crate::ArcSwapAny::load_observed) captures
//! such a pair and
//! [`compare_exchange_observed`](crate::ArcSwapAny::compare_exchange_observed) only commits when
//! the storage is still at *that exact publication*.
//!
//! Unlike a [`Guard`](crate::Guard), an `Observation` does **not** protect the observed value and
//! does **not** hold a reference count. The value may be dropped and its address reused while the
//! observation is carried around; that is exactly the situation the generation detects. The
//! observation is therefore a lightweight, `Copy`-able token, not a smart pointer.
//!
//! # Generation width and wrap
//!
//! The generation is the full machine word (`usize`, 64 bits on all platforms arc-swap supports).
//! Every *successful* publication (see [`store`](crate::ArcSwapAny::store),
//! [`swap`](crate::ArcSwapAny::swap), both flavours of compare-and-swap and
//! [`rcu`](crate::ArcSwapAny::rcu)) increments it by one, starting at `0` for the value the
//! storage was created with.
//!
//! The counter deliberately saturates at `usize::MAX` instead of wrapping. An observation taken
//! at the saturated generation can never be the basis of a successful conditional update, and
//! publications stop advancing the generation (the address still changes). This guarantees an
//! arbitrarily old token can never become valid again due to wrap-around. The practical boundary
//! is `usize::MAX - 1` successful publications per storage instance; on 64-bit platforms that is
//! about 1.8·10¹⁹ updates. This is *not* a claim that address-based ABA is eliminated for free —
//! it is eliminated only while the generation counter is live, which is what saturating instead
//! of wrapping preserves.
//!
//! # Memory ordering
//!
//! All generation reads and writes use [`SeqCst`](core::sync::atomic::Ordering::SeqCst), matching
//! the orderings of the pointer publication itself. A successful
//! `compare_exchange_observed` therefore synchronizes with the
//! [`load_observed`](crate::ArcSwapAny::load_observed) that produced the token, in addition to the
//! usual acquire/release guarantees of the pointer swap.
//!
//! # Token identity
//!
//! Two tokens compare equal (and both can drive a successful conditional update) exactly when the
//! storage is at the same publication: identical address *and* identical generation. Two
//! `load_observed` calls that race with an in-flight publication can disagree (one may catch the
//! publication mid-step), but such a trailing-edge token is detected at commit time because the
//! commit re-verifies the pair under the publication lock. A failed commit reports the pair it
//! actually found, which lets the caller distinguish "address changed" from "same address, newer
//! generation" via [`Observation::same_address`].
//!
//! # Complexity
//!
//! `load_observed` performs two `SeqCst` loads of the generation around one ordinary
//! [`load`](crate::ArcSwapAny::load); it never allocates and only touches a debt slot when the
//! observed pointer has to be protected long enough to notice a concurrent publication.
//! `compare_exchange_observed` adds one generation load and one conditional generation store to
//! the cost of the ordinary [`compare_and_swap`](crate::ArcSwapAny::compare_and_swap). Both
//! operations briefly acquire the same per-instance lightweight write spin-lock that all
//! publications use; this lock is never touched by ordinary readers.

use core::fmt;
use core::marker::PhantomData;

use crate::as_raw::sealed::Sealed;
use crate::ref_cnt::RefCnt;
use crate::AsRaw;

/// The value at which the generation counter saturates.
///
/// Observations carrying this generation can never match in
/// [`compare_exchange_observed`](crate::ArcSwapAny::compare_exchange_observed), even if the
/// address still happens to equal. This is what prevents a token from becoming valid again after
/// the generation would otherwise wrap.
pub(crate) const GENERATION_SATURATED: usize = usize::MAX;

/// An observation token identifying one particular publication of a value.
///
/// Obtained from [`load_observed`](crate::ArcSwapAny::load_observed) and consumed by
/// [`compare_exchange_observed`](crate::ArcSwapAny::compare_exchange_observed). See the
/// [module documentation](crate::observation) for the detailed semantics, in particular the generation width,
/// saturation and memory ordering.
///
/// An `Observation` is:
///
/// * `Copy` and cheap to clone — it contains only a raw address and a generation number.
/// * Not a smart pointer. It does not keep the observed value alive and dereferencing the address
///   through it is not possible. Use [`Guard`](crate::Guard) or
///   [`load_full`](crate::ArcSwapAny::load_full) for that.
/// * Specific to one [`ArcSwapAny`](crate::ArcSwapAny) instance by construction, but the type
///   system can't enforce that; using a token from one storage with another always fails the
///   comparison (the generation counter is per-instance) and never causes undefined behaviour.
///
/// # Examples
///
/// ```rust
/// use std::sync::Arc;
///
/// use arc_swap::ArcSwap;
///
/// let shared = ArcSwap::from_pointee(7);
/// let observation = shared.load_observed().1;
///
/// // Nothing changed, the conditional update commits.
/// let updated = shared.compare_exchange_observed(observation, Arc::new(8));
/// assert!(updated.is_ok());
///
/// // The same token can't commit twice: the first update already published a new generation.
/// let again = shared.compare_exchange_observed(observation, Arc::new(9));
/// assert!(again.is_err());
/// assert_eq!(8, **shared.load());
/// ```
pub struct Observation<T: RefCnt> {
    /// Raw address of the observed publication. Null represents `None` for nullable storages.
    pub(crate) ptr: *mut T::Base,

    /// Generation at which the address was observed.
    pub(crate) generation: usize,

    /// We carry the smart-pointer type for `as_raw` and `Send`/`Sync` bounds only.
    _phantom: PhantomData<T>,
}

impl<T: RefCnt> Observation<T> {
    pub(crate) fn new(ptr: *mut T::Base, generation: usize) -> Self {
        Self {
            ptr,
            generation,
            _phantom: PhantomData,
        }
    }

    /// The generation of the observed publication.
    ///
    /// This is the value that distinguishes two publications at the same address. Two
    /// observations with equal addresses and equal generations describe the very same
    /// publication; equal addresses with different generations describe an ABA transition.
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Compares the observed address with another pointer-like value.
    ///
    /// This compares only the address, not the generation. It is mostly useful after a failed
    /// [`compare_exchange_observed`](crate::ArcSwapAny::compare_exchange_observed) to tell apart
    /// "the address changed" from "same address, but a different generation" (an ABA transition):
    ///
    /// ```rust
    /// use std::sync::Arc;
    ///
    /// use arc_swap::ArcSwap;
    ///
    /// let shared = ArcSwap::from_pointee(1);
    /// let token = shared.load_observed().1;
    ///
    /// // Replace the value with a different Arc at a different address.
    /// shared.store(Arc::new(2));
    /// let current = shared.load_observed().1;
    /// assert!(!token.same_address(&current));
    /// ```
    pub fn same_address<C: AsRaw<T::Base>>(&self, current: &C) -> bool {
        self.ptr == current.as_raw()
    }

    /// Returns the observed address as a numeric value.
    ///
    /// This is offered for diagnostics and tests; the address must not be dereferenced. It is
    /// `0` for a `None` publication of [`ArcSwapOption`](crate::ArcSwapOption).
    pub fn ptr_addr(&self) -> usize {
        self.ptr as usize
    }

    /// Returns true if this observation was taken at the saturated generation.
    ///
    /// Such observations can never make
    /// [`compare_exchange_observed`](crate::ArcSwapAny::compare_exchange_observed) succeed. See the
    /// [module documentation](crate::observation#generation-width-and-wrap) for the boundary.
    pub fn is_saturated(&self) -> bool {
        self.generation == GENERATION_SATURATED
    }
}

impl<T: RefCnt> Clone for Observation<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: RefCnt> Copy for Observation<T> {}

impl<T: RefCnt> PartialEq for Observation<T> {
    /// Two observations are equal iff they describe the same publication — same address and same
    /// generation.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr && self.generation == other.generation
    }
}

impl<T: RefCnt> Eq for Observation<T> {}

impl<T: RefCnt> fmt::Debug for Observation<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter
            .debug_struct("Observation")
            .field("ptr", &self.ptr)
            .field("generation", &self.generation)
            .finish()
    }
}

// The observation only carries a raw address and a number, but semantically it is a witness about
// a value of type `T`. Send/Sync therefore follow the same rules as `T` itself (the same bounds
// `Arc<T>` uses). We never dereference the address and never touch the value, so sharing the
// *address* across threads is sound.
unsafe impl<T: RefCnt + Send> Send for Observation<T> {}
unsafe impl<T: RefCnt + Send + Sync> Sync for Observation<T> {}

impl<T: RefCnt> Sealed for Observation<T> {}
impl<T: RefCnt> AsRaw<T::Base> for Observation<T> {
    fn as_raw(&self) -> *mut T::Base {
        self.ptr
    }
}
