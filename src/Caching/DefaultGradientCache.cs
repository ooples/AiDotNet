global using System.Collections.Concurrent;
using AiDotNet.Attributes;
using AiDotNet.Enums;

using CacheEvictionPolicy = AiDotNet.Enums.CacheEvictionPolicy;
namespace AiDotNet.Caching;

/// <summary>
/// Provides a default implementation of the gradient caching mechanism for symbolic models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A gradient cache is like a memory bank that stores pre-calculated mathematical operations
/// that are frequently used during machine learning model training.
///
/// In machine learning, "gradients" are calculations that show how much a model's error would change
/// if we slightly adjusted a specific parameter. These calculations can be complex and time-consuming,
/// especially if they need to be repeated many times.
///
/// By storing these calculations in a cache (a temporary storage area), we can avoid recalculating the
/// same gradients repeatedly, which makes the training process much faster. Think of it like remembering
/// the answer to a difficult math problem so you don't have to solve it again when you need the same answer later.
///
/// This class provides a simple way to store and retrieve these gradient calculations using string keys
/// (like names or identifiers) to look them up quickly.
/// </para>
/// <para>
/// <b>Bounded capacity (memory-leak fix):</b> the cache evicts its oldest entry once the number of
/// stored gradients exceeds <see cref="Capacity"/>. This matters because the optimizer's gradient
/// cache key (<c>GradientBasedOptimizerBase.GenerateGradientCacheKey</c>) folds in a per-step
/// parameter-state fingerprint AND the per-batch input/target reference identities. During the batched
/// <c>Optimize</c> training loop every mini-batch step therefore produces a BRAND-NEW key that will
/// never be looked up again (the parameters have already moved on by the next step). Without a bound
/// the dictionary grew by one full parameter-sized gradient <c>Vector&lt;T&gt;</c> every step and was
/// only ever cleared by <c>Reset()</c> (which the training loop never calls) — a linear managed-heap
/// leak that made each epoch's GC progressively more expensive and per-epoch wall-time climb roughly
/// linearly with the epoch index. Bounding the cache is numerically transparent: a cache <i>hit</i>
/// returns the identical gradient that would otherwise be recomputed, and a <i>miss</i> (because an
/// old entry was evicted) simply recomputes that same gradient deterministically. The legitimate reuse
/// cases the cache exists for — line search / trust-region re-evaluation of the same solution, or a
/// caller invoking <c>CalculateGradient</c> twice in a row with identical arguments — all reuse a key
/// within a handful of consecutive calls, well inside the retained window, so they still hit.
/// </para>
/// </remarks>
[InfraType(InfraType.Cache)]
public class DefaultGradientCache<T> : IGradientCache<T>
{
    /// <summary>
    /// Default maximum number of gradient entries retained before the oldest is evicted.
    /// </summary>
    /// <remarks>
    /// Chosen to comfortably cover every legitimate short-window reuse pattern (numerical
    /// gradient-check double-evaluation, line-search / trust-region re-evaluation, DDP AllReduce
    /// read-back) while keeping the resident footprint bounded and independent of the number of
    /// training steps. Bounding at count rather than bytes keeps the policy model-size agnostic;
    /// the entry payload scales with the model, but the growth is capped at a constant multiple
    /// of a single gradient snapshot instead of growing without limit.
    /// </remarks>
    public const int DefaultCapacity = 8;

    /// <summary>
    /// Recommended default eviction policy: oldest-inserted first. A gradient key is reused only within
    /// a handful of consecutive calls, so FIFO retains exactly the recent window that legitimately hits.
    /// </summary>
    public const CacheEvictionPolicy DefaultEvictionPolicy = CacheEvictionPolicy.FIFO;

    /// <summary>Bounded, policy-evicting backing store (capacity + FIFO/LRU/LFU live here).</summary>
    private readonly BoundedEvictingStore<IGradientModel<T>> _store;

    /// <summary>When false the cache is a pass-through: every lookup misses and stores are dropped, so
    /// the optimizer always recomputes (numerically identical, just uncached).</summary>
    private readonly bool _enabled;

    /// <summary>Whether the cache actually stores entries (false = disabled pass-through).</summary>
    public bool Enabled => _enabled;

    /// <summary>
    /// Maximum number of gradient entries retained before an entry is evicted.
    /// </summary>
    public int Capacity => _store.Capacity;

    /// <summary>The policy used to choose the eviction victim once the cache is full.</summary>
    public CacheEvictionPolicy EvictionPolicy => _store.EvictionPolicy;

    /// <summary>
    /// Creates a gradient cache with the recommended defaults: <see cref="DefaultCapacity"/> entries,
    /// <see cref="DefaultEvictionPolicy"/> eviction.
    /// </summary>
    public DefaultGradientCache() : this(DefaultCapacity, DefaultEvictionPolicy)
    {
    }

    /// <summary>
    /// Creates a gradient cache bounded to <paramref name="capacity"/> entries with the recommended
    /// <see cref="DefaultEvictionPolicy"/> eviction.
    /// </summary>
    /// <param name="capacity">Maximum number of gradient entries to retain (must be positive).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="capacity"/> is not positive.</exception>
    public DefaultGradientCache(int capacity) : this(capacity, DefaultEvictionPolicy)
    {
    }

    /// <summary>
    /// Creates a gradient cache bounded to <paramref name="capacity"/> entries evicting per
    /// <paramref name="evictionPolicy"/>.
    /// </summary>
    /// <param name="capacity">Maximum number of gradient entries to retain (must be positive).</param>
    /// <param name="evictionPolicy">Which entry to evict once the cache is full (FIFO / LRU / LFU).</param>
    /// <param name="enabled">When false the cache is a pass-through that never stores (always recomputes).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="capacity"/> is not positive.</exception>
    public DefaultGradientCache(int capacity, CacheEvictionPolicy evictionPolicy, bool enabled = true)
    {
        _store = new BoundedEvictingStore<IGradientModel<T>>(capacity, evictionPolicy);
        _enabled = enabled;
    }

    /// <summary>
    /// Retrieves a cached gradient model using the specified key.
    /// </summary>
    /// <param name="key">The unique identifier for the gradient model.</param>
    /// <returns>The cached gradient model if found; otherwise, null.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method works like looking up a word in a dictionary. You provide a key (like a word),
    /// and it returns the corresponding gradient model (like the definition) if it exists in the cache.
    ///
    /// If the key doesn't exist in the cache, the method returns null, indicating that the gradient
    /// needs to be calculated from scratch.
    /// </para>
    /// </remarks>
    public IGradientModel<T>? GetCachedGradient(string key)
    {
        if (key == null) throw new ArgumentNullException(nameof(key), "Cache key cannot be null.");
        return _enabled ? _store.Get(key) : null;
    }

    /// <summary>
    /// Stores a gradient model in the cache with the specified key.
    /// </summary>
    /// <param name="key">The unique identifier for the gradient model.</param>
    /// <param name="gradient">The gradient model to cache.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method saves a gradient model in the cache so it can be retrieved later.
    ///
    /// It's like writing down the answer to a complex math problem with a label (the key) so you can
    /// look it up quickly next time instead of solving the problem again.
    ///
    /// If a gradient with the same key already exists in the cache, it will be replaced with the new one.
    /// Once the cache holds <see cref="Capacity"/> distinct keys, inserting a new one evicts the oldest
    /// so the cache never grows without bound (see the class remarks for why this matters).
    /// </para>
    /// </remarks>
    public void CacheGradient(string key, IGradientModel<T> gradient)
    {
        if (key == null) throw new ArgumentNullException(nameof(key), "Cache key cannot be null.");
        if (gradient == null) throw new ArgumentNullException(nameof(gradient), "Gradient cannot be null.");
        if (_enabled) _store.Set(key, gradient);
    }

    /// <summary>
    /// Removes all cached gradient models from the cache.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Use this method when you want to start fresh, such as when beginning a new training session
    /// or when you've made changes to your model that would invalidate previously cached gradients.
    /// </para>
    /// </remarks>
    public void ClearCache()
    {
        _store.Clear();
    }
}
