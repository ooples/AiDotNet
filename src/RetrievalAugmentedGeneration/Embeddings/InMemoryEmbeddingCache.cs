using System.Collections.Concurrent;
using System.Threading;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.RetrievalAugmentedGeneration.Embeddings;

/// <summary>
/// A thread-safe, in-memory <see cref="IEmbeddingCache{T}"/> backed by a <see cref="ConcurrentDictionary{TKey,TValue}"/>
/// with optional maximum size and approximate least-recently-used (LRU) eviction.
/// </summary>
/// <remarks>
/// <para>
/// When <c>maxSize</c> is zero (or negative) the cache is unbounded. When a positive <c>maxSize</c> is
/// supplied, inserting a new entry that would exceed the limit evicts the least-recently-used entries
/// until the cache is back within bounds. Recency is tracked with a monotonically increasing sequence
/// number that is updated on every read (hit) and write.
/// </para>
/// <para><b>For Beginners:</b> This keeps embeddings in memory so repeated text is not recomputed.
///
/// - It is safe to use from many threads at once.
/// - If you set a maximum size, the entries you have not used for the longest time are removed first
///   once the cache is full (this is called "LRU" eviction).
/// - If you do not set a maximum size, it keeps everything until you clear it.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for vector calculations (typically float or double).</typeparam>
public sealed class InMemoryEmbeddingCache<T> : IEmbeddingCache<T>
{
    private sealed class CacheEntry
    {
        public CacheEntry(Vector<T> value, long sequence)
        {
            Value = value;
            Sequence = sequence;
        }

        public Vector<T> Value { get; set; }

        /// <summary>Recency marker; larger values are more recently used.</summary>
        public long Sequence;
    }

    private readonly ConcurrentDictionary<string, CacheEntry> _entries = new();
    private readonly int _maxSize;
    private readonly object _evictionLock = new();
    private long _sequence;

    /// <summary>
    /// Initializes a new instance of the <see cref="InMemoryEmbeddingCache{T}"/> class.
    /// </summary>
    /// <param name="maxSize">
    /// The maximum number of entries to retain before LRU eviction begins.
    /// Use <c>0</c> (the default) or any non-positive value for an unbounded cache.
    /// </param>
    public InMemoryEmbeddingCache(int maxSize = 0)
    {
        _maxSize = maxSize;
    }

    /// <summary>
    /// Gets the maximum number of entries this cache retains, or <c>0</c> if the cache is unbounded.
    /// </summary>
    public int MaxSize => _maxSize;

    /// <inheritdoc />
    public int Count => _entries.Count;

    /// <inheritdoc />
    public bool TryGet(string key, out Vector<T>? embedding)
    {
        if (key == null)
            throw new ArgumentNullException(nameof(key));

        if (_entries.TryGetValue(key, out var entry))
        {
            // Mark as recently used.
            entry.Sequence = Interlocked.Increment(ref _sequence);
            embedding = entry.Value;
            return true;
        }

        embedding = null;
        return false;
    }

    /// <inheritdoc />
    public void Set(string key, Vector<T> embedding)
    {
        if (key == null)
            throw new ArgumentNullException(nameof(key));
        if (embedding == null)
            throw new ArgumentNullException(nameof(embedding));

        var seq = Interlocked.Increment(ref _sequence);
        _entries.AddOrUpdate(
            key,
            _ => new CacheEntry(embedding, seq),
            (_, existing) =>
            {
                existing.Value = embedding;
                existing.Sequence = seq;
                return existing;
            });

        if (_maxSize > 0 && _entries.Count > _maxSize)
        {
            EvictIfNeeded();
        }
    }

    /// <inheritdoc />
    public void Clear()
    {
        _entries.Clear();
    }

    /// <summary>
    /// Evicts least-recently-used entries until the cache is within its configured maximum size.
    /// </summary>
    private void EvictIfNeeded()
    {
        // A single lock keeps eviction correct under concurrency. Reads/writes to the
        // ConcurrentDictionary remain lock-free; only the (rare) shrink path serializes.
        lock (_evictionLock)
        {
            while (_entries.Count > _maxSize)
            {
                string? lruKey = null;
                long lruSequence = long.MaxValue;

                foreach (var pair in _entries)
                {
                    var seq = Interlocked.Read(ref pair.Value.Sequence);
                    if (seq < lruSequence)
                    {
                        lruSequence = seq;
                        lruKey = pair.Key;
                    }
                }

                if (lruKey == null)
                    break;

                _entries.TryRemove(lruKey, out _);
            }
        }
    }
}
