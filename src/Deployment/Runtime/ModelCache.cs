using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Deployment.Runtime;

/// <summary>
/// Cache for model inference results.
/// </summary>
/// <typeparam name="T">The numeric type for tensors</typeparam>
public class ModelCache<T>
{
    private readonly bool _enabled;
    private readonly ConcurrentDictionary<string, CacheEntry<T>> _cache;
    private readonly ConcurrentDictionary<string, long> _accessCounts;

    public ModelCache(bool enabled = true)
    {
        _enabled = enabled;
        _cache = new ConcurrentDictionary<string, CacheEntry<T>>();
        _accessCounts = new ConcurrentDictionary<string, long>();
    }

    /// <summary>
    /// Gets a cached result for the given input.
    /// </summary>
    public T[]? Get(string modelKey, T[] input)
    {
        if (!_enabled) return null;

        var inputHash = ComputeHash(input);
        var cacheKey = $"{modelKey}:{inputHash}";

        if (_cache.TryGetValue(cacheKey, out var entry))
        {
            // Update access time and count atomically
            Interlocked.Increment(ref entry.AccessCount);
            Interlocked.Exchange(ref entry.LastAccessedTicks, DateTime.UtcNow.Ticks);
            _accessCounts.AddOrUpdate(cacheKey, 1, (_, count) => count + 1);

            return entry.Result;
        }

        return null;
    }

    /// <summary>
    /// Puts a result in the cache.
    /// </summary>
    public void Put(string modelKey, T[] input, T[] result)
    {
        if (!_enabled) return;

        var inputHash = ComputeHash(input);
        var cacheKey = $"{modelKey}:{inputHash}";

        var entry = new CacheEntry<T>
        {
            Result = result,
            CachedAt = DateTime.UtcNow,
            LastAccessed = DateTime.UtcNow,
            AccessCount = 0
        };

        _cache[cacheKey] = entry;
        _accessCounts[cacheKey] = 0;
    }

    /// <summary>
    /// Clears the cache.
    /// </summary>
    public void Clear()
    {
        _cache.Clear();
        _accessCounts.Clear();
    }

    /// <summary>
    /// Removes entries older than the specified age.
    /// Note: Enumeration provides a snapshot of the cache, so concurrent modifications are safe but counts may be approximate.
    /// </summary>
    public int EvictOlderThan(TimeSpan maxAge)
    {
        var now = DateTime.UtcNow;
        var keysToRemove = _cache
            .Where(kvp => now - new DateTime(Interlocked.Read(ref kvp.Value.LastAccessedTicks), DateTimeKind.Utc) > maxAge)
            .Select(kvp => kvp.Key)
            .ToList();

        int removed = 0;
        foreach (var key in keysToRemove)
        {
            if (_cache.TryRemove(key, out _))
            {
                _accessCounts.TryRemove(key, out _);
                removed++;
            }
        }

        return removed;
    }

    /// <summary>
    /// Evicts least recently used entries to maintain size limit.
    /// Note: Enumeration provides a snapshot of the cache, so concurrent modifications are safe but counts may be approximate.
    /// </summary>
    public int EvictLRU(int maxEntries)
    {
        if (_cache.Count <= maxEntries)
            return 0;

        var entriesToRemove = _cache
            .OrderBy(kvp => new DateTime(Interlocked.Read(ref kvp.Value.LastAccessedTicks), DateTimeKind.Utc))
            .Take(_cache.Count - maxEntries)
            .Select(kvp => kvp.Key)
            .ToList();

        int removed = 0;
        foreach (var key in entriesToRemove)
        {
            if (_cache.TryRemove(key, out _))
            {
                _accessCounts.TryRemove(key, out _);
                removed++;
            }
        }

        return removed;
    }

    /// <summary>
    /// Evicts least frequently used entries.
    /// Note: Enumeration provides a snapshot of the cache, so concurrent modifications are safe but counts may be approximate.
    /// </summary>
    public int EvictLFU(int maxEntries)
    {
        if (_cache.Count <= maxEntries)
            return 0;

        var entriesToRemove = _accessCounts
            .OrderBy(kvp => kvp.Value)
            .Take(_cache.Count - maxEntries)
            .Select(kvp => kvp.Key)
            .ToList();

        int removed = 0;
        foreach (var key in entriesToRemove)
        {
            if (_cache.TryRemove(key, out _))
            {
                _accessCounts.TryRemove(key, out _);
                removed++;
            }
        }

        return removed;
    }

    /// <summary>
    /// Gets cache statistics.
    /// </summary>
    public CacheStatistics GetStatistics()
    {
        var now = DateTime.UtcNow;
        var entries = _cache.Values.ToList();

        return new CacheStatistics
        {
            TotalEntries = _cache.Count,
            TotalAccessCount = _accessCounts.Values.Sum(),
            AverageAccessCount = entries.Any() ? entries.Average(e => e.AccessCount) : 0,
            OldestEntryAge = entries.Any()
                ? now - entries.Min(e => e.CachedAt)
                : TimeSpan.Zero,
            NewestEntryAge = entries.Any()
                ? now - entries.Max(e => e.CachedAt)
                : TimeSpan.Zero
        };
    }

    private string ComputeHash(T[] input)
    {
        if (input == null || input.Length == 0)
            return "empty";

        // Compute stable hash using array content and length
        // Uses unchecked context to allow overflow for hash combining
        unchecked
        {
            int hash = 17;
            hash = hash * 31 + input.Length;

            // Combine hash codes of all elements
            // For large arrays, sample every nth element for performance
            int sampleSize = Math.Min(input.Length, 100);
            int step = Math.Max(1, input.Length / sampleSize);

            for (int i = 0; i < input.Length; i += step)
            {
                T element = input[i];
                if (element is not null)
                {
                    hash = hash * 31 + element.GetHashCode();
                }
            }

            // Include last element if not already sampled
            if (input.Length > 1 && (input.Length - 1) % step != 0)
            {
                T lastElement = input[input.Length - 1];
                if (lastElement is not null)
                {
                    hash = hash * 31 + lastElement.GetHashCode();
                }
            }

            // Convert to hex string for readability
            return hash.ToString("X8");
        }
    }
}

/// <summary>
/// Represents a cached inference result with thread-safe access tracking.
/// </summary>
internal class CacheEntry<T>
{
    public T[] Result { get; set; } = Array.Empty<T>();
    public DateTime CachedAt { get; set; }
    public long LastAccessedTicks;
    public long AccessCount;

    public DateTime LastAccessed
    {
        get => new DateTime(Interlocked.Read(ref LastAccessedTicks), DateTimeKind.Utc);
        set => Interlocked.Exchange(ref LastAccessedTicks, value.Ticks);
    }
}
