using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Deployment.Runtime;

/// <summary>
/// Cache for model inference results.
/// </summary>
/// <typeparam name="T">The numeric type for tensors</typeparam>
public class ModelCache<T> where T : struct
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
            // Update access time and count
            entry.LastAccessed = DateTime.UtcNow;
            entry.AccessCount++;
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
    /// </summary>
    public int EvictOlderThan(TimeSpan maxAge)
    {
        var now = DateTime.UtcNow;
        var keysToRemove = _cache
            .Where(kvp => now - kvp.Value.LastAccessed > maxAge)
            .Select(kvp => kvp.Key)
            .ToList();

        foreach (var key in keysToRemove)
        {
            _cache.TryRemove(key, out _);
            _accessCounts.TryRemove(key, out _);
        }

        return keysToRemove.Count;
    }

    /// <summary>
    /// Evicts least recently used entries to maintain size limit.
    /// </summary>
    public int EvictLRU(int maxEntries)
    {
        if (_cache.Count <= maxEntries)
            return 0;

        var entriesToRemove = _cache
            .OrderBy(kvp => kvp.Value.LastAccessed)
            .Take(_cache.Count - maxEntries)
            .Select(kvp => kvp.Key)
            .ToList();

        foreach (var key in entriesToRemove)
        {
            _cache.TryRemove(key, out _);
            _accessCounts.TryRemove(key, out _);
        }

        return entriesToRemove.Count;
    }

    /// <summary>
    /// Evicts least frequently used entries.
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

        foreach (var key in entriesToRemove)
        {
            _cache.TryRemove(key, out _);
            _accessCounts.TryRemove(key, out _);
        }

        return entriesToRemove.Count;
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
        // Convert input array to bytes
        var bytes = new byte[input.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>()];
        Buffer.BlockCopy(input, 0, bytes, 0, bytes.Length);

        // Compute SHA256 hash
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(bytes);

        // Convert to hex string
        return Convert.ToHexString(hashBytes);
    }
}

/// <summary>
/// Represents a cached inference result.
/// </summary>
internal class CacheEntry<T> where T : struct
{
    public T[] Result { get; set; } = Array.Empty<T>();
    public DateTime CachedAt { get; set; }
    public DateTime LastAccessed { get; set; }
    public long AccessCount { get; set; }
}

/// <summary>
/// Statistics for the model cache.
/// </summary>
public class CacheStatistics
{
    public int TotalEntries { get; set; }
    public long TotalAccessCount { get; set; }
    public double AverageAccessCount { get; set; }
    public TimeSpan OldestEntryAge { get; set; }
    public TimeSpan NewestEntryAge { get; set; }
}
