using System.Collections.Concurrent;
using System.Security.Cryptography;
using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.PromptEngineering.Compression;

/// <summary>
/// Wrapper compressor that caches compression results for frequently used prompts.
/// </summary>
/// <remarks>
/// <para>
/// This compressor wraps another compressor and caches the results. When the same
/// prompt is compressed multiple times, the cached result is returned instead of
/// re-computing the compression. This is particularly useful for LLM-based compressors
/// where each compression call is expensive.
/// </para>
/// <para><b>For Beginners:</b> Remembers compressions so the same prompt doesn't need to be compressed twice.
///
/// Example:
/// <code>
/// var llmCompressor = new LLMSummarizationCompressor(myLlmFunc);
/// var cachingCompressor = new CachingCompressor(llmCompressor, maxCacheSize: 1000);
///
/// // First call - compresses using LLM (slow)
/// var result1 = await cachingCompressor.CompressAsync(prompt);
///
/// // Second call with same prompt - returns cached result (fast)
/// var result2 = await cachingCompressor.CompressAsync(prompt);
///
/// // result1 == result2, but second call was instant
/// </code>
///
/// When to use:
/// - Wrapping expensive compressors (LLM-based)
/// - When the same prompts are used repeatedly
/// - In production systems to reduce API calls
/// </para>
/// </remarks>
public class CachingCompressor : PromptCompressorBase
{
    private readonly IPromptCompressor _innerCompressor;
    private readonly ConcurrentDictionary<string, CacheEntry> _cache;
    private readonly int _maxCacheSize;
    private readonly TimeSpan _cacheExpiry;
    private readonly object _lockObj = new();

    /// <summary>
    /// Represents a cached compression result.
    /// </summary>
    private class CacheEntry
    {
        public string CompressedPrompt { get; set; } = string.Empty;
        public DateTime CachedAt { get; set; }
        public DateTime LastAccessedAt { get; set; }
        public int AccessCount { get; set; }
    }

    /// <summary>
    /// Initializes a new instance of the CachingCompressor class.
    /// </summary>
    /// <param name="innerCompressor">The compressor to wrap with caching.</param>
    /// <param name="maxCacheSize">Maximum number of entries in the cache.</param>
    /// <param name="cacheExpiry">How long cached entries remain valid.</param>
    /// <param name="tokenCounter">Optional custom token counter function.</param>
    public CachingCompressor(
        IPromptCompressor innerCompressor,
        int maxCacheSize = 1000,
        TimeSpan? cacheExpiry = null,
        Func<string, int>? tokenCounter = null)
        : base($"Caching({innerCompressor.Name})", tokenCounter)
    {
        Guard.NotNull(innerCompressor);
        _innerCompressor = innerCompressor;
        _maxCacheSize = maxCacheSize > 0 ? maxCacheSize : 1000;
        _cacheExpiry = cacheExpiry ?? TimeSpan.FromHours(24);
        _cache = new ConcurrentDictionary<string, CacheEntry>();
    }

    /// <summary>
    /// Gets the number of items currently in the cache.
    /// </summary>
    public int CacheCount => _cache.Count;

    /// <summary>
    /// Gets the cache hit ratio (hits / total requests).
    /// </summary>
    public double CacheHitRatio
    {
        get
        {
            if (_totalRequests == 0) return 0;
            return (double)_cacheHits / _totalRequests;
        }
    }

    private long _cacheHits;
    private long _totalRequests;

    /// <summary>
    /// Compresses the prompt, using cache if available.
    /// </summary>
    protected override string CompressCore(string prompt, CompressionOptions options)
    {
        var cacheKey = ComputeCacheKey(prompt, options);
        Interlocked.Increment(ref _totalRequests);

        // Try to get from cache
        if (_cache.TryGetValue(cacheKey, out var entry))
        {
            if (DateTime.UtcNow - entry.CachedAt < _cacheExpiry)
            {
                Interlocked.Increment(ref _cacheHits);
                entry.LastAccessedAt = DateTime.UtcNow;
                entry.AccessCount++;
                return entry.CompressedPrompt;
            }

            // Entry expired, remove it
            _cache.TryRemove(cacheKey, out _);
        }

        // Compress and cache
        var compressed = _innerCompressor.Compress(prompt, options);
        AddToCache(cacheKey, compressed);

        return compressed;
    }

    /// <summary>
    /// Compresses the prompt asynchronously, using cache if available.
    /// </summary>
    public override async Task<string> CompressAsync(
        string prompt,
        CompressionOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var opts = options ?? CompressionOptions.Default;
        var cacheKey = ComputeCacheKey(prompt, opts);
        Interlocked.Increment(ref _totalRequests);

        // Try to get from cache
        if (_cache.TryGetValue(cacheKey, out var entry))
        {
            if (DateTime.UtcNow - entry.CachedAt < _cacheExpiry)
            {
                Interlocked.Increment(ref _cacheHits);
                entry.LastAccessedAt = DateTime.UtcNow;
                entry.AccessCount++;
                return entry.CompressedPrompt;
            }

            // Entry expired, remove it
            _cache.TryRemove(cacheKey, out _);
        }

        // Compress and cache
        var compressed = await _innerCompressor.CompressAsync(prompt, opts, cancellationToken)
            .ConfigureAwait(false);
        AddToCache(cacheKey, compressed);

        return compressed;
    }

    /// <summary>
    /// Computes a cache key for the prompt and options combination.
    /// </summary>
    private static string ComputeCacheKey(string prompt, CompressionOptions options)
    {
        // Create a deterministic key from the prompt and relevant options
        var keySource = $"{prompt}|{options.TargetReduction}|{options.MaxTokens}|{options.PreserveVariables}|{options.PreserveCodeBlocks}";

        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(keySource));
        return Convert.ToBase64String(hashBytes);
    }

    /// <summary>
    /// Adds an entry to the cache, evicting old entries if necessary.
    /// </summary>
    private void AddToCache(string key, string compressedPrompt)
    {
        // Evict if at capacity
        if (_cache.Count >= _maxCacheSize)
        {
            EvictOldEntries();
        }

        var entry = new CacheEntry
        {
            CompressedPrompt = compressedPrompt,
            CachedAt = DateTime.UtcNow,
            LastAccessedAt = DateTime.UtcNow,
            AccessCount = 1
        };

        _cache.TryAdd(key, entry);
    }

    /// <summary>
    /// Evicts old entries to make room for new ones.
    /// </summary>
    private void EvictOldEntries()
    {
        lock (_lockObj)
        {
            if (_cache.Count < _maxCacheSize)
            {
                return; // Another thread already evicted
            }

            // Remove expired entries first
            var now = DateTime.UtcNow;
            var expiredKeys = _cache
                .Where(kvp => now - kvp.Value.CachedAt >= _cacheExpiry)
                .Select(kvp => kvp.Key)
                .ToList();

            foreach (var key in expiredKeys)
            {
                _cache.TryRemove(key, out _);
            }

            // If still over capacity, remove least recently accessed
            if (_cache.Count >= _maxCacheSize)
            {
                var entriesToRemove = _cache
                    .OrderBy(kvp => kvp.Value.LastAccessedAt)
                    .ThenBy(kvp => kvp.Value.AccessCount)
                    .Take(_maxCacheSize / 4) // Remove 25%
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var key in entriesToRemove)
                {
                    _cache.TryRemove(key, out _);
                }
            }
        }
    }

    /// <summary>
    /// Clears all entries from the cache.
    /// </summary>
    public void ClearCache()
    {
        _cache.Clear();
        Interlocked.Exchange(ref _cacheHits, 0);
        Interlocked.Exchange(ref _totalRequests, 0);
    }

    /// <summary>
    /// Removes expired entries from the cache.
    /// </summary>
    /// <returns>The number of entries removed.</returns>
    public int PurgeExpiredEntries()
    {
        var now = DateTime.UtcNow;
        var expiredKeys = _cache
            .Where(kvp => now - kvp.Value.CachedAt >= _cacheExpiry)
            .Select(kvp => kvp.Key)
            .ToList();

        foreach (var key in expiredKeys)
        {
            _cache.TryRemove(key, out _);
        }

        return expiredKeys.Count;
    }

    /// <summary>
    /// Gets statistics about the cache.
    /// </summary>
    /// <returns>A dictionary with cache statistics.</returns>
    public Dictionary<string, object> GetCacheStats()
    {
        return new Dictionary<string, object>
        {
            { "CacheCount", CacheCount },
            { "MaxCacheSize", _maxCacheSize },
            { "CacheHitRatio", CacheHitRatio },
            { "TotalRequests", _totalRequests },
            { "CacheHits", _cacheHits },
            { "CacheMisses", _totalRequests - _cacheHits },
            { "CacheExpiry", _cacheExpiry }
        };
    }
}
