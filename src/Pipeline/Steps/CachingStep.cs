using AiDotNet.Enums;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace AiDotNet.Pipeline.Steps
{
    /// <summary>
    /// Pipeline step that provides caching functionality for expensive transformations
    /// </summary>
    public class CachingStep : PipelineStepBase
    {
        private readonly Dictionary<string, CacheEntry> _memoryCache = default!;
        private readonly string _cacheDirectory;
        private readonly object _cacheLock = new object();
        private TimeSpan _cacheExpiration = default!;
        private long _maxMemoryCacheSizeBytes;
        private long _currentMemoryUsage;
        private bool _enableDiskCache;
        private bool _enableMemoryCache;
        private CacheEvictionPolicy _evictionPolicy = default!;

        /// <summary>
        /// Enum for cache eviction policies
        /// </summary>
        public enum CacheEvictionPolicy
        {
            LRU, // Least Recently Used
            LFU, // Least Frequently Used
            FIFO, // First In First Out
            TTL // Time To Live only
        }

        /// <summary>
        /// Represents a cache entry
        /// </summary>
        private class CacheEntry
        {
            public double[][]? Data { get; set; }
            public DateTime CreatedAt { get; set; }
            public DateTime LastAccessedAt { get; set; }
            public int AccessCount { get; set; }
            public long SizeInBytes { get; set; }
            public string? DiskPath { get; set; }
        }

        /// <summary>
        /// Gets cache statistics
        /// </summary>
        public CacheStatistics Statistics { get; private set; }

        /// <summary>
        /// Cache statistics class
        /// </summary>
        public class CacheStatistics
        {
            public int TotalRequests { get; set; }
            public int CacheHits { get; set; }
            public int CacheMisses { get; set; }
            public long MemoryUsageBytes { get; set; }
            public int ItemsInMemory { get; set; }
            public int ItemsOnDisk { get; set; }
            public double HitRate => TotalRequests > 0 ? (double)CacheHits / TotalRequests : 0;
        }

        /// <summary>
        /// Initializes a new instance of the CachingStep class
        /// </summary>
        /// <param name="cacheDirectory">Directory for disk cache</param>
        /// <param name="name">Optional name for this step</param>
        public CachingStep(string? cacheDirectory = null, string? name = null) 
            : base(name ?? "Caching")
        {
            Position = PipelinePosition.Any;
            _memoryCache = new Dictionary<string, CacheEntry>();
            _cacheDirectory = cacheDirectory ?? Path.Combine(Path.GetTempPath(), "AiDotNet", "PipelineCache");
            _cacheExpiration = TimeSpan.FromHours(24);
            _maxMemoryCacheSizeBytes = 100 * 1024 * 1024; // 100 MB default
            _currentMemoryUsage = 0;
            _enableDiskCache = true;
            _enableMemoryCache = true;
            _evictionPolicy = CacheEvictionPolicy.LRU;
            Statistics = new CacheStatistics();

            // Ensure cache directory exists
            if (_enableDiskCache)
            {
                Directory.CreateDirectory(_cacheDirectory);
            }

            // Set default parameters
            SetParameter("CacheExpiration", _cacheExpiration);
            SetParameter("MaxMemoryCacheSizeMB", _maxMemoryCacheSizeBytes / (1024 * 1024));
            SetParameter("EnableDiskCache", _enableDiskCache);
            SetParameter("EnableMemoryCache", _enableMemoryCache);
            SetParameter("EvictionPolicy", _evictionPolicy);
        }

        /// <summary>
        /// Core fitting logic (caching step doesn't need fitting)
        /// </summary>
        protected override void FitCore(double[][] inputs, double[]? targets)
        {
            // Caching step doesn't need fitting
            UpdateMetadata("CacheDirectory", _cacheDirectory);
            UpdateMetadata("MaxMemoryCacheSizeMB", (_maxMemoryCacheSizeBytes / (1024 * 1024)).ToString());
        }

        /// <summary>
        /// Core transformation logic that checks cache before processing
        /// </summary>
        protected override double[][] TransformCore(double[][] inputs)
        {
            // Generate cache key based on input data
            var cacheKey = GenerateCacheKey(inputs);
            
            lock (_cacheLock)
            {
                Statistics.TotalRequests++;

                // Check memory cache first
                if (_enableMemoryCache && _memoryCache.TryGetValue(cacheKey, out var memoryEntry))
                {
                    memoryEntry.LastAccessedAt = DateTime.UtcNow;
                    memoryEntry.AccessCount++;
                    Statistics.CacheHits++;
                    UpdateMetadata("LastCacheHit", DateTime.UtcNow.ToString("O"));

                    if (memoryEntry.Data != null)
                    {
                        return CloneData(memoryEntry.Data);
                    }
                }

                // Check disk cache if memory cache miss
                if (_enableDiskCache)
                {
                    var diskPath = GetDiskCachePath(cacheKey);
                    if (File.Exists(diskPath))
                    {
                        try
                        {
                            var data = LoadFromDisk(diskPath);
                            Statistics.CacheHits++;
                            
                            // Add to memory cache if there's space
                            if (_enableMemoryCache)
                            {
                                AddToMemoryCache(cacheKey, data, diskPath);
                            }

                            return CloneData(data);
                        }
                        catch
                        {
                            // If disk read fails, treat as cache miss
                        }
                    }
                }

                Statistics.CacheMisses++;
            }

            // Cache miss - return original data and cache it asynchronously
            Task.Run(() => CacheData(cacheKey, inputs));
            
            return inputs;
        }

        /// <summary>
        /// Caches data asynchronously
        /// </summary>
        private void CacheData(string cacheKey, double[][] data)
        {
            lock (_cacheLock)
            {
                var dataSize = EstimateDataSize(data);

                // Save to disk cache
                if (_enableDiskCache)
                {
                    var diskPath = GetDiskCachePath(cacheKey);
                    try
                    {
                        SaveToDisk(data, diskPath);
                        Statistics.ItemsOnDisk++;
                        
                        // Add to memory cache if there's space
                        if (_enableMemoryCache)
                        {
                            AddToMemoryCache(cacheKey, data, diskPath);
                        }
                    }
                    catch
                    {
                        // Ignore disk write failures
                    }
                }
                else if (_enableMemoryCache)
                {
                    // Only memory cache enabled
                    AddToMemoryCache(cacheKey, data, null);
                }

                UpdateCacheStatistics();
            }
        }

        /// <summary>
        /// Adds data to memory cache with eviction if necessary
        /// </summary>
        private void AddToMemoryCache(string cacheKey, double[][] data, string? diskPath)
        {
            var dataSize = EstimateDataSize(data);

            // Check if we need to evict items
            while (_currentMemoryUsage + dataSize > _maxMemoryCacheSizeBytes && _memoryCache.Count > 0)
            {
                EvictFromMemoryCache();
            }

            if (_currentMemoryUsage + dataSize <= _maxMemoryCacheSizeBytes)
            {
                var entry = new CacheEntry
                {
                    Data = CloneData(data),
                    CreatedAt = DateTime.UtcNow,
                    LastAccessedAt = DateTime.UtcNow,
                    AccessCount = 1,
                    SizeInBytes = dataSize,
                    DiskPath = diskPath
                };

                _memoryCache[cacheKey] = entry;
                _currentMemoryUsage += dataSize;
                Statistics.ItemsInMemory = _memoryCache.Count;
                Statistics.MemoryUsageBytes = _currentMemoryUsage;
            }
        }

        /// <summary>
        /// Evicts an item from memory cache based on eviction policy
        /// </summary>
        private void EvictFromMemoryCache()
        {
            string? keyToEvict = null;

            switch (_evictionPolicy)
            {
                case CacheEvictionPolicy.LRU:
                    keyToEvict = _memoryCache
                        .OrderBy(kvp => kvp.Value.LastAccessedAt)
                        .FirstOrDefault().Key;
                    break;

                case CacheEvictionPolicy.LFU:
                    keyToEvict = _memoryCache
                        .OrderBy(kvp => kvp.Value.AccessCount)
                        .ThenBy(kvp => kvp.Value.LastAccessedAt)
                        .FirstOrDefault().Key;
                    break;

                case CacheEvictionPolicy.FIFO:
                    keyToEvict = _memoryCache
                        .OrderBy(kvp => kvp.Value.CreatedAt)
                        .FirstOrDefault().Key;
                    break;

                case CacheEvictionPolicy.TTL:
                    var expiredKeys = _memoryCache
                        .Where(kvp => DateTime.UtcNow - kvp.Value.CreatedAt > _cacheExpiration)
                        .Select(kvp => kvp.Key)
                        .ToList();
                    
                    if (expiredKeys.Any())
                    {
                        keyToEvict = expiredKeys.First();
                    }
                    else
                    {
                        // Fall back to LRU if no expired items
                        keyToEvict = _memoryCache
                            .OrderBy(kvp => kvp.Value.LastAccessedAt)
                            .FirstOrDefault().Key;
                    }
                    break;
            }

            if (keyToEvict != null && _memoryCache.TryGetValue(keyToEvict, out var entry))
            {
                _currentMemoryUsage -= entry.SizeInBytes;
                _memoryCache.Remove(keyToEvict);
                Statistics.ItemsInMemory = _memoryCache.Count;
                Statistics.MemoryUsageBytes = _currentMemoryUsage;
            }
        }

        /// <summary>
        /// Generates a cache key from input data
        /// </summary>
        private string GenerateCacheKey(double[][] data)
        {
            using (var sha256 = SHA256.Create())
            {
                var sb = new StringBuilder();
                
                // Include data dimensions
                sb.Append($"{data.Length}x{data[0].Length}:");
                
                // Sample some data points for the hash
                var sampleSize = Math.Min(100, data.Length);
                var step = Math.Max(1, data.Length / sampleSize);
                
                for (int i = 0; i < data.Length; i += step)
                {
                    for (int j = 0; j < Math.Min(10, data[i].Length); j++)
                    {
                        sb.Append(data[i][j].ToString("G17"));
                        sb.Append(',');
                    }
                }

                var bytes = Encoding.UTF8.GetBytes(sb.ToString());
                var hash = sha256.ComputeHash(bytes);
                return Convert.ToBase64String(hash).Replace("/", "_").Replace("+", "-");
            }
        }

        /// <summary>
        /// Gets the disk cache file path for a cache key
        /// </summary>
        private string GetDiskCachePath(string cacheKey)
        {
            return Path.Combine(_cacheDirectory, $"{cacheKey}.cache");
        }

        /// <summary>
        /// Saves data to disk
        /// </summary>
        private void SaveToDisk(double[][] data, string path)
        {
            var directory = Path.GetDirectoryName(path);
            if (directory != null && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var json = Newtonsoft.Json.JsonConvert.SerializeObject(data);
            File.WriteAllText(path, json);
        }

        /// <summary>
        /// Loads data from disk
        /// </summary>
        private double[][] LoadFromDisk(string path)
        {
            var json = File.ReadAllText(path);
            return Newtonsoft.Json.JsonConvert.DeserializeObject<double[][]>(json) ?? throw new InvalidOperationException("Failed to deserialize cache data");
        }

        /// <summary>
        /// Estimates the size of data in bytes
        /// </summary>
        private long EstimateDataSize(double[][] data)
        {
            return data.Length * data[0].Length * sizeof(double);
        }

        /// <summary>
        /// Clones data to avoid reference issues
        /// </summary>
        private double[][] CloneData(double[][] data)
        {
            var clone = new double[data.Length][];
            for (int i = 0; i < data.Length; i++)
            {
                clone[i] = new double[data[i].Length];
                Array.Copy(data[i], clone[i], data[i].Length);
            }
            return clone;
        }

        /// <summary>
        /// Updates cache statistics
        /// </summary>
        private void UpdateCacheStatistics()
        {
            Statistics.ItemsInMemory = _memoryCache.Count;
            Statistics.MemoryUsageBytes = _currentMemoryUsage;
            
            if (_enableDiskCache)
            {
                Statistics.ItemsOnDisk = Directory.GetFiles(_cacheDirectory, "*.cache").Length;
            }

            UpdateMetadata("CacheHitRate", Statistics.HitRate.ToString("P2"));
            UpdateMetadata("ItemsInMemory", Statistics.ItemsInMemory.ToString());
            UpdateMetadata("ItemsOnDisk", Statistics.ItemsOnDisk.ToString());
        }

        /// <summary>
        /// Clears all cache entries
        /// </summary>
        public void ClearCache()
        {
            lock (_cacheLock)
            {
                _memoryCache.Clear();
                _currentMemoryUsage = 0;

                if (_enableDiskCache && Directory.Exists(_cacheDirectory))
                {
                    foreach (var file in Directory.GetFiles(_cacheDirectory, "*.cache"))
                    {
                        try
                        {
                            File.Delete(file);
                        }
                        catch
                        {
                            // Ignore file deletion errors
                        }
                    }
                }

                Statistics = new CacheStatistics();
                UpdateCacheStatistics();
            }
        }

        /// <summary>
        /// Removes expired entries from cache
        /// </summary>
        public void RemoveExpiredEntries()
        {
            lock (_cacheLock)
            {
                var expiredKeys = _memoryCache
                    .Where(kvp => DateTime.UtcNow - kvp.Value.CreatedAt > _cacheExpiration)
                    .Select(kvp => kvp.Key)
                    .ToList();

                foreach (var key in expiredKeys)
                {
                    if (_memoryCache.TryGetValue(key, out var entry))
                    {
                        _currentMemoryUsage -= entry.SizeInBytes;
                        _memoryCache.Remove(key);

                        // Remove from disk if exists
                        if (entry.DiskPath != null && File.Exists(entry.DiskPath))
                        {
                            try
                            {
                                File.Delete(entry.DiskPath);
                            }
                            catch
                            {
                                // Ignore deletion errors
                            }
                        }
                    }
                }

                UpdateCacheStatistics();
            }
        }

        /// <summary>
        /// Sets a single parameter value
        /// </summary>
        protected override void SetParameter(string name, object value)
        {
            base.SetParameter(name, value);

            switch (name)
            {
                case "CacheExpiration":
                    _cacheExpiration = (TimeSpan)value;
                    break;
                case "MaxMemoryCacheSizeMB":
                    _maxMemoryCacheSizeBytes = Convert.ToInt64(value) * 1024 * 1024;
                    break;
                case "EnableDiskCache":
                    _enableDiskCache = Convert.ToBoolean(value);
                    break;
                case "EnableMemoryCache":
                    _enableMemoryCache = Convert.ToBoolean(value);
                    break;
                case "EvictionPolicy":
                    _evictionPolicy = (CacheEvictionPolicy)value;
                    break;
            }
        }

        /// <summary>
        /// Indicates whether this step requires fitting before transformation
        /// </summary>
        protected override bool RequiresFitting()
        {
            return false; // Caching doesn't require fitting
        }

        /// <summary>
        /// Gets metadata about this pipeline step
        /// </summary>
        public override Dictionary<string, string> GetMetadata()
        {
            var metadata = base.GetMetadata();
            metadata["CacheHits"] = Statistics.CacheHits.ToString();
            metadata["CacheMisses"] = Statistics.CacheMisses.ToString();
            metadata["HitRate"] = Statistics.HitRate.ToString("P2");
            metadata["MemoryUsageMB"] = (Statistics.MemoryUsageBytes / (1024.0 * 1024.0)).ToString("F2");
            return metadata;
        }
    }
}