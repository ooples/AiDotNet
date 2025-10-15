using System;

namespace AiDotNet.Models.Options
{
    /// <summary>
    /// Configuration for output caching
    /// </summary>
    public class CacheConfig
    {
        /// <summary>
        /// Whether to enable output caching
        /// </summary>
        public bool EnableCache { get; set; } = true;

        /// <summary>
        /// Maximum number of cached entries
        /// </summary>
        public int MaxCacheEntries { get; set; } = 1000;

        /// <summary>
        /// Cache expiration time in seconds
        /// </summary>
        public int CacheExpirationSeconds { get; set; } = 3600;

        /// <summary>
        /// Whether to cache attention weights
        /// </summary>
        public bool CacheAttentionWeights { get; set; } = false;

        /// <summary>
        /// Validates the cache configuration
        /// </summary>
        public void Validate()
        {
            if (MaxCacheEntries <= 0)
            {
                throw new InvalidOperationException("MaxCacheEntries must be greater than 0");
            }

            if (CacheExpirationSeconds <= 0)
            {
                throw new InvalidOperationException("CacheExpirationSeconds must be greater than 0");
            }
        }
    }
}