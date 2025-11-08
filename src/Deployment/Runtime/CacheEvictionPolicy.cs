namespace AiDotNet.Deployment.Runtime;

/// <summary>
/// Cache eviction policies.
/// </summary>
public enum CacheEvictionPolicy
{
    /// <summary>Least Recently Used</summary>
    LRU,

    /// <summary>Least Frequently Used</summary>
    LFU,

    /// <summary>First In First Out</summary>
    FIFO,

    /// <summary>Random eviction</summary>
    Random
}
