namespace AiDotNet.Enums;

/// <summary>
/// Cache eviction policies for managing limited cache memory.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> When running AI models, caching helps avoid loading the same model
/// multiple times, which speeds things up. However, caches have limited space. An eviction
/// policy determines which items to remove when the cache becomes full. Think of it like
/// deciding which apps to close on your phone when memory runs low.
///
/// - **LRU (Least Recently Used)**: Removes items that haven't been used in the longest time
/// - **LFU (Least Frequently Used)**: Removes items that are used the least often
/// - **FIFO (First In First Out)**: Removes the oldest items first, like a queue
/// - **Random**: Removes items randomly (simplest but least efficient)
///
/// Most applications use LRU as it provides a good balance between performance and simplicity.
/// </remarks>
public enum CacheEvictionPolicy
{
    /// <summary>
    /// Least Recently Used - removes items that haven't been accessed in the longest time.
    /// Best for most general-purpose caching scenarios.
    /// </summary>
    LRU,

    /// <summary>
    /// Least Frequently Used - removes items that have been accessed the fewest times.
    /// Best when some items are accessed much more frequently than others.
    /// </summary>
    LFU,

    /// <summary>
    /// First In First Out - removes the oldest items first, regardless of usage.
    /// Simple but may remove frequently-used items.
    /// </summary>
    FIFO,

    /// <summary>
    /// Random eviction - removes items randomly when cache is full.
    /// Simplest implementation but least predictable performance.
    /// </summary>
    Random
}
