using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Caches sampled episodes for reuse, reducing the cost of repeated episode generation.
/// Useful when episode sampling is expensive (e.g., loading from disk or complex preprocessing).
/// Supports LRU eviction when the cache exceeds a configured capacity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class EpisodeCache<T, TInput, TOutput>
{
    private readonly int _capacity;
    private readonly LinkedList<IEpisode<T, TInput, TOutput>> _order;
    private readonly Dictionary<int, LinkedListNode<IEpisode<T, TInput, TOutput>>> _lookup;

    /// <summary>
    /// Gets the maximum number of episodes the cache can hold.
    /// </summary>
    public int Capacity => _capacity;

    /// <summary>
    /// Gets the current number of cached episodes.
    /// </summary>
    public int Count => _lookup.Count;

    /// <summary>
    /// Gets the total number of cache hits since creation.
    /// </summary>
    public long HitCount { get; private set; }

    /// <summary>
    /// Gets the total number of cache misses since creation.
    /// </summary>
    public long MissCount { get; private set; }

    /// <summary>
    /// Creates an episode cache with the specified capacity.
    /// </summary>
    /// <param name="capacity">Maximum number of episodes to cache. Default: 1000.</param>
    public EpisodeCache(int capacity = 1000)
    {
        _capacity = Math.Max(1, capacity);
        _order = new LinkedList<IEpisode<T, TInput, TOutput>>();
        _lookup = new Dictionary<int, LinkedListNode<IEpisode<T, TInput, TOutput>>>();
    }

    /// <summary>
    /// Tries to retrieve a cached episode by its ID.
    /// </summary>
    /// <param name="episodeId">The episode ID to look up.</param>
    /// <param name="episode">The cached episode if found.</param>
    /// <returns>True if the episode was found in the cache.</returns>
    public bool TryGet(int episodeId, out IEpisode<T, TInput, TOutput>? episode)
    {
        if (_lookup.TryGetValue(episodeId, out var node))
        {
            // Move to front (most recently used)
            _order.Remove(node);
            _order.AddFirst(node);
            episode = node.Value;
            HitCount++;
            return true;
        }

        episode = null;
        MissCount++;
        return false;
    }

    /// <summary>
    /// Adds an episode to the cache. If the cache is full, the least recently used episode is evicted.
    /// </summary>
    /// <param name="episode">The episode to cache.</param>
    public void Put(IEpisode<T, TInput, TOutput> episode)
    {
        if (_lookup.ContainsKey(episode.EpisodeId))
        {
            // Already cached — move to front
            var existing = _lookup[episode.EpisodeId];
            _order.Remove(existing);
            _order.AddFirst(existing);
            return;
        }

        // Evict LRU if at capacity
        if (_lookup.Count >= _capacity)
        {
            var last = _order.Last;
            if (last != null)
            {
                _lookup.Remove(last.Value.EpisodeId);
                _order.RemoveLast();
            }
        }

        var node = _order.AddFirst(episode);
        _lookup[episode.EpisodeId] = node;
    }

    /// <summary>
    /// Adds multiple episodes to the cache.
    /// </summary>
    /// <param name="episodes">The episodes to cache.</param>
    public void PutAll(IEnumerable<IEpisode<T, TInput, TOutput>> episodes)
    {
        foreach (var ep in episodes)
        {
            Put(ep);
        }
    }

    /// <summary>
    /// Clears all cached episodes and resets hit/miss counters.
    /// </summary>
    public void Clear()
    {
        _order.Clear();
        _lookup.Clear();
        HitCount = 0;
        MissCount = 0;
    }

    /// <summary>
    /// Gets the cache hit rate as a value in [0, 1].
    /// </summary>
    public double HitRate
    {
        get
        {
            long total = HitCount + MissCount;
            return total == 0 ? 0.0 : (double)HitCount / total;
        }
    }
}
