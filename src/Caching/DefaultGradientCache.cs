namespace AiDotNet.Caching;

public class DefaultGradientCache<T> : IGradientCache<T>
{
    private readonly ConcurrentDictionary<string, ISymbolicModel<T>> _cache = new();

    public ISymbolicModel<T>? GetCachedGradient(string key)
    {
        _cache.TryGetValue(key, out var gradient);
        return gradient;
    }

    public void CacheGradient(string key, ISymbolicModel<T> gradient)
    {
        _cache[key] = gradient;
    }

    public void ClearCache()
    {
        _cache.Clear();
    }
}