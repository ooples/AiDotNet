namespace AiDotNet.Caching;

public class DefaultModelCache<T> : IModelCache<T>
{
    private readonly ConcurrentDictionary<string, ISymbolicModel<T>> _cache = new();

    public ISymbolicModel<T> GetCachedModel(string key)
    {
        return _cache.TryGetValue(key, out var model) ? model : new NullSymbolicModel<T>();
    }

    public void CacheModel(string key, ISymbolicModel<T> model)
    {
        _cache[key] = model;
    }

    public void ClearCache()
    {
        _cache.Clear();
    }
}