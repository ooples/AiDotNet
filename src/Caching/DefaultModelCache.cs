namespace AiDotNet.Caching;

public class DefaultModelCache<T> : IModelCache<T>
{
    private readonly ConcurrentDictionary<string, OptimizationStepData<T>> _cache = new();

    public void ClearCache()
    {
        _cache.Clear();
    }

    public OptimizationStepData<T>? GetCachedStepData(string key)
    {
        return _cache.TryGetValue(key, out var model) ? model : new();
    }

    public void CacheStepData(string key, OptimizationStepData<T> stepData)
    {
        _cache[key] = stepData;
    }
}