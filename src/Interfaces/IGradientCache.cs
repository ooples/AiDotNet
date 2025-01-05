namespace AiDotNet.Interfaces;

public interface IGradientCache<T>
{
    ISymbolicModel<T>? GetCachedGradient(string key);
    void CacheGradient(string key, ISymbolicModel<T> gradient);
    void ClearCache();
}