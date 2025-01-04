namespace AiDotNet.Interfaces;

public interface IModelCache<T>
{
    ISymbolicModel<T> GetCachedModel(string key);
    void CacheModel(string key, ISymbolicModel<T> model);
    void ClearCache();
}