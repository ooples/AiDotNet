namespace AiDotNet.Interfaces;

public interface IModelCache<T>
{
    OptimizationStepData<T>? GetCachedStepData(string key);
    void CacheStepData(string key, OptimizationStepData<T> stepData);
    void ClearCache();
}