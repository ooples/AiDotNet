using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;

namespace AiDotNet.Serving.Tests;

internal sealed class TestModelRepository : IModelRepository
{
    private readonly Dictionary<string, ModelInfo> _models = new(StringComparer.OrdinalIgnoreCase);

    public void SetModelInfo(string modelName, ModelInfo? info)
    {
        if (info == null)
        {
            _models.Remove(modelName);
            return;
        }

        _models[modelName] = info;
    }

    public bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null)
    {
        throw new NotSupportedException("Test repository does not support loading models.");
    }

    public bool LoadModelFromRegistry<T>(
        string name,
        IServableModel<T> model,
        int registryVersion,
        string registryStage,
        string? sourcePath = null)
    {
        throw new NotSupportedException("Test repository does not support loading models.");
    }

    public IServableModel<T>? GetModel<T>(string name)
    {
        throw new NotSupportedException("Test repository does not support retrieving model instances.");
    }

    public bool UnloadModel(string name)
    {
        throw new NotSupportedException("Test repository does not support unloading models.");
    }

    public List<ModelInfo> GetAllModelInfo() => _models.Values.ToList();

    public ModelInfo? GetModelInfo(string name)
    {
        return _models.TryGetValue(name, out var info) ? info : null;
    }

    public bool ModelExists(string name) => _models.ContainsKey(name);

    public bool LoadMultimodalModel<T>(string name, IServableMultimodalModel<T> model, string? sourcePath = null)
    {
        throw new NotSupportedException("Test repository does not support loading models.");
    }

    public IServableMultimodalModel<T>? GetMultimodalModel<T>(string name)
    {
        throw new NotSupportedException("Test repository does not support retrieving model instances.");
    }
}

