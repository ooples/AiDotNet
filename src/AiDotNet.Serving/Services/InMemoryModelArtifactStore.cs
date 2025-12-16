using System.Collections.Concurrent;

namespace AiDotNet.Serving.Services;

/// <summary>
/// In-memory store for protected model artifacts.
/// </summary>
public sealed class InMemoryModelArtifactStore : IModelArtifactStore
{
    private readonly ConcurrentDictionary<string, ProtectedModelArtifact> _artifacts = new(StringComparer.OrdinalIgnoreCase);

    public bool TryGet(string modelName, out ProtectedModelArtifact? artifact)
    {
        if (_artifacts.TryGetValue(modelName, out var value))
        {
            artifact = value;
            return true;
        }

        artifact = null;
        return false;
    }

    public ProtectedModelArtifact GetOrCreate(string modelName, Func<ProtectedModelArtifact> factory)
    {
        return _artifacts.GetOrAdd(modelName, _ => factory());
    }

    public void Remove(string modelName)
    {
        _artifacts.TryRemove(modelName, out _);
    }
}

