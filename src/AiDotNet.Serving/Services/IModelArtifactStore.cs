namespace AiDotNet.Serving.Services;

/// <summary>
/// Stores protected model artifacts for key release.
/// </summary>
public interface IModelArtifactStore
{
    bool TryGet(string modelName, out ProtectedModelArtifact? artifact);
    ProtectedModelArtifact GetOrCreate(string modelName, Func<ProtectedModelArtifact> factory);
    void Remove(string modelName);
}

