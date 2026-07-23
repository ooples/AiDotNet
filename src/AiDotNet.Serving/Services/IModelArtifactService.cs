using AiDotNet.Serving.Models;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Provides tier-aware access to model artifacts and key material.
/// </summary>
public interface IModelArtifactService
{
    string GetPlainArtifactPath(string modelName);
    ProtectedModelArtifact GetOrCreateEncryptedArtifact(string modelName);
    ModelArtifactKeyResponse CreateKeyResponse(ProtectedModelArtifact artifact);
    void RemoveProtectedArtifact(string modelName);
}
