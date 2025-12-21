using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using Microsoft.Extensions.Options;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Implements artifact retrieval and protection for model distribution tiers.
/// </summary>
public sealed class ModelArtifactService : IModelArtifactService
{
    private readonly IModelRepository _modelRepository;
    private readonly ServingOptions _servingOptions;
    private readonly IModelArtifactProtector _protector;
    private readonly IModelArtifactStore _store;

    public ModelArtifactService(
        IModelRepository modelRepository,
        IOptions<ServingOptions> servingOptions,
        IModelArtifactProtector protector,
        IModelArtifactStore store)
    {
        _modelRepository = modelRepository ?? throw new ArgumentNullException(nameof(modelRepository));
        _servingOptions = servingOptions?.Value ?? throw new ArgumentNullException(nameof(servingOptions));
        _protector = protector ?? throw new ArgumentNullException(nameof(protector));
        _store = store ?? throw new ArgumentNullException(nameof(store));
    }

    public string GetPlainArtifactPath(string modelName)
    {
        var sourcePath = GetValidatedSourcePath(modelName);
        return sourcePath;
    }

    public ProtectedModelArtifact GetOrCreateEncryptedArtifact(string modelName)
    {
        var sourcePath = GetValidatedSourcePath(modelName);
        var modelsRoot = GetModelsRoot();
        var protectedDir = Path.Combine(modelsRoot, ".protected");

        return _store.GetOrCreate(modelName, () => _protector.ProtectToFile(modelName, sourcePath, protectedDir));
    }

    public ModelArtifactKeyResponse CreateKeyResponse(ProtectedModelArtifact artifact)
    {
        if (artifact == null)
        {
            throw new ArgumentNullException(nameof(artifact));
        }

        return new ModelArtifactKeyResponse
        {
            KeyId = artifact.KeyId,
            Algorithm = artifact.Algorithm,
            KeyBase64 = Convert.ToBase64String(artifact.Key),
            NonceBase64 = Convert.ToBase64String(artifact.Nonce)
        };
    }

    public void RemoveProtectedArtifact(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            return;
        }

        _store.Remove(modelName);
    }

    private string GetValidatedSourcePath(string modelName)
    {
        if (string.IsNullOrWhiteSpace(modelName))
        {
            throw new ArgumentException("Model name is required.", nameof(modelName));
        }

        var info = _modelRepository.GetModelInfo(modelName);
        if (info == null)
        {
            throw new FileNotFoundException($"Model '{modelName}' not found.");
        }

        if (string.IsNullOrWhiteSpace(info.SourcePath))
        {
            throw new InvalidOperationException($"Model '{modelName}' does not have a source artifact path.");
        }

        var modelsRoot = GetModelsRoot();

        var candidatePath = Path.GetFullPath(info.SourcePath);
        if (!candidatePath.StartsWith(modelsRoot, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException("Model artifact path is outside the configured model directory.");
        }

        if (!File.Exists(candidatePath))
        {
            throw new FileNotFoundException("Model artifact file not found.", candidatePath);
        }

        return candidatePath;
    }

    private string GetModelsRoot()
    {
        var modelsRoot = Path.GetFullPath(_servingOptions.ModelDirectory);
        if (!modelsRoot.EndsWith(Path.DirectorySeparatorChar.ToString()) &&
            !modelsRoot.EndsWith(Path.AltDirectorySeparatorChar.ToString()))
        {
            modelsRoot += Path.DirectorySeparatorChar;
        }

        return modelsRoot;
    }
}
