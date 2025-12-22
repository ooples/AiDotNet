namespace AiDotNet.Serving.Services;

/// <summary>
/// Protects model artifact files for distribution.
/// </summary>
public interface IModelArtifactProtector
{
    ProtectedModelArtifact ProtectToFile(string modelName, string sourcePath, string outputDirectory);
}

