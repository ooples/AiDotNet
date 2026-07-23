using AiDotNet.Serving.Services;

namespace AiDotNet.Serving.Tests;

internal sealed class CountingModelArtifactProtector : IModelArtifactProtector
{
    public int CallCount { get; private set; }
    public string? LastModelName { get; private set; }
    public string? LastSourcePath { get; private set; }
    public string? LastOutputDirectory { get; private set; }

    public ProtectedModelArtifact ProtectToFile(string modelName, string sourcePath, string outputDirectory)
    {
        CallCount++;
        LastModelName = modelName;
        LastSourcePath = sourcePath;
        LastOutputDirectory = outputDirectory;

        var key = Enumerable.Repeat((byte)0xAB, 32).ToArray();
        var nonce = Enumerable.Repeat((byte)0xCD, 12).ToArray();
        var encryptedPath = Path.Combine(outputDirectory, "dummy.aidn.enc");

        return new ProtectedModelArtifact(modelName, encryptedPath, keyId: "kid", key: key, nonce: nonce, algorithm: "TEST");
    }
}

