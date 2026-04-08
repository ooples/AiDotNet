using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests;

public class ModelArtifactServiceTests
{
    [Fact]
    public void CreateKeyResponse_Throws_WhenArtifactNull()
    {
        var service = CreateService(modelDirectory: "models");

        Assert.Throws<ArgumentNullException>(() => service.CreateKeyResponse(artifact: null!));
    }

    [Fact]
    public void CreateKeyResponse_ReturnsBase64KeyMaterial()
    {
        var service = CreateService(modelDirectory: "models");
        var key = Enumerable.Range(0, 32).Select(i => (byte)i).ToArray();
        var nonce = Enumerable.Range(0, 12).Select(i => (byte)(i + 10)).ToArray();
        var artifact = new ProtectedModelArtifact(
            modelName: "m",
            encryptedPath: "m.enc",
            keyId: "kid",
            key: key,
            nonce: nonce,
            algorithm: "ALG");

        var response = service.CreateKeyResponse(artifact);

        Assert.Equal("kid", response.KeyId);
        Assert.Equal("ALG", response.Algorithm);
        Assert.Equal(Convert.ToBase64String(key), response.KeyBase64);
        Assert.Equal(Convert.ToBase64String(nonce), response.NonceBase64);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void GetPlainArtifactPath_Throws_WhenModelNameMissing(string? modelName)
    {
        var service = CreateService(modelDirectory: "models");

        var ex = Assert.Throws<ArgumentException>(() => service.GetPlainArtifactPath(modelName!));

        Assert.Equal("modelName", ex.ParamName);
    }

    [Fact]
    public void GetPlainArtifactPath_Throws_WhenModelNotFound()
    {
        var service = CreateService(modelDirectory: "models");

        Assert.Throws<FileNotFoundException>(() => service.GetPlainArtifactPath("missing"));
    }

    [Fact]
    public void GetPlainArtifactPath_Throws_WhenModelHasNoSourcePath()
    {
        var repo = new TestModelRepository();
        repo.SetModelInfo("m", new ModelInfo { Name = "m", SourcePath = null });

        var service = CreateService(modelDirectory: "models", repository: repo);

        Assert.Throws<InvalidOperationException>(() => service.GetPlainArtifactPath("m"));
    }

    [Fact]
    public void GetPlainArtifactPath_Throws_WhenSourcePathOutsideConfiguredDirectory()
    {
        var workDir = CreateWorkDir();
        var modelsDir = Path.Combine(workDir, "models");
        var outsideDir = Path.Combine(workDir, "outside");
        Directory.CreateDirectory(modelsDir);
        Directory.CreateDirectory(outsideDir);

        var outsideFile = Path.Combine(outsideDir, "model.bin");
        File.WriteAllText(outsideFile, "x");

        var repo = new TestModelRepository();
        repo.SetModelInfo("m", new ModelInfo { Name = "m", SourcePath = outsideFile });

        var service = CreateService(modelDirectory: modelsDir, repository: repo);

        Assert.Throws<InvalidOperationException>(() => service.GetPlainArtifactPath("m"));
    }

    [Fact]
    public void GetPlainArtifactPath_Throws_WhenSourceFileMissing()
    {
        var workDir = CreateWorkDir();
        var modelsDir = Path.Combine(workDir, "models");
        Directory.CreateDirectory(modelsDir);

        var missingFile = Path.Combine(modelsDir, "model.bin");

        var repo = new TestModelRepository();
        repo.SetModelInfo("m", new ModelInfo { Name = "m", SourcePath = missingFile });

        var service = CreateService(modelDirectory: modelsDir, repository: repo);

        Assert.Throws<FileNotFoundException>(() => service.GetPlainArtifactPath("m"));
    }

    [Fact]
    public void GetPlainArtifactPath_ReturnsFullPath_WhenSourceFileValid()
    {
        var workDir = CreateWorkDir();
        var modelsDir = Path.Combine(workDir, "models");
        Directory.CreateDirectory(modelsDir);

        var file = Path.Combine(modelsDir, "model.bin");
        File.WriteAllText(file, "x");

        var repo = new TestModelRepository();
        repo.SetModelInfo("m", new ModelInfo { Name = "m", SourcePath = file });

        var service = CreateService(modelDirectory: modelsDir, repository: repo);

        var result = service.GetPlainArtifactPath("m");

        Assert.Equal(Path.GetFullPath(file), result);
    }

    [Fact]
    public void GetOrCreateEncryptedArtifact_UsesStoreAndCachesPerModelName()
    {
        var workDir = CreateWorkDir();
        var modelsDir = Path.Combine(workDir, "models") + Path.DirectorySeparatorChar;
        Directory.CreateDirectory(modelsDir);

        var file = Path.Combine(modelsDir, "model.bin");
        File.WriteAllText(file, "x");

        var repo = new TestModelRepository();
        repo.SetModelInfo("m", new ModelInfo { Name = "m", SourcePath = file });

        var protector = new CountingModelArtifactProtector();
        var store = new InMemoryModelArtifactStore();

        var service = CreateService(modelDirectory: modelsDir, repository: repo, protector: protector, store: store);

        var artifact1 = service.GetOrCreateEncryptedArtifact("m");
        var artifact2 = service.GetOrCreateEncryptedArtifact("m");

        Assert.Same(artifact1, artifact2);
        Assert.Equal(1, protector.CallCount);
        Assert.Equal("m", protector.LastModelName);
        Assert.Equal(Path.GetFullPath(file), protector.LastSourcePath);
        Assert.EndsWith(Path.Combine(".protected"), protector.LastOutputDirectory!, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RemoveProtectedArtifact_NoOps_WhenNameBlank()
    {
        var store = new InMemoryModelArtifactStore();
        var artifact = new ProtectedModelArtifact(
            modelName: "m",
            encryptedPath: "m.enc",
            keyId: "kid",
            key: new byte[32],
            nonce: new byte[12],
            algorithm: "ALG");

        store.GetOrCreate("m", () => artifact);

        var service = CreateService(modelDirectory: "models", store: store);

        service.RemoveProtectedArtifact("   ");

        Assert.True(store.TryGet("m", out _));
    }

    [Fact]
    public void RemoveProtectedArtifact_RemovesArtifact_WhenNameProvided()
    {
        var store = new InMemoryModelArtifactStore();
        var artifact = new ProtectedModelArtifact(
            modelName: "m",
            encryptedPath: "m.enc",
            keyId: "kid",
            key: new byte[32],
            nonce: new byte[12],
            algorithm: "ALG");

        store.GetOrCreate("m", () => artifact);

        var service = CreateService(modelDirectory: "models", store: store);

        service.RemoveProtectedArtifact("m");

        Assert.False(store.TryGet("m", out _));
    }

    private static ModelArtifactService CreateService(
        string modelDirectory,
        IModelRepository? repository = null,
        IModelArtifactProtector? protector = null,
        IModelArtifactStore? store = null)
    {
        var repo = repository ?? new TestModelRepository();
        var options = Options.Create(new ServingOptions { ModelDirectory = modelDirectory });
        var artifactProtector = protector ?? new CountingModelArtifactProtector();
        var artifactStore = store ?? new InMemoryModelArtifactStore();

        return new ModelArtifactService(repo, options, artifactProtector, artifactStore);
    }

    private static string CreateWorkDir()
    {
        var root = Path.Combine(Path.GetTempPath(), "AiDotNet.Serving.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        return root;
    }
}

