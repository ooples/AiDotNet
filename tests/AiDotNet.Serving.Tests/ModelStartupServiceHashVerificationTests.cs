using AiDotNet.Serving.Configuration;
using AiDotNet.Serving.Models;
using AiDotNet.Serving.Services;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Xunit;

namespace AiDotNet.Serving.Tests;

public sealed class ModelStartupServiceHashVerificationTests
{
    [Fact]
    public async Task StartAsync_ThrowsWhenSha256DoesNotMatch()
    {
        var baseDir = AppContext.BaseDirectory;
        var modelsDir = Path.Combine(baseDir, "models");
        Directory.CreateDirectory(modelsDir);

        var modelFile = Path.Combine(modelsDir, "dummy.model");
        await File.WriteAllTextAsync(modelFile, "not a real model");

        var options = Options.Create(new ServingOptions
        {
            ModelDirectory = "models",
            StartupModels =
            {
                new StartupModel
                {
                    Name = "dummy",
                    Path = "dummy.model",
                    NumericType = NumericType.Double,
                    Sha256 = "00"
                }
            }
        });

        var repo = new InMemoryModelRepository();
        var service = new ModelStartupService(repo, NullLogger<ModelStartupService>.Instance, options);

        var loadMethod = typeof(ModelStartupService).GetMethod(
            "LoadModelAsync",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
        Assert.NotNull(loadMethod);

        var loadTask = (Task)loadMethod!.Invoke(service, new object[] { options.Value.StartupModels[0] })!;

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => loadTask);
        Assert.Contains("SHA-256 verification", ex.Message);
    }

    private sealed class InMemoryModelRepository : IModelRepository
    {
        public bool LoadModel<T>(string name, IServableModel<T> model, string? sourcePath = null) => true;

        public IServableModel<T>? GetModel<T>(string name) => null;

        public bool UnloadModel(string name) => true;

        public List<ModelInfo> GetAllModelInfo() => new();

        public ModelInfo? GetModelInfo(string name) => null;

        public bool ModelExists(string name) => false;

        public bool LoadModelFromRegistry<T>(
            string name,
            IServableModel<T> model,
            int registryVersion,
            string registryStage,
            string? sourcePath = null)
            => true;
    }
}
