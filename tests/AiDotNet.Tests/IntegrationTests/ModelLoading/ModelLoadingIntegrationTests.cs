using AiDotNet.ModelLoading;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ModelLoading;

/// <summary>
/// Integration tests for model loading classes.
/// </summary>
public class ModelLoadingIntegrationTests
{
    #region SafeTensorsLoader Tests

    [Fact]
    public void SafeTensorsLoader_Construction()
    {
        var loader = new SafeTensorsLoader<double>();
        Assert.NotNull(loader);
    }

    [Fact]
    public void SafeTensorsLoader_Load_NonexistentFile_Throws()
    {
        var loader = new SafeTensorsLoader<double>();
        Assert.ThrowsAny<Exception>(() => loader.Load("nonexistent_file.safetensors"));
    }

    [Fact]
    public void SafeTensorsLoader_GetTensorInfo_NonexistentFile_Throws()
    {
        var loader = new SafeTensorsLoader<double>();
        Assert.ThrowsAny<Exception>(() => loader.GetTensorInfo("nonexistent.safetensors"));
    }

    #endregion

    #region WeightMapping Tests

    [Fact]
    public void WeightMapping_Construction_WithDefaults()
    {
        var mapping = new WeightMapping();
        Assert.NotNull(mapping);
    }

    [Fact]
    public void WeightMapping_Construction_WithMappings_PreservesMappings()
    {
        var mappings = new Dictionary<string, string>
        {
            { "source.layer1.weight", "target.layer1.weight" },
            { "source.layer2.bias", "target.layer2.bias" },
        };
        var mapping = new WeightMapping(mappings);
        Assert.NotNull(mapping);

        // Verify mappings were stored correctly
        Assert.Equal("target.layer1.weight", mapping.Map("source.layer1.weight"));
        Assert.Equal("target.layer2.bias", mapping.Map("source.layer2.bias"));
    }

    [Fact]
    public void WeightMapping_AddMapping_CanBeLookedUp()
    {
        var mapping = new WeightMapping();
        mapping.AddMapping("old_name", "new_name");

        var result = mapping.Map("old_name");
        Assert.Equal("new_name", result);
    }

    [Fact]
    public void WeightMapping_Map_ReturnsNullForUnknown()
    {
        var mapping = new WeightMapping();
        var result = mapping.Map("unknown_key");
        Assert.Null(result);
    }

    [Fact]
    public void WeightMapping_AddPatternMapping_MatchesRegex()
    {
        var mapping = new WeightMapping();
        mapping.AddPatternMapping(@"model\.layers\.(\d+)\.weight", "layer_$1_weight");

        var result = mapping.Map("model.layers.3.weight");
        Assert.Equal("layer_3_weight", result);
    }

    #endregion

    #region ParameterRegistry Tests

    [Fact]
    public void ParameterRegistry_Construction()
    {
        var registry = new ParameterRegistry<double>();
        Assert.NotNull(registry);
    }

    [Fact]
    public void ParameterRegistry_Register_TracksParameters()
    {
        var registry = new ParameterRegistry<double>();
        Tensor<double>? storedTensor = null;

        registry.Register(
            "layer1.weight",
            new[] { 3, 4 },
            () => storedTensor,
            tensor => storedTensor = tensor);

        Assert.NotNull(registry);
    }

    #endregion

    #region ONNXImporter Tests

    [Fact]
    public void ONNXImporter_Construction_WithDefaults()
    {
        var importer = new ONNXImporter<double>();
        Assert.NotNull(importer);
    }

    [Fact]
    public void ONNXImporter_Construction_Verbose()
    {
        var importer = new ONNXImporter<double>(verbose: true);
        Assert.NotNull(importer);
    }

    #endregion

    #region PretrainedModelLoader Tests

    [Fact]
    public void PretrainedModelLoader_Construction_WithDefaults()
    {
        var loader = new PretrainedModelLoader<double>();
        Assert.NotNull(loader);
    }

    [Fact]
    public void PretrainedModelLoader_Construction_Verbose()
    {
        var loader = new PretrainedModelLoader<double>(verbose: true);
        Assert.NotNull(loader);
    }

    #endregion

    #region HuggingFaceModelLoader Tests

    [Fact]
    public void HuggingFaceModelLoader_Construction_WithDefaults()
    {
        var loader = new HuggingFaceModelLoader<double>();
        Assert.NotNull(loader);
    }

    [Fact]
    public void HuggingFaceModelLoader_Construction_WithCacheDir()
    {
        var tempDir = Path.GetTempPath();
        var loader = new HuggingFaceModelLoader<double>(cacheDir: tempDir);
        Assert.NotNull(loader);
    }

    #endregion
}
