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
    public void ParameterRegistry_Construction_EmptyRegistry()
    {
        var registry = new ParameterRegistry<double>();
        Assert.NotNull(registry);
        Assert.Equal(0, registry.Count);
        Assert.Empty(registry.GetNames());
    }

    [Fact]
    public void ParameterRegistry_Register_IncrementsCount()
    {
        var registry = new ParameterRegistry<double>();
        Tensor<double>? storedTensor = null;

        registry.Register(
            "layer1.weight",
            new[] { 3, 4 },
            () => storedTensor,
            tensor => storedTensor = tensor);

        Assert.Equal(1, registry.Count);
        Assert.Contains("layer1.weight", registry.GetNames());
    }

    [Fact]
    public void ParameterRegistry_TryGet_ReturnsTensorFromGetter()
    {
        var registry = new ParameterRegistry<double>();
        var tensor = new Tensor<double>(new[] { 3, 4 });
        tensor[0] = 42.0;

        registry.Register(
            "layer1.weight",
            new[] { 3, 4 },
            () => tensor,
            _ => { });

        bool found = registry.TryGet("layer1.weight", out var result);
        Assert.True(found);
        Assert.NotNull(result);
        Assert.Equal(42.0, result[0]);
    }

    [Fact]
    public void ParameterRegistry_TryGet_ReturnsFalseForUnknown()
    {
        var registry = new ParameterRegistry<double>();
        bool found = registry.TryGet("nonexistent", out var result);
        Assert.False(found);
        Assert.Null(result);
    }

    [Fact]
    public void ParameterRegistry_TrySet_InvokesSetter()
    {
        var registry = new ParameterRegistry<double>();
        Tensor<double>? storedTensor = null;

        registry.Register(
            "layer1.weight",
            new[] { 2, 3 },
            () => storedTensor,
            tensor => storedTensor = tensor);

        var newTensor = new Tensor<double>(new[] { 2, 3 });
        newTensor[0] = 99.0;
        bool result = registry.TrySet("layer1.weight", newTensor);

        Assert.True(result);
        Assert.NotNull(storedTensor);
        Assert.Equal(99.0, storedTensor[0]);
    }

    [Fact]
    public void ParameterRegistry_TrySet_ShapeMismatch_ThrowsArgumentException()
    {
        var registry = new ParameterRegistry<double>();
        Tensor<double>? storedTensor = null;

        registry.Register(
            "layer1.weight",
            new[] { 2, 3 },
            () => storedTensor,
            tensor => storedTensor = tensor);

        var wrongShape = new Tensor<double>(new[] { 4, 5 });
        Assert.Throws<ArgumentException>(() => registry.TrySet("layer1.weight", wrongShape));
    }

    [Fact]
    public void ParameterRegistry_TrySet_UnknownName_ReturnsFalse()
    {
        var registry = new ParameterRegistry<double>();
        var tensor = new Tensor<double>(new[] { 2 });
        bool result = registry.TrySet("nonexistent", tensor);
        Assert.False(result);
    }

    [Fact]
    public void ParameterRegistry_GetShape_ReturnsRegisteredShape()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("param", new[] { 5, 10 }, () => null, _ => { });

        var shape = registry.GetShape("param");
        Assert.NotNull(shape);
        Assert.Equal(new[] { 5, 10 }, shape);
    }

    [Fact]
    public void ParameterRegistry_GetShape_UnknownName_ReturnsNull()
    {
        var registry = new ParameterRegistry<double>();
        Assert.Null(registry.GetShape("nonexistent"));
    }

    [Fact]
    public void ParameterRegistry_CaseInsensitive_LookupWorks()
    {
        var registry = new ParameterRegistry<double>();
        var tensor = new Tensor<double>(new[] { 3 });
        tensor[0] = 7.0;

        registry.Register("Layer1.Weight", new[] { 3 }, () => tensor, _ => { });

        // Lookup with different casing should work
        bool found = registry.TryGet("layer1.weight", out var result);
        Assert.True(found);
        Assert.NotNull(result);
        Assert.Equal(7.0, result[0]);
    }

    [Fact]
    public void ParameterRegistry_RegisterChild_PrefixesNames()
    {
        var parent = new ParameterRegistry<double>();
        var child = new ParameterRegistry<double>();

        child.Register("weight", new[] { 3 }, () => null, _ => { });
        child.Register("bias", new[] { 1 }, () => null, _ => { });

        parent.RegisterChild("encoder.layer1", child);

        Assert.Equal(2, parent.Count);
        Assert.Contains("encoder.layer1.weight", parent.GetNames());
        Assert.Contains("encoder.layer1.bias", parent.GetNames());
    }

    [Fact]
    public void ParameterRegistry_Validate_IdentifiesMatchedAndMissing()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("layer1.weight", new[] { 3 }, () => null, _ => { });
        registry.Register("layer1.bias", new[] { 1 }, () => null, _ => { });
        registry.Register("layer2.weight", new[] { 5 }, () => null, _ => { });

        // Only provide weights for layer1
        var validation = registry.Validate(new[] { "layer1.weight", "layer1.bias", "extra.param" });

        Assert.Equal(2, validation.Matched.Count);
        Assert.Single(validation.MissingParameters); // layer2.weight is missing
        Assert.Contains("layer2.weight", validation.MissingParameters);
        Assert.Single(validation.UnmatchedWeights); // extra.param has no match
        Assert.Contains("extra.param", validation.UnmatchedWeights);
    }

    [Fact]
    public void ParameterRegistry_Load_SetsParameters()
    {
        var registry = new ParameterRegistry<double>();
        Tensor<double>? storedWeight = null;
        Tensor<double>? storedBias = null;

        registry.Register("weight", new[] { 3 }, () => storedWeight, t => storedWeight = t);
        registry.Register("bias", new[] { 1 }, () => storedBias, t => storedBias = t);

        var weights = new Dictionary<string, Tensor<double>>
        {
            ["weight"] = new Tensor<double>(new[] { 3 }),
            ["bias"] = new Tensor<double>(new[] { 1 }),
        };
        weights["weight"][0] = 1.0;
        weights["weight"][1] = 2.0;
        weights["weight"][2] = 3.0;
        weights["bias"][0] = 0.5;

        var result = registry.Load(weights);

        Assert.True(result.Success);
        Assert.Equal(2, result.LoadedCount);
        Assert.NotNull(storedWeight);
        Assert.NotNull(storedBias);
        Assert.Equal(1.0, storedWeight[0]);
        Assert.Equal(0.5, storedBias[0]);
    }

    [Fact]
    public void ParameterRegistry_Load_Strict_FailsOnMissingParameter()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("weight", new[] { 3 }, () => null, _ => { });

        // Try to load a parameter that doesn't exist in registry
        var weights = new Dictionary<string, Tensor<double>>
        {
            ["nonexistent"] = new Tensor<double>(new[] { 3 }),
        };

        var result = registry.Load(weights, strict: true);
        Assert.False(result.Success);
        Assert.Equal(1, result.FailedCount);
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
