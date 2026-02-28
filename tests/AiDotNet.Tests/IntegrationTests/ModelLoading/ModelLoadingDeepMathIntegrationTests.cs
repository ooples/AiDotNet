using AiDotNet.ModelLoading;
using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ModelLoading;

/// <summary>
/// Deep integration tests for ModelLoading:
/// WeightMapping (direct mappings, pattern-based regex mappings, factory methods),
/// ParameterRegistry (register, lookup, validate, load),
/// WeightLoadValidation and WeightLoadResult data models.
/// </summary>
public class ModelLoadingDeepMathIntegrationTests
{
    // ============================
    // WeightMapping: Direct Mappings
    // ============================

    [Fact]
    public void WeightMapping_DirectMapping_ReturnsTarget()
    {
        var mapping = new WeightMapping();
        mapping.AddMapping("src.weight", "dst.weight");

        Assert.Equal("dst.weight", mapping.Map("src.weight"));
    }

    [Fact]
    public void WeightMapping_NoMapping_ReturnsNull()
    {
        var mapping = new WeightMapping();
        Assert.Null(mapping.Map("nonexistent.weight"));
    }

    [Fact]
    public void WeightMapping_DirectOverride_LastWins()
    {
        var mapping = new WeightMapping();
        mapping.AddMapping("src.weight", "first.weight");
        mapping.AddMapping("src.weight", "second.weight");

        Assert.Equal("second.weight", mapping.Map("src.weight"));
    }

    [Fact]
    public void WeightMapping_DirectMappingsCount_Correct()
    {
        var mapping = new WeightMapping();
        mapping.AddMapping("a", "b");
        mapping.AddMapping("c", "d");
        mapping.AddMapping("e", "f");

        Assert.Equal(3, mapping.DirectMappingCount);
    }

    [Fact]
    public void WeightMapping_ConstructorWithMappings_WorksCorrectly()
    {
        var dict = new Dictionary<string, string>
        {
            { "src1", "dst1" },
            { "src2", "dst2" }
        };
        var mapping = new WeightMapping(dict);

        Assert.Equal("dst1", mapping.Map("src1"));
        Assert.Equal("dst2", mapping.Map("src2"));
        Assert.Equal(2, mapping.DirectMappingCount);
    }

    [Fact]
    public void WeightMapping_DirectMappings_ReadOnlyDictionary()
    {
        var mapping = new WeightMapping();
        mapping.AddMapping("a", "b");

        var direct = mapping.DirectMappings;
        Assert.Equal("b", direct["a"]);
    }

    // ============================
    // WeightMapping: Pattern Mappings
    // ============================

    [Fact]
    public void WeightMapping_PatternMapping_TransformsName()
    {
        var mapping = new WeightMapping();
        mapping.AddPatternMapping(@"layer\.(\d+)\.weight", "block.$1.weight");

        Assert.Equal("block.5.weight", mapping.Map("layer.5.weight"));
    }

    [Fact]
    public void WeightMapping_PatternMapping_NoMatch_ReturnsNull()
    {
        var mapping = new WeightMapping();
        mapping.AddPatternMapping(@"layer\.(\d+)\.weight", "block.$1.weight");

        Assert.Null(mapping.Map("encoder.weight"));
    }

    [Fact]
    public void WeightMapping_PatternMapping_MultipleGroups()
    {
        var mapping = new WeightMapping();
        mapping.AddPatternMapping(@"encoder\.down\.(\d+)\.block\.(\d+)\.weight", "enc.d$1.b$2.w");

        Assert.Equal("enc.d3.b2.w", mapping.Map("encoder.down.3.block.2.weight"));
    }

    [Fact]
    public void WeightMapping_PatternMappingCount_Correct()
    {
        var mapping = new WeightMapping();
        mapping.AddPatternMapping(@"a\.(\d+)", "b.$1");
        mapping.AddPatternMapping(@"c\.(\d+)", "d.$1");

        Assert.Equal(2, mapping.PatternMappingCount);
    }

    [Fact]
    public void WeightMapping_DirectBeforePattern_DirectTakesPriority()
    {
        var mapping = new WeightMapping();
        mapping.AddMapping("layer.0.weight", "direct.result");
        mapping.AddPatternMapping(@"layer\.(\d+)\.weight", "pattern.$1.result");

        // Direct mapping should take priority over pattern
        Assert.Equal("direct.result", mapping.Map("layer.0.weight"));
    }

    [Fact]
    public void WeightMapping_PatternUsedWhenNoDirectMatch()
    {
        var mapping = new WeightMapping();
        mapping.AddMapping("layer.0.weight", "direct.result");
        mapping.AddPatternMapping(@"layer\.(\d+)\.weight", "pattern.$1.result");

        // No direct mapping for layer.5, falls through to pattern
        Assert.Equal("pattern.5.result", mapping.Map("layer.5.weight"));
    }

    // ============================
    // WeightMapping: Factory Methods
    // ============================

    [Fact]
    public void WeightMapping_StableDiffusionV1VAE_HasDirectMappings()
    {
        var mapping = WeightMapping.CreateStableDiffusionV1VAE();

        Assert.True(mapping.DirectMappingCount > 0);
        Assert.Equal("vae.inputConv.weight", mapping.Map("encoder.conv_in.weight"));
        Assert.Equal("vae.inputConv.bias", mapping.Map("encoder.conv_in.bias"));
        Assert.Equal("vae.outputConv.weight", mapping.Map("decoder.conv_out.weight"));
    }

    [Fact]
    public void WeightMapping_StableDiffusionV1VAE_HasPatternMappings()
    {
        var mapping = WeightMapping.CreateStableDiffusionV1VAE();

        Assert.True(mapping.PatternMappingCount > 0);

        // Test encoder block pattern mapping
        Assert.Equal("vae.encoder.down1.res2.norm1.gamma",
            mapping.Map("encoder.down.1.block.2.norm1.weight"));

        // Test decoder block pattern mapping
        Assert.Equal("vae.decoder.up0.res1.conv1.weight",
            mapping.Map("decoder.up.0.block.1.conv1.weight"));
    }

    [Fact]
    public void WeightMapping_StableDiffusionV1VAE_DownsampleMappings()
    {
        var mapping = WeightMapping.CreateStableDiffusionV1VAE();

        Assert.Equal("vae.encoder.down2.downsample.weight",
            mapping.Map("encoder.down.2.downsample.conv.weight"));
    }

    [Fact]
    public void WeightMapping_StableDiffusionV1UNet_TimeEmbedding()
    {
        var mapping = WeightMapping.CreateStableDiffusionV1UNet();

        Assert.Equal("unet.timeEmbed.linear1.weight", mapping.Map("time_embed.0.weight"));
        Assert.Equal("unet.timeEmbed.linear1.bias", mapping.Map("time_embed.0.bias"));
        Assert.Equal("unet.timeEmbed.linear2.weight", mapping.Map("time_embed.2.weight"));
    }

    [Fact]
    public void WeightMapping_StableDiffusionV1UNet_MiddleBlock()
    {
        var mapping = WeightMapping.CreateStableDiffusionV1UNet();

        Assert.Equal("unet.mid.block1.norm1.gamma",
            mapping.Map("middle_block.0.in_layers.0.weight"));
        Assert.Equal("unet.mid.block2.norm1.gamma",
            mapping.Map("middle_block.2.in_layers.0.weight"));
    }

    [Fact]
    public void WeightMapping_CLIPTextEncoder_EmbeddingMappings()
    {
        var mapping = WeightMapping.CreateCLIPTextEncoder();

        Assert.Equal("textEncoder.tokenEmbedding.weight",
            mapping.Map("text_model.embeddings.token_embedding.weight"));
        Assert.Equal("textEncoder.positionEmbedding.weight",
            mapping.Map("text_model.embeddings.position_embedding.weight"));
    }

    [Fact]
    public void WeightMapping_CLIPTextEncoder_AttentionPatterns()
    {
        var mapping = WeightMapping.CreateCLIPTextEncoder();

        Assert.Equal("textEncoder.layers.3.selfAttn.toQ.weight",
            mapping.Map("text_model.encoder.layers.3.self_attn.q_proj.weight"));
        Assert.Equal("textEncoder.layers.3.selfAttn.toK.weight",
            mapping.Map("text_model.encoder.layers.3.self_attn.k_proj.weight"));
    }

    [Fact]
    public void WeightMapping_CLIPTextEncoder_MLPPatterns()
    {
        var mapping = WeightMapping.CreateCLIPTextEncoder();

        Assert.Equal("textEncoder.layers.7.mlp.fc1.weight",
            mapping.Map("text_model.encoder.layers.7.mlp.fc1.weight"));
        Assert.Equal("textEncoder.layers.7.mlp.fc2.bias",
            mapping.Map("text_model.encoder.layers.7.mlp.fc2.bias"));
    }

    [Fact]
    public void WeightMapping_CLIPTextEncoder_FinalLayerNorm()
    {
        var mapping = WeightMapping.CreateCLIPTextEncoder();

        Assert.Equal("textEncoder.finalNorm.gamma",
            mapping.Map("text_model.final_layer_norm.weight"));
        Assert.Equal("textEncoder.finalNorm.beta",
            mapping.Map("text_model.final_layer_norm.bias"));
    }

    [Fact]
    public void WeightMapping_SDXLVAE_InheritsFromV1()
    {
        var sdxl = WeightMapping.CreateSDXLVAE();
        var v1 = WeightMapping.CreateStableDiffusionV1VAE();

        // SDXL VAE should have at least the same direct mappings as V1
        Assert.True(sdxl.DirectMappingCount >= v1.DirectMappingCount);

        // Same core mappings should work
        Assert.Equal("vae.inputConv.weight", sdxl.Map("encoder.conv_in.weight"));
    }

    // ============================
    // ParameterRegistry: Registration
    // ============================

    [Fact]
    public void ParameterRegistry_Empty_ZeroCount()
    {
        var registry = new ParameterRegistry<double>();
        Assert.Equal(0, registry.Count);
    }

    [Fact]
    public void ParameterRegistry_Register_IncrementsCount()
    {
        var registry = new ParameterRegistry<double>();
        Tensor<double>? stored = null;

        registry.Register("layer.weight", new[] { 3, 3 },
            () => stored,
            t => stored = t);

        Assert.Equal(1, registry.Count);
    }

    [Fact]
    public void ParameterRegistry_GetNames_ReturnsAllNames()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("a.weight", new[] { 2 }, () => null, _ => { });
        registry.Register("b.weight", new[] { 3 }, () => null, _ => { });

        var names = registry.GetNames().ToList();
        Assert.Equal(2, names.Count);
        Assert.Contains("a.weight", names);
        Assert.Contains("b.weight", names);
    }

    [Fact]
    public void ParameterRegistry_TryGet_ExistingName_ReturnsTrue()
    {
        var registry = new ParameterRegistry<double>();
        var tensor = new Tensor<double>(new[] { 2, 3 });
        registry.Register("layer.weight", new[] { 2, 3 },
            () => tensor,
            _ => { });

        Assert.True(registry.TryGet("layer.weight", out var result));
        Assert.Same(tensor, result);
    }

    [Fact]
    public void ParameterRegistry_TryGet_NonExistentName_ReturnsFalse()
    {
        var registry = new ParameterRegistry<double>();
        Assert.False(registry.TryGet("nonexistent", out _));
    }

    [Fact]
    public void ParameterRegistry_CaseInsensitiveLookup()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("Layer.Weight", new[] { 2 }, () => null, _ => { });

        Assert.True(registry.TryGet("layer.weight", out _));
        Assert.True(registry.TryGet("LAYER.WEIGHT", out _));
    }

    [Fact]
    public void ParameterRegistry_GetShape_ReturnsCorrectShape()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("conv.weight", new[] { 64, 3, 7, 7 }, () => null, _ => { });

        var shape = registry.GetShape("conv.weight");
        Assert.NotNull(shape);
        Assert.Equal(new[] { 64, 3, 7, 7 }, shape);
    }

    [Fact]
    public void ParameterRegistry_GetShape_NonExistent_ReturnsNull()
    {
        var registry = new ParameterRegistry<double>();
        Assert.Null(registry.GetShape("nonexistent"));
    }

    [Fact]
    public void ParameterRegistry_TrySet_CorrectShape_ReturnsTrue()
    {
        var registry = new ParameterRegistry<double>();
        Tensor<double>? stored = null;

        registry.Register("layer.weight", new[] { 2, 3 },
            () => stored,
            t => stored = t);

        var tensor = new Tensor<double>(new[] { 2, 3 });
        Assert.True(registry.TrySet("layer.weight", tensor));
        Assert.Same(tensor, stored);
    }

    [Fact]
    public void ParameterRegistry_TrySet_WrongShape_ThrowsArgument()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("layer.weight", new[] { 2, 3 },
            () => null,
            _ => { });

        var wrongTensor = new Tensor<double>(new[] { 4, 5 });
        Assert.Throws<ArgumentException>(() => registry.TrySet("layer.weight", wrongTensor));
    }

    [Fact]
    public void ParameterRegistry_TrySet_NonExistent_ReturnsFalse()
    {
        var registry = new ParameterRegistry<double>();
        var tensor = new Tensor<double>(new[] { 2 });

        Assert.False(registry.TrySet("nonexistent", tensor));
    }

    // ============================
    // ParameterRegistry: Child Registration
    // ============================

    [Fact]
    public void ParameterRegistry_RegisterChild_PrefixesNames()
    {
        var parent = new ParameterRegistry<double>();
        var child = new ParameterRegistry<double>();

        child.Register("weight", new[] { 3 }, () => null, _ => { });
        child.Register("bias", new[] { 3 }, () => null, _ => { });

        parent.RegisterChild("encoder.layer0", child);

        Assert.Equal(2, parent.Count);
        Assert.True(parent.TryGet("encoder.layer0.weight", out _));
        Assert.True(parent.TryGet("encoder.layer0.bias", out _));
    }

    // ============================
    // ParameterRegistry: Validate
    // ============================

    [Fact]
    public void ParameterRegistry_Validate_AllMatched()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("a.weight", new[] { 2 }, () => null, _ => { });
        registry.Register("b.weight", new[] { 3 }, () => null, _ => { });

        var result = registry.Validate(new[] { "a.weight", "b.weight" });

        Assert.Equal(2, result.Matched.Count);
        Assert.Empty(result.UnmatchedWeights);
        Assert.Empty(result.MissingParameters);
        Assert.True(result.IsComplete);
    }

    [Fact]
    public void ParameterRegistry_Validate_UnmatchedWeights()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("a.weight", new[] { 2 }, () => null, _ => { });

        var result = registry.Validate(new[] { "a.weight", "extra.weight" });

        Assert.Single(result.Matched);
        Assert.Single(result.UnmatchedWeights);
        Assert.Contains("extra.weight", result.UnmatchedWeights);
    }

    [Fact]
    public void ParameterRegistry_Validate_MissingParameters()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("a.weight", new[] { 2 }, () => null, _ => { });
        registry.Register("b.weight", new[] { 3 }, () => null, _ => { });

        var result = registry.Validate(new[] { "a.weight" });

        Assert.False(result.IsComplete);
        Assert.Single(result.MissingParameters);
        Assert.Contains("b.weight", result.MissingParameters);
    }

    [Fact]
    public void ParameterRegistry_Validate_WithMapping()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("target.weight", new[] { 2 }, () => null, _ => { });

        var result = registry.Validate(
            new[] { "source.weight" },
            name => name == "source.weight" ? "target.weight" : null);

        Assert.Single(result.Matched);
        Assert.True(result.IsComplete);
    }

    [Fact]
    public void ParameterRegistry_Validate_MappingReturnsNull_Unmatched()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("target.weight", new[] { 2 }, () => null, _ => { });

        var result = registry.Validate(
            new[] { "source.weight" },
            _ => null);

        Assert.Empty(result.Matched);
        Assert.Single(result.UnmatchedWeights);
    }

    // ============================
    // WeightLoadValidation: Computed Properties
    // ============================

    [Fact]
    public void WeightLoadValidation_IsComplete_NoMissing()
    {
        var v = new WeightLoadValidation();
        v.Matched.Add("a");
        Assert.True(v.IsComplete);
    }

    [Fact]
    public void WeightLoadValidation_IsComplete_HasMissing()
    {
        var v = new WeightLoadValidation();
        v.MissingParameters.Add("b");
        Assert.False(v.IsComplete);
    }

    [Fact]
    public void WeightLoadValidation_IsValid_NoShapeMismatches()
    {
        var v = new WeightLoadValidation();
        Assert.True(v.IsValid);
    }

    [Fact]
    public void WeightLoadValidation_IsValid_HasShapeMismatches()
    {
        var v = new WeightLoadValidation();
        v.ShapeMismatches.Add(("param", new[] { 2, 3 }, new[] { 4, 5 }));
        Assert.False(v.IsValid);
    }

    [Fact]
    public void WeightLoadValidation_ToString_IncludesAllCounts()
    {
        var v = new WeightLoadValidation();
        v.Matched.Add("a");
        v.UnmatchedWeights.Add("b");
        v.MissingParameters.Add("c");

        var str = v.ToString();
        Assert.Contains("Matched: 1", str);
        Assert.Contains("Unmatched weights: 1", str);
        Assert.Contains("Missing params: 1", str);
    }

    // ============================
    // WeightLoadResult: Properties
    // ============================

    [Fact]
    public void WeightLoadResult_Defaults_Empty()
    {
        var r = new WeightLoadResult();
        Assert.False(r.Success);
        Assert.Equal(0, r.LoadedCount);
        Assert.Equal(0, r.FailedCount);
        Assert.Equal(0, r.SkippedCount);
        Assert.Empty(r.LoadedParameters);
        Assert.Empty(r.FailedParameters);
        Assert.Null(r.ErrorMessage);
    }

    [Fact]
    public void WeightLoadResult_ToString_Success()
    {
        var r = new WeightLoadResult { Success = true, LoadedCount = 5, FailedCount = 1, SkippedCount = 2 };
        var str = r.ToString();
        Assert.Contains("Loaded: 5", str);
        Assert.Contains("Failed: 1", str);
        Assert.Contains("Skipped: 2", str);
    }

    [Fact]
    public void WeightLoadResult_ToString_Failed_ShowsError()
    {
        var r = new WeightLoadResult { Success = false, ErrorMessage = "Shape mismatch" };
        var str = r.ToString();
        Assert.Contains("Load failed", str);
        Assert.Contains("Shape mismatch", str);
    }

    // ============================
    // ParameterRegistry: Load Weights
    // ============================

    [Fact]
    public void ParameterRegistry_Load_NonStrict_SkipsUnknown()
    {
        var registry = new ParameterRegistry<double>();
        Tensor<double>? stored = null;

        registry.Register("layer.weight", new[] { 2 },
            () => stored,
            t => stored = t);

        var weights = new Dictionary<string, Tensor<double>>
        {
            { "layer.weight", new Tensor<double>(new[] { 2 }) },
            { "unknown.weight", new Tensor<double>(new[] { 3 }) }
        };

        var result = registry.Load(weights, strict: false);

        Assert.True(result.Success);
        Assert.Equal(1, result.LoadedCount);
        Assert.Equal(1, result.SkippedCount);
    }

    [Fact]
    public void ParameterRegistry_Load_Strict_FailsOnUnknown()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("layer.weight", new[] { 2 }, () => null, _ => { });

        var weights = new Dictionary<string, Tensor<double>>
        {
            { "unknown.weight", new Tensor<double>(new[] { 3 }) }
        };

        var result = registry.Load(weights, strict: true);

        Assert.False(result.Success);
        Assert.Equal(1, result.FailedCount);
    }

    [Fact]
    public void ParameterRegistry_Load_WithMapping_TranslatesNames()
    {
        var registry = new ParameterRegistry<double>();
        Tensor<double>? stored = null;

        registry.Register("target.weight", new[] { 2 },
            () => stored,
            t => stored = t);

        var weights = new Dictionary<string, Tensor<double>>
        {
            { "source.weight", new Tensor<double>(new[] { 2 }) }
        };

        var result = registry.Load(weights,
            mapping: name => name == "source.weight" ? "target.weight" : null);

        Assert.True(result.Success);
        Assert.Equal(1, result.LoadedCount);
        Assert.NotNull(stored);
    }

    [Fact]
    public void ParameterRegistry_Load_MappingReturnsNull_Skips()
    {
        var registry = new ParameterRegistry<double>();

        var weights = new Dictionary<string, Tensor<double>>
        {
            { "source.weight", new Tensor<double>(new[] { 2 }) }
        };

        var result = registry.Load(weights, mapping: _ => null);

        Assert.True(result.Success);
        Assert.Equal(0, result.LoadedCount);
        Assert.Equal(1, result.SkippedCount);
    }

    [Fact]
    public void ParameterRegistry_Load_ShapeMismatch_Strict_Fails()
    {
        var registry = new ParameterRegistry<double>();
        registry.Register("layer.weight", new[] { 2, 3 }, () => null, _ => { });

        var weights = new Dictionary<string, Tensor<double>>
        {
            { "layer.weight", new Tensor<double>(new[] { 5, 5 }) }
        };

        var result = registry.Load(weights, strict: true);

        Assert.False(result.Success);
        Assert.Equal(1, result.FailedCount);
        Assert.NotNull(result.ErrorMessage);
    }
}
