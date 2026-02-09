using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Inference;
using AiDotNet.Inference.Quantization;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// End-to-end integration tests for InferenceOptimizer covering GQA, RoPE preservation,
/// multi-format quantization, and statistics reporting.
/// </summary>
public class InferenceOptimizerIntegrationTests
{
    [Fact]
    public void Optimizer_RewritesMHA_ToCachedMHA_PreservingRoPE()
    {
        var model = CreateMHAModel(PositionalEncodingType.Rotary);
        Assert.Contains(model.Layers, l => l is MultiHeadAttentionLayer<float>);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = true,
            EnablePagedKVCache = false,
            EnableFlashAttention = false,
            AttentionMasking = AttentionMaskingMode.Causal,
            PositionalEncoding = PositionalEncodingType.Rotary
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        Assert.Contains(optimized.Layers, l => l is CachedMultiHeadAttention<float>);
        Assert.DoesNotContain(optimized.Layers, l => l is MultiHeadAttentionLayer<float>);

        var cached = optimized.Layers.OfType<CachedMultiHeadAttention<float>>().First();
        Assert.True(cached.InferenceMode);
    }

    [Fact]
    public void Optimizer_RewritesGQA_ToCachedGQA_WithKVCache()
    {
        var model = CreateGQAModel(numHeads: 8, numKVHeads: 2);
        Assert.Contains(model.Layers, l => l is GroupedQueryAttentionLayer<float>);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = true,
            EnablePagedKVCache = false,
            EnableFlashAttention = false,
            AttentionMasking = AttentionMaskingMode.Causal
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        Assert.Contains(optimized.Layers, l => l is CachedGroupedQueryAttention<float>);
        Assert.DoesNotContain(optimized.Layers, l => l is GroupedQueryAttentionLayer<float>);

        var cachedGqa = optimized.Layers.OfType<CachedGroupedQueryAttention<float>>().First();
        Assert.True(cachedGqa.InferenceMode);
        Assert.Equal(2, cachedGqa.KVHeadCount);
    }

    [Fact]
    public void Optimizer_GQA_KVCacheUsesKVHeadCount()
    {
        var model = CreateGQAModel(numHeads: 8, numKVHeads: 2);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = true,
            EnablePagedKVCache = false,
            EnableFlashAttention = false,
            AttentionMasking = AttentionMaskingMode.Causal
        };

        var optimizer = new InferenceOptimizer<float>(config);
        optimizer.OptimizeForInference(model, cloneModel: false);

        var kvCache = optimizer.KVCache;
        Assert.NotNull(kvCache);

        // Verify the cached GQA layer has correct KV head count
        var cachedGqa = model.Layers.OfType<CachedGroupedQueryAttention<float>>().First();
        Assert.Equal(2, cachedGqa.KVHeadCount);
        Assert.Equal(8, cachedGqa.HeadCount);

        // KV cache memory should be proportionally smaller:
        // 2 KV heads vs 8 total heads = 4x smaller cache
        var cacheStats = kvCache!.GetStatistics();
        Assert.True((double)cacheStats["MaxMemoryMB"] > 0);
    }

    [Fact]
    public void Optimizer_GQA_WithRoPE_PreservesEncoding()
    {
        var model = CreateGQAModel(numHeads: 8, numKVHeads: 2, posEncoding: PositionalEncodingType.Rotary);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = true,
            EnablePagedKVCache = false,
            EnableFlashAttention = false,
            AttentionMasking = AttentionMaskingMode.Causal
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        var cachedGqa = optimized.Layers.OfType<CachedGroupedQueryAttention<float>>().First();
        Assert.True(cachedGqa.InferenceMode);
    }

    [Theory]
    [InlineData(InferenceQuantizationMode.WeightOnlyInt8)]
    [InlineData(InferenceQuantizationMode.WeightOnlyFP8)]
    [InlineData(InferenceQuantizationMode.WeightOnlyNF4)]
    public void Optimizer_QuantizesMHA_AllFormats(InferenceQuantizationMode mode)
    {
        var model = CreateMHAModel();

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = false,
            InferenceQuantization = mode
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        Assert.Contains(optimized.Layers, l => l is QuantizedAttentionLayer);
        Assert.DoesNotContain(optimized.Layers, l => l is MultiHeadAttentionLayer<float>);

        var quantized = optimized.Layers.OfType<QuantizedAttentionLayer>().First();
        Assert.Equal(mode, quantized.QuantizationFormat);
    }

    [Theory]
    [InlineData(InferenceQuantizationMode.WeightOnlyInt8)]
    [InlineData(InferenceQuantizationMode.WeightOnlyFP8)]
    [InlineData(InferenceQuantizationMode.WeightOnlyNF4)]
    public void Optimizer_QuantizesGQA_AllFormats(InferenceQuantizationMode mode)
    {
        var model = CreateGQAModel(numHeads: 4, numKVHeads: 2);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = false,
            InferenceQuantization = mode
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        Assert.Contains(optimized.Layers, l => l is QuantizedAttentionLayer);
        Assert.DoesNotContain(optimized.Layers, l => l is GroupedQueryAttentionLayer<float>);

        var quantized = optimized.Layers.OfType<QuantizedAttentionLayer>().First();
        Assert.Equal(mode, quantized.QuantizationFormat);
        Assert.True(quantized.IsGQA);
        Assert.Equal(2, quantized.KVHeadCount);
    }

    [Fact]
    public void Optimizer_QuantizationNone_DoesNotRewriteAttention()
    {
        var model = CreateMHAModel();

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = false,
            InferenceQuantization = InferenceQuantizationMode.None
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.False(anyApplied);
        Assert.Contains(optimized.Layers, l => l is MultiHeadAttentionLayer<float>);
        Assert.DoesNotContain(optimized.Layers, l => l is QuantizedAttentionLayer);
    }

    [Fact]
    public void Optimizer_Statistics_IncludeNewFields()
    {
        var model = CreateMHAModel();

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = false,
            InferenceQuantization = InferenceQuantizationMode.WeightOnlyInt8
        };

        var optimizer = new InferenceOptimizer<float>(config);
        optimizer.OptimizeForInference(model, cloneModel: false);

        var stats = optimizer.GetStatistics();
        Assert.True(stats.ContainsKey("InferenceQuantizationMode"));
        Assert.Equal("WeightOnlyInt8", stats["InferenceQuantizationMode"]);
        Assert.True(stats.ContainsKey("PositionalEncoding"));
    }

    [Fact]
    public void Optimizer_MHA_RewriteToFlash_PreservesOutputShape()
    {
        var model = CreateMHAModel();
        Assert.Contains(model.Layers, l => l is MultiHeadAttentionLayer<float>);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = true,
            AttentionMasking = AttentionMaskingMode.Disabled
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        Assert.Contains(optimized.Layers, l => l is FlashAttentionLayer<float>);
        Assert.DoesNotContain(optimized.Layers, l => l is MultiHeadAttentionLayer<float>);
    }

    [Fact]
    public void Optimizer_MixedLayers_QuantizesBothDenseAndAttention()
    {
        // Model with MHA + Dense layers
        var model = CreateMixedModel();

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = false,
            InferenceQuantization = InferenceQuantizationMode.WeightOnlyInt8
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        // Both attention and dense layers should be quantized
        Assert.Contains(optimized.Layers, l => l is QuantizedAttentionLayer);
        Assert.Contains(optimized.Layers, l => l.GetType().Name.Contains("QuantizedDenseLayer"));
    }

    #region Helpers

    private static NeuralNetworkBase<float> CreateMHAModel(
        PositionalEncodingType posEncoding = PositionalEncodingType.None)
    {
        const int seqLen = 4;
        const int embDim = 32;
        const int numHeads = 4;
        const int flatSize = seqLen * embDim;

        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads,
            activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>());
        if (posEncoding != PositionalEncodingType.None)
        {
            mha.ConfigurePositionalEncoding(posEncoding);
        }

        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(flatSize),
            new ReshapeLayer<float>(new[] { flatSize }, new[] { seqLen, embDim }),
            mha,
            new FlattenLayer<float>(new[] { seqLen, embDim }),
            new DenseLayer<float>(flatSize, flatSize,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple,
            inputSize: flatSize,
            outputSize: flatSize,
            layers: layers);

        return new NeuralNetwork<float>(architecture);
    }

    private static NeuralNetworkBase<float> CreateGQAModel(
        int numHeads = 8,
        int numKVHeads = 2,
        PositionalEncodingType posEncoding = PositionalEncodingType.None)
    {
        const int seqLen = 4;
        const int embDim = 64;
        int flatSize = seqLen * embDim;

        var gqa = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads,
            activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>());
        if (posEncoding != PositionalEncodingType.None)
        {
            gqa.ConfigurePositionalEncoding(posEncoding);
        }

        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(flatSize),
            new ReshapeLayer<float>(new[] { flatSize }, new[] { seqLen, embDim }),
            gqa,
            new FlattenLayer<float>(new[] { seqLen, embDim }),
            new DenseLayer<float>(flatSize, flatSize,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple,
            inputSize: flatSize,
            outputSize: flatSize,
            layers: layers);

        return new NeuralNetwork<float>(architecture);
    }

    private static NeuralNetworkBase<float> CreateMixedModel()
    {
        const int seqLen = 4;
        const int embDim = 32;
        const int flatSize = seqLen * embDim;

        var mha = new MultiHeadAttentionLayer<float>(seqLen, embDim, 4,
            activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>());

        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(flatSize),
            new ReshapeLayer<float>(new[] { flatSize }, new[] { seqLen, embDim }),
            mha,
            new FlattenLayer<float>(new[] { seqLen, embDim }),
            new DenseLayer<float>(flatSize, 16,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
            new DenseLayer<float>(16, flatSize,
                activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.TextGeneration,
            complexity: NetworkComplexity.Simple,
            inputSize: flatSize,
            outputSize: flatSize,
            layers: layers);

        return new NeuralNetwork<float>(architecture);
    }

    #endregion
}
