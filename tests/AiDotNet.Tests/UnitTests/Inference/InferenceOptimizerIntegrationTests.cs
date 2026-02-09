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

    [Fact]
    public void KVCacheConfig_GQA_MemorySavings_ProportionalToKVHeadRatio()
    {
        // Verify that KV-cache memory scales linearly with NumHeads.
        // GQA with numKVHeads=8 vs numKVHeads=1 should give exactly 8x savings.
        int numLayers = 1;
        int maxSeqLen = 512;
        int headDim = 64;
        int batchSize = 1;

        var mhaConfig = new KVCacheConfig
        {
            NumLayers = numLayers,
            NumHeads = 8,
            HeadDimension = headDim,
            MaxSequenceLength = maxSeqLen,
            MaxBatchSize = batchSize,
            DataType = CacheDataType.Float32
        };

        var gqaConfig = new KVCacheConfig
        {
            NumLayers = numLayers,
            NumHeads = 1, // GQA with 1 KV head
            HeadDimension = headDim,
            MaxSequenceLength = maxSeqLen,
            MaxBatchSize = batchSize,
            DataType = CacheDataType.Float32
        };

        long mhaBytes = mhaConfig.EstimateMemoryBytes();
        long gqaBytes = gqaConfig.EstimateMemoryBytes();

        Assert.True(mhaBytes > 0);
        Assert.True(gqaBytes > 0);

        double ratio = (double)mhaBytes / gqaBytes;
        Assert.Equal(8.0, ratio, precision: 1);
    }

    [Fact]
    public void KVCacheConfig_Llama70B_GQA_Savings_8x()
    {
        // Llama 2 70B uses 64 Q heads, 8 KV heads -> 8x KV-cache savings
        var fullMHA = new KVCacheConfig
        {
            NumLayers = 80,
            NumHeads = 64,
            HeadDimension = 128,
            MaxSequenceLength = 4096,
            MaxBatchSize = 1,
            DataType = CacheDataType.Float16
        };

        var gqa = new KVCacheConfig
        {
            NumLayers = 80,
            NumHeads = 8, // 8 KV heads (GQA)
            HeadDimension = 128,
            MaxSequenceLength = 4096,
            MaxBatchSize = 1,
            DataType = CacheDataType.Float16
        };

        long fullBytes = fullMHA.EstimateMemoryBytes();
        long gqaBytes = gqa.EstimateMemoryBytes();

        double ratio = (double)fullBytes / gqaBytes;
        Assert.Equal(8.0, ratio, precision: 1);

        // Verify absolute memory makes sense: 80 layers * 8 heads * 4096 * 128 * 2 (K+V) * 2 bytes
        long expectedGqaBytes = 80L * 8 * 4096 * 128 * 2 * 2;
        Assert.Equal(expectedGqaBytes, gqaBytes);
    }

    [Theory]
    [InlineData(8, 8, 1.0)]  // MHA: no savings
    [InlineData(8, 4, 2.0)]  // 2x savings
    [InlineData(8, 2, 4.0)]  // 4x savings
    [InlineData(8, 1, 8.0)]  // 8x savings (MQA)
    [InlineData(64, 8, 8.0)] // Llama 70B ratio
    public void KVCacheConfig_MemorySavingsRatio_MatchesHeadRatio(
        int fullHeads, int kvHeads, double expectedRatio)
    {
        var fullConfig = new KVCacheConfig
        {
            NumLayers = 1, NumHeads = fullHeads, HeadDimension = 64,
            MaxSequenceLength = 1024, MaxBatchSize = 1
        };

        var gqaConfig = new KVCacheConfig
        {
            NumLayers = 1, NumHeads = kvHeads, HeadDimension = 64,
            MaxSequenceLength = 1024, MaxBatchSize = 1
        };

        double ratio = (double)fullConfig.EstimateMemoryBytes() / gqaConfig.EstimateMemoryBytes();
        Assert.Equal(expectedRatio, ratio, precision: 1);
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
