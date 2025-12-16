using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Inference;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

public class InferenceOptimizerTests
{
    [Fact]
    public void InferenceOptimizer_RewritesMultiHeadAttention_ToFlashAttention_WhenEnabled()
    {
        var model = CreateTinyTransformer(taskType: NeuralNetworkTaskType.Regression);
        Assert.Contains(model.Layers, l => l is MultiHeadAttentionLayer<float>);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = true,
            AttentionMasking = AttentionMaskingMode.Disabled
        };

        var optimizer = new InferenceOptimizer<float>(config);
        // Clone relies on serialization of every layer in the graph; this test focuses on rewrite behavior.
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        Assert.Contains(optimized.Layers, l => l is FlashAttentionLayer<float>);
        Assert.DoesNotContain(optimized.Layers, l => l is MultiHeadAttentionLayer<float>);
    }

    [Fact]
    public void InferenceOptimizer_RewritesMultiHeadAttention_ToCachedAttention_ForTextGeneration_WhenKVCacheEnabled()
    {
        var model = CreateTinyTransformer(taskType: NeuralNetworkTaskType.TextGeneration);
        Assert.Contains(model.Layers, l => l is MultiHeadAttentionLayer<float>);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = true,
            EnableFlashAttention = true,
            // Paged KV-cache is industry-standard and enabled by default; keep it enabled for this test.
            AttentionMasking = AttentionMaskingMode.Auto
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: true);

        Assert.True(anyApplied);
        Assert.Contains(optimized.Layers, l => l is PagedCachedMultiHeadAttention<float>);
        Assert.DoesNotContain(optimized.Layers, l => l is MultiHeadAttentionLayer<float>);

        foreach (var layer in optimized.Layers)
        {
            if (layer is PagedCachedMultiHeadAttention<float> cached)
            {
                Assert.True(cached.InferenceMode);
                Assert.NotNull(cached.Kernel);
            }
        }
    }

    [Fact]
    public void InferenceOptimizer_RewritesSelfAttention_ToCachedAttention_WhenKVCacheEnabled()
    {
        var model = CreateTinySelfAttentionModel(taskType: NeuralNetworkTaskType.TextGeneration);
        Assert.Contains(model.Layers, l => l is SelfAttentionLayer<float>);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = true,
            EnablePagedKVCache = false,
            EnableFlashAttention = false,
            AttentionMasking = AttentionMaskingMode.Auto
        };

        var optimizer = new InferenceOptimizer<float>(config);
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        Assert.Contains(optimized.Layers, l => l is CachedMultiHeadAttention<float>);
        Assert.DoesNotContain(optimized.Layers, l => l is SelfAttentionLayer<float>);

        // In-place rewrite expected when cloneModel=false.
        Assert.DoesNotContain(model.Layers, l => l is SelfAttentionLayer<float>);
    }

    [Fact]
    public void InferenceOptimizer_SpeculativeDecoding_FallsBackToNGram_WhenSmallNeuralUnavailable()
    {
        var model = CreateTinyTransformer(taskType: NeuralNetworkTaskType.TextGeneration);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = false,
            EnableSpeculativeDecoding = true,
            DraftModelType = DraftModelType.SmallNeural
        };

        var optimizer = new InferenceOptimizer<float>(config);

        // Should never throw: SmallNeural draft models are not available in MVP and must fall back.
        var (_, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        Assert.NotNull(optimizer.DraftModel);
        Assert.Equal(DraftModelType.SmallNeural, config.DraftModelType);
        Assert.True(optimizer.DraftModel!.VocabSize > 0);
    }

    [Fact]
    public void InferenceOptimizer_SpeculativeDecoding_FallsBackToNGram_WhenCustomNotProvided()
    {
        var model = CreateTinyTransformer(taskType: NeuralNetworkTaskType.TextGeneration);

        var config = new InferenceOptimizationConfig
        {
            EnableKVCache = false,
            EnableFlashAttention = false,
            EnableSpeculativeDecoding = true,
            DraftModelType = DraftModelType.Custom
        };

        var optimizer = new InferenceOptimizer<float>(config);

        // Should never throw: the public facade does not wire custom draft models in MVP.
        var (_, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: false);

        Assert.True(anyApplied);
        Assert.NotNull(optimizer.DraftModel);
        Assert.True(optimizer.DraftModel!.VocabSize > 0);
    }

    private static Transformer<float> CreateTinyTransformer(NeuralNetworkTaskType taskType)
    {
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: taskType,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 8,
            feedForwardDimension: 16,
            complexity: NetworkComplexity.Simple,
            inputSize: 1,
            outputSize: 8,
            dropoutRate: 0.0,
            maxSequenceLength: 4,
            vocabularySize: 0,
            usePositionalEncoding: false);

        return new Transformer<float>(architecture);
    }

    private static NeuralNetworkBase<float> CreateTinySelfAttentionModel(NeuralNetworkTaskType taskType)
    {
        const int seqLen = 4;
        const int embDim = 8;
        const int headCount = 2;
        const int flatSize = seqLen * embDim;

        var layers = new System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<float>>
        {
            new InputLayer<float>(flatSize),
            new ReshapeLayer<float>(new[] { flatSize }, new[] { seqLen, embDim }),
            new SelfAttentionLayer<float>(seqLen, embDim, headCount, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>()),
            new FlattenLayer<float>(new[] { seqLen, embDim }),
            new DenseLayer<float>(flatSize, flatSize, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<float>())
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: taskType,
            complexity: NetworkComplexity.Simple,
            inputSize: flatSize,
            outputSize: flatSize,
            layers: layers);

        var model = new NeuralNetwork<float>(architecture);

        // Ensure parameters are initialized deterministically for stable tests.
        var p = model.GetParameters();
        var deterministic = new float[p.Length];
        for (int i = 0; i < deterministic.Length; i++)
        {
            deterministic[i] = ((i % 17) - 8) / 8.0f;
        }
        model.UpdateParameters(new AiDotNet.Tensors.LinearAlgebra.Vector<float>(deterministic));

        return model;
    }
}
