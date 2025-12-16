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
        var (optimized, anyApplied) = optimizer.OptimizeForInference(model, cloneModel: true);

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
}
