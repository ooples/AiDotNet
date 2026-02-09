using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Unit tests for <see cref="GroupedQueryAttentionLayer{T}"/>.
/// </summary>
public class GroupedQueryAttentionLayerTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesLayer()
    {
        int seqLen = 16;
        int embDim = 64;
        int numHeads = 8;
        int numKVHeads = 4;

        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);

        Assert.Equal(numHeads, layer.NumHeads);
        Assert.Equal(numKVHeads, layer.NumKVHeads);
        Assert.Equal(embDim / numHeads, layer.HeadDimension);
        Assert.Equal(numHeads / numKVHeads, layer.HeadsPerGroup);
        Assert.Equal(AttentionVariant.GroupedQuery, layer.Variant);
    }

    [Fact]
    public void Variant_MHA_WhenKVHeadsEqualsHeads()
    {
        var layer = new GroupedQueryAttentionLayer<float>(16, 64, 8, 8);
        Assert.Equal(AttentionVariant.MultiHead, layer.Variant);
    }

    [Fact]
    public void Variant_MQA_WhenKVHeadsIsOne()
    {
        var layer = new GroupedQueryAttentionLayer<float>(16, 64, 8, 1);
        Assert.Equal(AttentionVariant.MultiQuery, layer.Variant);
    }

    [Fact]
    public void Variant_GQA_WhenKVHeadsBetween()
    {
        var layer = new GroupedQueryAttentionLayer<float>(16, 64, 8, 2);
        Assert.Equal(AttentionVariant.GroupedQuery, layer.Variant);
    }

    [Fact]
    public void Constructor_ThrowsWhenEmbDimNotDivisibleByHeads()
    {
        Assert.Throws<ArgumentException>(() =>
            new GroupedQueryAttentionLayer<float>(16, 63, 8, 4));
    }

    [Fact]
    public void Constructor_ThrowsWhenHeadsNotDivisibleByKVHeads()
    {
        Assert.Throws<ArgumentException>(() =>
            new GroupedQueryAttentionLayer<float>(16, 64, 8, 3));
    }

    [Theory]
    [InlineData(8, 8)]   // MHA
    [InlineData(8, 1)]   // MQA
    [InlineData(8, 2)]   // GQA (4 heads per group)
    [InlineData(8, 4)]   // GQA (2 heads per group)
    public void Forward_2D_ProducesValidOutput(int numHeads, int numKVHeads)
    {
        int seqLen = 8;
        int embDim = 64;
        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);
        var input = CreateRandomTensor(new[] { seqLen, embDim });

        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Theory]
    [InlineData(8, 8)]
    [InlineData(8, 2)]
    [InlineData(8, 1)]
    public void Forward_3D_ProducesValidOutput(int numHeads, int numKVHeads)
    {
        int batchSize = 2;
        int seqLen = 8;
        int embDim = 64;
        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);
        var input = CreateRandomTensor(new[] { batchSize, seqLen, embDim });

        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Backward_ProducesValidGradients()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        int numKVHeads = 2;
        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);
        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });

        var output = layer.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = layer.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
    }

    [Fact]
    public void ParameterCount_ReflectsReducedKVWeights()
    {
        int embDim = 64;
        int numHeads = 8;
        int headDim = embDim / numHeads; // 8

        // Standard MHA: Q, K, V each [64, 64] + O [64, 64] + bias [64]
        var mha = new GroupedQueryAttentionLayer<float>(16, embDim, numHeads, numHeads);
        int mhaParams = mha.ParameterCount;

        // GQA with 2 KV heads: Q [64, 64], K/V each [64, 16], O [64, 64], bias [64]
        var gqa = new GroupedQueryAttentionLayer<float>(16, embDim, numHeads, 2);
        int gqaParams = gqa.ParameterCount;

        // GQA should have fewer parameters than MHA
        Assert.True(gqaParams < mhaParams,
            $"GQA params ({gqaParams}) should be less than MHA params ({mhaParams})");
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        int seqLen = 8;
        int embDim = 32;
        int numHeads = 4;
        int numKVHeads = 2;
        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);

        var params1 = layer.GetParameters();
        Assert.True(params1.Length > 0);

        // Set parameters back (should not throw)
        layer.SetParameters(params1);

        var params2 = layer.GetParameters();
        Assert.Equal(params1.Length, params2.Length);
    }

    [Fact]
    public void ConfigurePositionalEncoding_RoPE_SetsProperty()
    {
        var layer = new GroupedQueryAttentionLayer<float>(16, 64, 8, 4);

        layer.ConfigurePositionalEncoding(PositionalEncodingType.Rotary);

        Assert.Equal(PositionalEncodingType.Rotary, layer.PositionalEncoding);
    }

    [Fact]
    public void ConfigurePositionalEncoding_ALiBi_SetsProperty()
    {
        var layer = new GroupedQueryAttentionLayer<float>(16, 64, 8, 4);

        layer.ConfigurePositionalEncoding(PositionalEncodingType.ALiBi);

        Assert.Equal(PositionalEncodingType.ALiBi, layer.PositionalEncoding);
    }

    [Fact]
    public void Forward_WithRoPE_ProducesValidOutput()
    {
        int seqLen = 8;
        int embDim = 64;
        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, 8, 4);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.Rotary);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Forward_WithALiBi_ProducesValidOutput()
    {
        int seqLen = 8;
        int embDim = 64;
        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, 8, 4);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.ALiBi);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void ResetState_ClearsInternalState()
    {
        var layer = new GroupedQueryAttentionLayer<float>(8, 32, 4, 2);
        var input = CreateRandomTensor(new[] { 1, 8, 32 });
        layer.Forward(input);

        layer.ResetState();
        // Should not throw and should be usable again
        var output = layer.Forward(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void GQA_WithKVHeadsEqualsHeads_MatchesMHAStructure()
    {
        // When numKVHeads == numHeads, GQA is equivalent to MHA
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;

        var gqa = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numHeads);

        // All projections should have same dimension
        var qWeights = gqa.GetQueryWeights();
        var kWeights = gqa.GetKeyWeights();
        var vWeights = gqa.GetValueWeights();

        Assert.Equal(qWeights.Shape[1], kWeights.Shape[1]);
        Assert.Equal(qWeights.Shape[1], vWeights.Shape[1]);
    }

    [Fact]
    public void GQA_ReducedKVDimensions()
    {
        int embDim = 64;
        int numHeads = 8;
        int numKVHeads = 2;
        int headDim = embDim / numHeads; // 8

        var gqa = new GroupedQueryAttentionLayer<float>(16, embDim, numHeads, numKVHeads);

        var qWeights = gqa.GetQueryWeights();
        var kWeights = gqa.GetKeyWeights();

        // Q: [embDim, numHeads * headDim] = [64, 64]
        Assert.Equal(new[] { embDim, numHeads * headDim }, qWeights.Shape);

        // K: [embDim, numKVHeads * headDim] = [64, 16]
        Assert.Equal(new[] { embDim, numKVHeads * headDim }, kWeights.Shape);
    }

    #region Helpers

    private static Tensor<float> CreateRandomTensor(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }

    private static bool ContainsNaN(Tensor<float> tensor)
    {
        foreach (var value in tensor.ToArray())
        {
            if (float.IsNaN(value)) return true;
        }
        return false;
    }

    #endregion
}
