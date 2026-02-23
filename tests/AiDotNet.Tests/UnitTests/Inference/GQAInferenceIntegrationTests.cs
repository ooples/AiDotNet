using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// Integration tests for Grouped-Query Attention (GQA) inference optimizations.
/// </summary>
public class GQAInferenceIntegrationTests
{
    [Theory]
    [InlineData(8, 8)]   // MHA
    [InlineData(8, 4)]   // GQA: 2 heads per group
    [InlineData(8, 2)]   // GQA: 4 heads per group
    [InlineData(8, 1)]   // MQA
    public void GQA_AllVariants_ProduceValidOutput(int numHeads, int numKVHeads)
    {
        int seqLen = 8;
        int embDim = 64;
        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);

        var input = CreateRandomTensor(new[] { 2, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GQA_KVWeightDimensionReduced()
    {
        int embDim = 64;
        int numHeads = 8;
        int numKVHeads = 2;
        int headDim = embDim / numHeads; // 8

        var layer = new GroupedQueryAttentionLayer<float>(16, embDim, numHeads, numKVHeads);

        // Q projection: [embDim, numHeads * headDim]
        var qWeights = layer.GetQueryWeights();
        Assert.Equal(embDim, qWeights.Shape[0]);
        Assert.Equal(numHeads * headDim, qWeights.Shape[1]);

        // K/V projection: [embDim, numKVHeads * headDim] — reduced!
        var kWeights = layer.GetKeyWeights();
        var vWeights = layer.GetValueWeights();
        Assert.Equal(embDim, kWeights.Shape[0]);
        Assert.Equal(numKVHeads * headDim, kWeights.Shape[1]);
        Assert.Equal(embDim, vWeights.Shape[0]);
        Assert.Equal(numKVHeads * headDim, vWeights.Shape[1]);

        // Verify reduction: K/V should be 4x smaller than Q
        Assert.Equal(qWeights.Shape[1] / 4, kWeights.Shape[1]);
    }

    [Fact]
    public void GQA_MemorySavings_VerifyParameterCount()
    {
        int embDim = 256;
        int numHeads = 32;

        // Standard MHA: Q, K, V all [256, 256]
        var mha = new GroupedQueryAttentionLayer<float>(16, embDim, numHeads, numHeads);
        int mhaParams = mha.ParameterCount;

        // GQA with 8 KV heads: Q [256, 256], K/V [256, 64]
        var gqa = new GroupedQueryAttentionLayer<float>(16, embDim, numHeads, 8);
        int gqaParams = gqa.ParameterCount;

        // MQA with 1 KV head: Q [256, 256], K/V [256, 8]
        var mqa = new GroupedQueryAttentionLayer<float>(16, embDim, numHeads, 1);
        int mqaParams = mqa.ParameterCount;

        // GQA should be between MQA and MHA
        Assert.True(mqaParams < gqaParams, "MQA should have fewer params than GQA");
        Assert.True(gqaParams < mhaParams, "GQA should have fewer params than MHA");
    }

    [Fact]
    public void GQA_WithRoPE_ProducesValidOutput()
    {
        int seqLen = 8;
        int embDim = 64;
        int numHeads = 8;
        int numKVHeads = 2;

        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.Rotary);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GQA_WithALiBi_ProducesValidOutput()
    {
        int seqLen = 8;
        int embDim = 64;
        int numHeads = 8;
        int numKVHeads = 2;

        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.ALiBi);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GQA_Backward_ProducesGradient()
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
    public void GQA_UpdateParameters_DoesNotThrow()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        int numKVHeads = 2;

        var layer = new GroupedQueryAttentionLayer<float>(seqLen, embDim, numHeads, numKVHeads);
        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });

        layer.Forward(input);
        layer.Backward(CreateRandomTensor(new[] { 1, seqLen, embDim }));
        layer.UpdateParameters(0.01f);
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
