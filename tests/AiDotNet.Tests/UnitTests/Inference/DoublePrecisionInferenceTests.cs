using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// Tests that RoPE, ALiBi, GQA, and FlashAttention layers work with double precision.
/// All other tests in this suite use float; these verify generic T=double correctness.
/// </summary>
public class DoublePrecisionInferenceTests
{
    [Fact]
    public void RoPE_Double_ForwardProducesValidOutput()
    {
        int headDim = 8;
        var rope = new RotaryPositionalEncodingLayer<double>(64, headDim);

        var input = CreateRandomTensor(new[] { 1, 4, headDim });
        var output = rope.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void RoPE_Double_ApplyRoPEProducesValidOutput()
    {
        int headDim = 8;
        var rope = new RotaryPositionalEncodingLayer<double>(64, headDim);

        var q = CreateRandomTensor(new[] { 1, 2, 4, headDim });
        var k = CreateRandomTensor(new[] { 1, 2, 4, headDim });

        var (rotQ, rotK) = rope.ApplyRoPE(q, k, startPosition: 0);

        Assert.Equal(q.Shape, rotQ.Shape);
        Assert.Equal(k.Shape, rotK.Shape);
        Assert.False(ContainsNaN(rotQ));
        Assert.False(ContainsNaN(rotK));
    }

    [Fact]
    public void ALiBi_Double_ComputeBiasProducesValidOutput()
    {
        int numHeads = 4;
        var alibi = new ALiBiPositionalBiasLayer<double>(numHeads, 64);

        var bias = alibi.ComputeBias(8, 8);

        Assert.Equal(3, bias.Shape.Length);
        Assert.Equal(numHeads, bias.Shape[0]);
        Assert.Equal(8, bias.Shape[1]);
        Assert.Equal(8, bias.Shape[2]);
        Assert.False(ContainsNaN(bias));
    }

    [Fact]
    public void MHA_Double_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var layer = new MultiHeadAttentionLayer<double>(seqLen, embDim, numHeads);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MHA_Double_WithRoPE_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var layer = new MultiHeadAttentionLayer<double>(seqLen, embDim, numHeads);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.Rotary);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MHA_Double_WithALiBi_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var layer = new MultiHeadAttentionLayer<double>(seqLen, embDim, numHeads);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.ALiBi);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GQA_Double_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        int numKVHeads = 2;
        var layer = new GroupedQueryAttentionLayer<double>(seqLen, embDim, numHeads, numKVHeads);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GQA_Double_WithRoPE_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        int numKVHeads = 2;
        var layer = new GroupedQueryAttentionLayer<double>(seqLen, embDim, numHeads, numKVHeads);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.Rotary);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GQA_Double_Backward_ProducesValidGradients()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        int numKVHeads = 2;
        var layer = new GroupedQueryAttentionLayer<double>(seqLen, embDim, numHeads, numKVHeads);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = layer.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Theory]
    [InlineData(4, 4)] // MHA equivalent
    [InlineData(4, 1)] // MQA
    [InlineData(4, 2)] // GQA
    public void GQA_Double_AllVariants_ProduceValidOutput(int numHeads, int numKVHeads)
    {
        int seqLen = 4;
        int embDim = 32;
        var layer = new GroupedQueryAttentionLayer<double>(seqLen, embDim, numHeads, numKVHeads);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void FlashAttention_Double_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var layer = new FlashAttentionLayer<double>(seqLen, embDim, numHeads);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void FlashAttention_Double_WithRoPE_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var layer = new FlashAttentionLayer<double>(seqLen, embDim, numHeads);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.Rotary);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void FlashAttention_Double_WithALiBi_ForwardProducesValidOutput()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;
        var layer = new FlashAttentionLayer<double>(seqLen, embDim, numHeads);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.ALiBi);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    #region Helpers

    private static Tensor<double> CreateRandomTensor(int[] shape)
    {
        var tensor = new Tensor<double>(shape);
        var random = new Random(42);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = random.NextDouble() * 2 - 1;
        }
        return tensor;
    }

    private static bool ContainsNaN(Tensor<double> tensor)
    {
        foreach (var value in tensor.ToArray())
        {
            if (double.IsNaN(value)) return true;
        }
        return false;
    }

    #endregion
}
