using AiDotNet.Enums;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// Integration tests for RoPE (Rotary Position Embedding) with attention and KV-cache.
/// </summary>
public class RoPEInferenceIntegrationTests
{
    [Fact]
    public void MHA_WithRoPE_ForwardProducesValidOutput()
    {
        int seqLen = 8;
        int embDim = 64;
        int numHeads = 4;
        var layer = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.Rotary);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void MHA_WithRoPE_DifferentFromNoPositionalEncoding()
    {
        int seqLen = 4;
        int embDim = 32;
        int numHeads = 4;

        var layerNoPos = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);
        var layerRoPE = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);
        layerRoPE.ConfigurePositionalEncoding(PositionalEncodingType.Rotary);

        // Copy parameters so only difference is RoPE
        layerRoPE.SetParameters(layerNoPos.GetParameters());

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var outputNoPos = layerNoPos.Forward(input);
        var outputRoPE = layerRoPE.Forward(input);

        // With RoPE, attention scores will be different due to rotation
        Assert.False(TensorsEqual(outputNoPos, outputRoPE),
            "RoPE should produce different output than no positional encoding");
    }

    [Fact]
    public void MHA_WithALiBi_ForwardProducesValidOutput()
    {
        int seqLen = 8;
        int embDim = 64;
        int numHeads = 4;
        var layer = new MultiHeadAttentionLayer<float>(seqLen, embDim, numHeads);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.ALiBi);

        var input = CreateRandomTensor(new[] { 1, seqLen, embDim });
        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void GQA_WithRoPE_ForwardProducesValidOutput()
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
    public void GQA_WithALiBi_ForwardProducesValidOutput()
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
    public void RoPE_RelativePositionProperty()
    {
        // RoPE encodes relative positions: shifting both Q and K by the same amount
        // should not change the attention pattern
        int headDim = 8;
        var rope = new RotaryPositionalEncodingLayer<float>(128, headDim);

        var q = CreateRandomTensor(new[] { 1, 1, 2, headDim });
        var k = CreateRandomTensor(new[] { 1, 1, 2, headDim });

        var (rotQ0, rotK0) = rope.ApplyRoPE(q, k, startPosition: 0);
        var (rotQ10, rotK10) = rope.ApplyRoPE(q, k, startPosition: 10);

        // Compute dot products Q[0] · K[1] at both offsets
        float dot0 = DotProduct(rotQ0, rotK0, 0, 1, headDim);
        float dot10 = DotProduct(rotQ10, rotK10, 0, 1, headDim);

        // The relative dot product should be the same regardless of absolute position
        // (Q and K at same relative positions get same rotation)
        Assert.InRange(MathF.Abs(dot0 - dot10), 0f, 0.01f);
    }

    [Fact]
    public void MHA_PositionalEncoding_PreservedInMetadata()
    {
        var layer = new MultiHeadAttentionLayer<float>(8, 32, 4);
        layer.ConfigurePositionalEncoding(PositionalEncodingType.Rotary);

        Assert.Equal(PositionalEncodingType.Rotary, layer.PositionalEncoding);
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

    private static bool TensorsEqual(Tensor<float> a, Tensor<float> b, float tol = 1e-6f)
    {
        if (a.Length != b.Length) return false;
        var aArr = a.ToArray();
        var bArr = b.ToArray();
        for (int i = 0; i < aArr.Length; i++)
        {
            if (MathF.Abs(aArr[i] - bArr[i]) > tol) return false;
        }
        return true;
    }

    private static float DotProduct(Tensor<float> q, Tensor<float> k, int qPos, int kPos, int headDim)
    {
        float dot = 0f;
        for (int d = 0; d < headDim; d++)
        {
            dot += q[new[] { 0, 0, qPos, d }] * k[new[] { 0, 0, kPos, d }];
        }
        return dot;
    }

    #endregion
}
