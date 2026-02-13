using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Unit tests for <see cref="RotaryPositionalEncodingLayer{T}"/>.
/// </summary>
public class RotaryPositionalEncodingLayerTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesLayer()
    {
        var layer = new RotaryPositionalEncodingLayer<float>(128, 64);

        Assert.NotNull(layer);
        Assert.False(layer.SupportsJitCompilation);
        Assert.True(layer.SupportsTraining);
    }

    [Theory]
    [InlineData(64, 32)]
    [InlineData(128, 64)]
    [InlineData(256, 128)]
    public void Forward_PreservesShape(int maxSeqLen, int headDim)
    {
        var layer = new RotaryPositionalEncodingLayer<float>(maxSeqLen, headDim);
        var input = CreateRandomTensor(new[] { 2, 4, maxSeqLen, headDim });

        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Backward_PreservesShape()
    {
        var layer = new RotaryPositionalEncodingLayer<float>(64, 32);
        var input = CreateRandomTensor(new[] { 1, 2, 8, 32 });

        var output = layer.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = layer.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void ApplyRoPE_RotatesQueriesAndKeys()
    {
        int headDim = 8;
        var layer = new RotaryPositionalEncodingLayer<float>(64, headDim);

        // [batch=1, heads=2, seq=4, headDim=8]
        var queries = CreateRandomTensor(new[] { 1, 2, 4, headDim });
        var keys = CreateRandomTensor(new[] { 1, 2, 4, headDim });

        var (rotQ, rotK) = layer.ApplyRoPE(queries, keys, startPosition: 0);

        Assert.Equal(queries.Shape, rotQ.Shape);
        Assert.Equal(keys.Shape, rotK.Shape);
        Assert.False(ContainsNaN(rotQ));
        Assert.False(ContainsNaN(rotK));

        // Rotated tensors should differ from originals (rotation changes values)
        Assert.False(TensorsEqual(queries, rotQ));
        Assert.False(TensorsEqual(keys, rotK));
    }

    [Fact]
    public void ApplyRoPE_WithStartPosition_ProducesDifferentResults()
    {
        int headDim = 8;
        var layer = new RotaryPositionalEncodingLayer<float>(64, headDim);

        var queries = CreateRandomTensor(new[] { 1, 1, 2, headDim });
        var keys = CreateRandomTensor(new[] { 1, 1, 2, headDim });

        var (rotQ0, _) = layer.ApplyRoPE(queries, keys, startPosition: 0);
        var (rotQ5, _) = layer.ApplyRoPE(queries, keys, startPosition: 5);

        // Different start positions should produce different rotations
        Assert.False(TensorsEqual(rotQ0, rotQ5));
    }

    [Fact]
    public void ApplyRoPE_CacheExtendsForLongSequences()
    {
        int headDim = 8;
        int initialMaxSeq = 16;
        var layer = new RotaryPositionalEncodingLayer<float>(initialMaxSeq, headDim);

        // Use sequence length > initialMaxSeq to trigger cache extension
        var queries = CreateRandomTensor(new[] { 1, 1, 32, headDim });
        var keys = CreateRandomTensor(new[] { 1, 1, 32, headDim });

        var (rotQ, rotK) = layer.ApplyRoPE(queries, keys, startPosition: 0);

        Assert.Equal(queries.Shape, rotQ.Shape);
        Assert.False(ContainsNaN(rotQ));
        Assert.False(ContainsNaN(rotK));
    }

    [Fact]
    public void GetParameters_ReturnsEmpty()
    {
        var layer = new RotaryPositionalEncodingLayer<float>(64, 32);
        var params1 = layer.GetParameters();

        Assert.Equal(0, params1.Length);
    }

    [Fact]
    public void ResetState_DoesNotThrow()
    {
        var layer = new RotaryPositionalEncodingLayer<float>(64, 32);
        layer.ResetState();
    }

    [Fact]
    public void RoPE_RotationFormula_CorrectForTrivialCase()
    {
        // Test that cos/sin rotation works correctly for dimension 2
        int headDim = 2;
        var layer = new RotaryPositionalEncodingLayer<float>(64, headDim);

        // Create a known input vector
        var queries = new Tensor<float>(new[] { 1, 1, 1, headDim });
        queries[new[] { 0, 0, 0, 0 }] = 1.0f; // x_0 = 1
        queries[new[] { 0, 0, 0, 1 }] = 0.0f; // x_1 = 0
        var keys = new Tensor<float>(new[] { 1, 1, 1, headDim });
        keys[new[] { 0, 0, 0, 0 }] = 1.0f;
        keys[new[] { 0, 0, 0, 1 }] = 0.0f;

        var (rotQ, _) = layer.ApplyRoPE(queries, keys, startPosition: 0);

        // At position 0, cos(0)=1, sin(0)=0, so rotation should be identity
        // x'_0 = x_0 * cos(0) - x_1 * sin(0) = 1 * 1 - 0 * 0 = 1
        // x'_1 = x_0 * sin(0) + x_1 * cos(0) = 1 * 0 + 0 * 1 = 0
        Assert.InRange(rotQ[new[] { 0, 0, 0, 0 }], 0.99f, 1.01f);
        Assert.InRange(rotQ[new[] { 0, 0, 0, 1 }], -0.01f, 0.01f);
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

    #endregion
}
