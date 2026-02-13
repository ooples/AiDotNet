using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Unit tests for <see cref="ALiBiPositionalBiasLayer{T}"/>.
/// </summary>
public class ALiBiPositionalBiasLayerTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void Constructor_ValidHeadCount_CreatesSlopes(int numHeads)
    {
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);
        var slopes = layer.GetSlopes();

        Assert.Equal(numHeads, slopes.Length);
        Assert.Equal(numHeads, layer.NumHeads);

        // All slopes should be positive
        foreach (var slope in slopes)
        {
            Assert.True(slope > 0f);
        }
    }

    [Fact]
    public void GetSlopes_GeometricSequence()
    {
        // For 8 heads, slopes should be: 2^(-1), 2^(-2), ..., 2^(-8)
        int numHeads = 8;
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);
        var slopes = layer.GetSlopes();

        for (int h = 0; h < numHeads; h++)
        {
            double expected = Math.Pow(2.0, -8.0 / numHeads * (h + 1));
            Assert.InRange(slopes[h], (float)(expected - 0.001), (float)(expected + 0.001));
        }
    }

    [Fact]
    public void ComputeBias_CorrectShape()
    {
        int numHeads = 4;
        int queryLen = 8;
        int keyLen = 8;
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);

        var bias = layer.ComputeBias(queryLen, keyLen);

        Assert.Equal(new[] { numHeads, queryLen, keyLen }, bias.Shape);
    }

    [Fact]
    public void ComputeBias_DiagonalIsZero()
    {
        int numHeads = 4;
        int seqLen = 6;
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);

        var bias = layer.ComputeBias(seqLen, seqLen);

        // Diagonal entries (i == j) should have distance 0, so bias = 0
        for (int h = 0; h < numHeads; h++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                Assert.InRange(bias[new[] { h, i, i }], -0.001f, 0.001f);
            }
        }
    }

    [Fact]
    public void ComputeBias_FuturePositionsMasked()
    {
        int numHeads = 2;
        int seqLen = 4;
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);

        var bias = layer.ComputeBias(seqLen, seqLen);

        // Future positions (j > i for same query/key length) should be -inf
        for (int h = 0; h < numHeads; h++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = i + 1; j < seqLen; j++)
                {
                    Assert.True(float.IsNegativeInfinity(bias[new[] { h, i, j }]),
                        $"Expected -inf at head={h}, query={i}, key={j}");
                }
            }
        }
    }

    [Fact]
    public void ComputeBias_DistancePenaltyIncreases()
    {
        int numHeads = 2;
        int seqLen = 6;
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);

        var bias = layer.ComputeBias(seqLen, seqLen);

        // For non-masked positions, bias magnitude should increase with distance
        for (int h = 0; h < numHeads; h++)
        {
            // Check that bias(i, i-1) < bias(i, i-2) (more negative for larger distance)
            for (int i = 2; i < seqLen; i++)
            {
                float bias1 = bias[new[] { h, i, i - 1 }]; // distance 1
                float bias2 = bias[new[] { h, i, i - 2 }]; // distance 2
                Assert.True(bias2 < bias1,
                    $"Expected bias at distance 2 ({bias2}) < bias at distance 1 ({bias1}) for head {h}");
            }
        }
    }

    [Fact]
    public void ComputeBias_CachingWorks()
    {
        int numHeads = 4;
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);

        var bias1 = layer.ComputeBias(8, 8);
        var bias2 = layer.ComputeBias(8, 8);

        // Should return same cached instance
        Assert.Same(bias1, bias2);
    }

    [Fact]
    public void ComputeBias_DifferentSizeInvalidatesCache()
    {
        int numHeads = 4;
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);

        var bias1 = layer.ComputeBias(8, 8);
        var bias2 = layer.ComputeBias(16, 16);

        Assert.NotSame(bias1, bias2);
        Assert.Equal(new[] { numHeads, 16, 16 }, bias2.Shape);
    }

    [Fact]
    public void Forward_3D_PreservesShape()
    {
        int numHeads = 4;
        int seqLen = 8;
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);
        var input = CreateRandomTensor(new[] { numHeads, seqLen, seqLen });

        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Forward_4D_PreservesShape()
    {
        int numHeads = 4;
        int seqLen = 8;
        int batchSize = 2;
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);
        var input = CreateRandomTensor(new[] { batchSize, numHeads, seqLen, seqLen });

        var output = layer.Forward(input);

        Assert.Equal(input.Shape, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Backward_PassesGradientThrough()
    {
        int numHeads = 2;
        int seqLen = 4;
        var layer = new ALiBiPositionalBiasLayer<float>(numHeads);
        var input = CreateRandomTensor(new[] { numHeads, seqLen, seqLen });

        layer.Forward(input);
        var grad = CreateRandomTensor(new[] { numHeads, seqLen, seqLen });
        var inputGrad = layer.Backward(grad);

        // Backward should pass gradient through unchanged (constant additive bias)
        Assert.Equal(grad.Shape, inputGrad.Shape);
        var gradArr = grad.ToArray();
        var inputGradArr = inputGrad.ToArray();
        for (int i = 0; i < gradArr.Length; i++)
        {
            Assert.Equal(gradArr[i], inputGradArr[i]);
        }
    }

    [Fact]
    public void GetParameters_ReturnsEmpty()
    {
        var layer = new ALiBiPositionalBiasLayer<float>(4);
        Assert.Equal(0, layer.GetParameters().Length);
    }

    [Fact]
    public void ResetState_ClearsBiasCache()
    {
        var layer = new ALiBiPositionalBiasLayer<float>(4);
        layer.ComputeBias(8, 8);

        layer.ResetState();

        // After reset, next call should recompute (we can verify it doesn't crash)
        var bias = layer.ComputeBias(8, 8);
        Assert.NotNull(bias);
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
