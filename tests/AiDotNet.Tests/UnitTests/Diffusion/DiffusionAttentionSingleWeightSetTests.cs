using AiDotNet.Diffusion.Attention;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// DiffusionAttention used to hold a second, standard attention with its own weights for sequences shorter than a
/// threshold. Below the threshold the layer then computed with untrained weights, and the unused set counted as
/// parameters either way. One attention now serves every sequence length.
/// </summary>
public class DiffusionAttentionSingleWeightSetTests
{
    private const int Channels = 32;
    private const int Heads = 4;

    [Fact]
    public void ShortAndLongSequences_ShareOneWeightSet()
    {
        var attention = new DiffusionAttention<float>(Channels, Heads, spatialSize: 16);
        var random = new Random(7);

        // 16x16 = 256 positions and 4x4 = 16 positions: the old layer sent these to different weights.
        _ = attention.Forward(RandomImage(random, 16));
        long afterLong = attention.ParameterCount;
        _ = attention.Forward(RandomImage(random, 4));

        // Q, K, V and output projections, plus the output bias: one attention's worth.
        long expected = 4L * Channels * Channels + Channels;
        Assert.Equal(expected, afterLong);
        Assert.Equal(expected, attention.ParameterCount);
    }

    [Fact]
    public void ShortSequence_MatchesTheSameWeightsOnALongerGrid()
    {
        // A grid of one image repeated: attention over a uniform grid returns the same value at every position,
        // whatever the sequence length, as long as the same weights compute it.
        var attention = new DiffusionAttention<float>(Channels, Heads, spatialSize: 16);
        var pixel = new float[Channels];
        var random = new Random(11);
        for (int c = 0; c < Channels; c++) pixel[c] = (float)(random.NextDouble() * 2 - 1);

        var small = attention.Forward(UniformImage(pixel, 2));
        var large = attention.Forward(UniformImage(pixel, 16));

        for (int c = 0; c < Channels; c++)
            Assert.Equal(large[0, c, 0, 0], small[0, c, 0, 0], 4);
    }

    private static Tensor<float> RandomImage(Random random, int size)
    {
        var image = new Tensor<float>([1, Channels, size, size]);
        for (int i = 0; i < image.Length; i++) image[i] = (float)(random.NextDouble() * 2 - 1);
        return image;
    }

    private static Tensor<float> UniformImage(float[] pixel, int size)
    {
        var image = new Tensor<float>([1, Channels, size, size]);
        for (int c = 0; c < Channels; c++)
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    image[0, c, y, x] = pixel[c];
        return image;
    }
}
