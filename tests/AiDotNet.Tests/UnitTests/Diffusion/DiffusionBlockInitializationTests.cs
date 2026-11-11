using AiDotNet.Diffusion.Attention;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// The U-Net's blocks start as Ho et al. (2020) start them: each residual block's second conv, the attention block's
/// output projection and the output conv at zero, so every block starts as the identity and an untrained predictor
/// predicts zero noise. With full-size residuals instead, DDPM's loss went from 2 to 50 after one Adam step.
/// </summary>
public class DiffusionBlockInitializationTests
{
    private const int Channels = 32;

    [Fact]
    public void ResidualAttentionBlock_StartsAsTheIdentity()
    {
        var block = new DiffusionAttention<double>(Channels, numHeads: 4, spatialSize: 4, residualGroupNorm: true);
        var input = RandomImage(new Random(3), Channels, 4);

        var output = block.Forward(input);

        for (int i = 0; i < input.Length; i++) Assert.Equal(input[i], output[i], 12);
    }

    [Fact]
    public void ResidualAttentionBlock_CountsItsGroupNorm()
    {
        var block = new DiffusionAttention<double>(Channels, numHeads: 4, spatialSize: 4, residualGroupNorm: true);
        _ = block.Forward(RandomImage(new Random(5), Channels, 4));

        // Q, K, V, output projection and its bias, then the GroupNorm's scale and shift.
        Assert.Equal(4L * Channels * Channels + Channels + 2L * Channels, block.ParameterCount);
    }

    [Fact]
    public void ResidualAttentionBlock_ComputesAttentionOnceItsOutputProjectionTrains()
    {
        var block = new DiffusionAttention<double>(Channels, numHeads: 4, spatialSize: 4, residualGroupNorm: true);
        var input = RandomImage(new Random(7), Channels, 4);
        _ = block.Forward(input);

        // Nonzero parameters everywhere: the output now differs from the input, so the residual is not all it does.
        var parameters = block.GetParameters();
        var rng = new Random(9);
        for (int i = 0; i < parameters.Length; i++) parameters[i] = rng.NextDouble() * 0.2 - 0.1;
        block.SetParameters(parameters);
        block.ResetState();
        var output = block.Forward(input);

        double largest = 0;
        for (int i = 0; i < input.Length; i++) largest = Math.Max(largest, Math.Abs(output[i] - input[i]));
        Assert.True(largest > 1e-3, $"The attention added at most {largest:G3} to its input.");
    }

    [Fact]
    public void ResidualAttentionBlock_RejectsSequenceInput()
    {
        var block = new DiffusionAttention<double>(Channels, numHeads: 4, spatialSize: 4, residualGroupNorm: true);
        Assert.Throws<ArgumentException>(() => block.Forward(new Tensor<double>([1, 16, Channels])));
    }

    [Fact]
    public void ResidualBlock_StartsAsItsSkipPath()
    {
        var block = new DiffusionResBlock<double>(Channels, Channels, spatialSize: 4);
        var input = RandomImage(new Random(13), Channels, 4);

        var output = block.Forward(input);

        for (int i = 0; i < input.Length; i++) Assert.Equal(input[i], output[i], 12);
    }

    [Fact]
    public void UNet_PredictsZeroNoiseUntrained()
    {
        var unet = new UNetNoisePredictor<double>(inputChannels: 4, outputChannels: 4, baseChannels: 32,
            channelMultipliers: [1, 2], numResBlocks: 1, attentionResolutions: [1], contextDim: 0, numHeads: 2,
            inputHeight: 8, seed: 42);

        var noise = unet.PredictNoise(RandomImage(new Random(17), 4, 8), 500);

        for (int i = 0; i < noise.Length; i++) Assert.Equal(0.0, noise[i]);
    }

    [Fact]
    public void FlashAttentionProjections_HaveTheXavierUniformSpread()
    {
        const int width = 256;
        var layer = new FlashAttentionLayer<double>(sequenceLength: 4, embeddingDimension: width, headCount: 4);
        _ = layer.Forward(new Tensor<double>([1, 4, width]));

        // The first width x width parameters are the query projection: U(-a, a) with a = sqrt(6 / (2 width)), whose
        // standard deviation is sqrt(2 / (2 width)) = 1/16. The old fill used that standard deviation as the
        // half-width of U(-0.5, 0.5) scaled to it, a spread 2*sqrt(3) times smaller.
        var parameters = layer.GetParameters();
        double sum = 0, sumSquares = 0;
        int n = width * width;
        for (int i = 0; i < n; i++)
        {
            sum += parameters[i];
            sumSquares += parameters[i] * parameters[i];
        }

        double mean = sum / n;
        double std = Math.Sqrt(sumSquares / n - mean * mean);
        Assert.InRange(std, 0.95 / 16, 1.05 / 16);
    }

    private static Tensor<double> RandomImage(Random random, int channels, int size)
    {
        var image = new Tensor<double>([1, channels, size, size]);
        for (int i = 0; i < image.Length; i++) image[i] = random.NextDouble() * 2 - 1;
        return image;
    }
}
