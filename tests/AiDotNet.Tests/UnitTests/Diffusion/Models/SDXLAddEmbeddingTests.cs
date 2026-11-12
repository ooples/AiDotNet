using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Diffusion.VAE;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Models;

/// <summary>
/// Reference tests for SDXL's micro-conditioning (Podell et al., 2023, section 2.2; diffusers
/// <c>_get_add_time_ids</c>, <c>Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)</c> and the
/// <c>add_embedding</c> TimestepEmbedding whose output is added to the time embedding).
/// </summary>
public class SDXLAddEmbeddingTests
{
    private static UNetNoisePredictor<double> SmallUNet() => new UNetNoisePredictor<double>(
        inputChannels: 4, outputChannels: 4,
        baseChannels: 32, channelMultipliers: [1, 2],
        numResBlocks: 1, attentionResolutions: [1],
        contextDim: 0, numHeads: 2, inputHeight: 8, seed: 3);

    private static SDXLModel<double> SmallModel(UNetNoisePredictor<double> unet) => new SDXLModel<double>(
        unet: unet,
        vae: new StandardVAE<double>(
            inputChannels: 3, latentChannels: 4,
            baseChannels: 32, channelMultipliers: [1, 2],
            numResBlocksPerLevel: 1, latentScaleFactor: 0.18215, seed: 3),
        crossAttentionDim: 12,
        seed: 3);

    [Fact]
    public void MicroCondition_UsesHeightWidthOrderOfAddTimeIds()
    {
        var model = SmallModel(SmallUNet());
        // original 640x480 (w x h), crop (top 8, left 4), target 512x256.
        var ids = model.CreateMicroCondition(640, 480, 8, 4, 512, 256);
        Assert.Equal(new double[] { 480, 640, 8, 4, 256, 512 }, ids.ToArray());
    }

    [Fact]
    public void FourierEmbedConditions_MatchesFlippedSinusoidalTimestepProjection()
    {
        var conditions = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new double[] { 1024, 3 }));
        int dim = 8;
        var features = SDXLModel<double>.FourierEmbedConditions(conditions, dim);

        Assert.Equal(new[] { 1, 16 }, features.Shape.ToArray());
        int half = dim / 2;
        for (int c = 0; c < 2; c++)
        {
            double v = conditions[0, c];
            for (int i = 0; i < half; i++)
            {
                double frequency = Math.Exp(-Math.Log(10000.0) * i / half);
                Assert.Equal(Math.Cos(v * frequency), features[0, (c * dim) + i], 10);
                Assert.Equal(Math.Sin(v * frequency), features[0, (c * dim) + half + i], 10);
            }
        }
    }

    [Fact]
    public void AdditionTimeEmbedDim_IsOneFifthOfTimeEmbedding()
    {
        Assert.Equal(256, SDXLModel<double>.AdditionTimeEmbedDim(1280));
        Assert.Equal(12, SDXLModel<double>.AdditionTimeEmbedDim(64));
    }

    [Fact]
    public void AddedTimeEmbedding_DependsOnMicroConditionsAndLeavesTextEmbeddingUnchanged()
    {
        var unet = SmallUNet();
        var model = SmallModel(unet);
        var text = new Tensor<double>(new[] { 1, 3, 12 });
        for (int i = 0; i < text.Length; i++)
            text[i] = Math.Sin(i * 0.37);
        var before = text.ToArray();

        var square = model.ComputeAddedTimeEmbedding(text, model.CreateMicroCondition(512, 512, 0, 0, 512, 512));
        var cropped = model.ComputeAddedTimeEmbedding(text, model.CreateMicroCondition(512, 512, 64, 0, 512, 512));

        Assert.Equal(before, text.ToArray());
        Assert.Equal(unet.TimeEmbeddingDim, square.Length);
        double difference = 0;
        for (int i = 0; i < square.Length; i++)
            difference += Math.Abs(square[i] - cropped[i]);
        Assert.True(difference > 0, "A different crop must change the added time embedding.");
    }

    [Fact]
    public void PoolTextEmbedding_WithoutPooledEncoder_IsTokenMean()
    {
        var model = SmallModel(SmallUNet());
        var text = new Tensor<double>(new[] { 1, 2, 12 });
        for (int i = 0; i < text.Length; i++)
            text[i] = i;
        var pooled = model.PoolTextEmbedding(text);
        for (int d = 0; d < 12; d++)
            Assert.Equal((d + (d + 12)) / 2.0, pooled[0, d], 12);
    }

    [Fact]
    public void UNet_AddedTimeEmbedding_IsAddedToProjectedTimeEmbedding()
    {
        var unet = SmallUNet();
        var latents = new Tensor<double>(new[] { 1, 4, 8, 8 });
        for (int i = 0; i < latents.Length; i++)
            latents[i] = Math.Cos(i * 0.11);

        var plain = unet.PredictNoise(latents, 500, null).ToArray();
        var zero = unet.PredictNoise(latents, 500, null, new Tensor<double>(new[] { 1, unet.TimeEmbeddingDim })).ToArray();
        var shifted = new Tensor<double>(new[] { 1, unet.TimeEmbeddingDim });
        for (int i = 0; i < shifted.Length; i++)
            shifted[i] = 0.5;
        var moved = unet.PredictNoise(latents, 500, null, shifted).ToArray();

        for (int i = 0; i < plain.Length; i++)
            Assert.Equal(plain[i], zero[i], 10);
        double difference = 0;
        for (int i = 0; i < plain.Length; i++)
            difference += Math.Abs(plain[i] - moved[i]);
        Assert.True(difference > 0, "A non-zero added embedding must change the prediction.");
    }
}
