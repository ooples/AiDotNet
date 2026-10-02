using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// A latent diffusion model trains its denoiser on the VAE latent of an image, not on the image (#2157).
/// </summary>
/// <remarks>
/// Every <c>LatentDiffusionModelBase</c> subclass inherited the pixel-space default, so an image passed to
/// <c>Train</c> was noised and denoised in pixel space, while <c>Generate</c> denoises a latent and decodes it.
/// The latent model's <c>PredictNoise</c> zero-pads a 3-channel image up to the U-Net's 4 input channels, so the
/// mismatch never threw. The model-family fixtures train on latent-shaped <c>[1, 4]</c> tensors and could not see it.
/// </remarks>
public class LatentDiffusionTrainingSampleTests
{
    private const int ImageChannelCount = 3;
    private const int LatentChannelCount = 4;
    private const int ImagePixels = 8;
    private const int LatentPixels = 4; // two channel-multiplier levels downsample by 2

    /// <summary>Records the shape of every sample the training loop hands the denoiser.</summary>
    private sealed class RecordingStyDiff : StyDiffModel<double>
    {
        public List<int[]> TrainingSampleShapes { get; } = new();

        public RecordingStyDiff(int seed)
            : base(
                predictor: new UNetNoisePredictor<double>(
                    inputChannels: LatentChannelCount, outputChannels: LatentChannelCount, baseChannels: 8,
                    channelMultipliers: new[] { 1 }, numResBlocks: 1, attentionResolutions: Array.Empty<int>(),
                    contextDim: 8, numHeads: 2, inputHeight: LatentPixels, seed: seed),
                vae: new StandardVAE<double>(
                    inputChannels: ImageChannelCount, latentChannels: LatentChannelCount, baseChannels: 8,
                    channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: seed),
                seed: seed)
        {
        }

        protected override Tensor<double> PredictTrainingNoise(
            Tensor<double> noisySample, int[] timesteps, bool isBatched,
            Tensor<double> input, Tensor<double> expectedOutput)
        {
            TrainingSampleShapes.Add(noisySample.Shape.ToArray());
            return base.PredictTrainingNoise(noisySample, timesteps, isBatched, input, expectedOutput);
        }
    }

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble() * 2.0 - 1.0;
        return tensor;
    }

    private static double[] Values(Vector<double> vector) => Enumerable.Range(0, vector.Length).Select(i => vector[i]).ToArray();

    [Fact(Timeout = 120000)]
    public async Task Train_OnAnImage_DenoisesItsVaeLatent()
    {
        await Task.Yield();
        var model = new RecordingStyDiff(seed: 7);
        var images = Random(new[] { 2, ImageChannelCount, ImagePixels, ImagePixels }, 1);

        model.Train(images, images);

        Assert.NotEmpty(model.TrainingSampleShapes);
        Assert.All(model.TrainingSampleShapes, shape =>
            Assert.Equal(new[] { 2, LatentChannelCount, LatentPixels, LatentPixels }, shape));
    }

    [Fact(Timeout = 120000)]
    public async Task Train_OnAnImage_MovesTheDenoiserAndLeavesTheAutoencoderFixed()
    {
        await Task.Yield();
        var model = new RecordingStyDiff(seed: 8);
        var images = Random(new[] { 2, ImageChannelCount, ImagePixels, ImagePixels }, 2);
        model.Train(images, images);

        var denoiserBefore = Values(model.NoisePredictor.GetParameters());
        var autoencoderBefore = Values(((StandardVAE<double>)model.VAE).GetParameters());
        model.Train(images, images);
        var denoiserAfter = Values(model.NoisePredictor.GetParameters());
        var autoencoderAfter = Values(((StandardVAE<double>)model.VAE).GetParameters());

        Assert.True(denoiserBefore.Zip(denoiserAfter, (a, b) => a != b).Any(changed => changed),
            "A training step on images left every denoiser weight where it was.");
        // Rombach et al. 2022 train the denoiser on the latents of a fixed first stage.
        Assert.Equal(autoencoderBefore, autoencoderAfter);
    }

    [Fact(Timeout = 120000)]
    public async Task Train_OnALatent_UsesItAsItIs()
    {
        await Task.Yield();
        var model = new RecordingStyDiff(seed: 9);
        var latents = Random(new[] { 2, LatentChannelCount, LatentPixels, LatentPixels }, 3);

        model.Train(latents, latents);

        Assert.NotEmpty(model.TrainingSampleShapes);
        Assert.All(model.TrainingSampleShapes, shape =>
            Assert.Equal(new[] { 2, LatentChannelCount, LatentPixels, LatentPixels }, shape));
    }
}
