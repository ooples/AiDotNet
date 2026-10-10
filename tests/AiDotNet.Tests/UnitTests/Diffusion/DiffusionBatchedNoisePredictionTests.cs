using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Diffusion.NoisePredictors;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Training invariants of a latent diffusion model beside the sample-space rule in
/// <see cref="LatentDiffusionTrainingSampleTests"/>: the first stage stays fixed while the denoiser trains, and a
/// batched noise prediction equals predicting each element on its own.
/// </summary>
public class DiffusionBatchedNoisePredictionTests
{
    private const int ImageChannelCount = 3;
    private const int LatentChannelCount = 4;
    private const int ImagePixels = 8;
    private const int LatentPixels = 4;   // two channel-multiplier levels downsample by 2

    private static StyDiffModel<double> CreateModel(int seed) => new(
        predictor: new UNetNoisePredictor<double>(
            inputChannels: LatentChannelCount, outputChannels: LatentChannelCount, baseChannels: 8,
            channelMultipliers: new[] { 1 }, numResBlocks: 1, attentionResolutions: Array.Empty<int>(),
            contextDim: 8, numHeads: 2, inputHeight: LatentPixels, seed: seed),
        vae: new StandardVAE<double>(
            inputChannels: ImageChannelCount, latentChannels: LatentChannelCount, baseChannels: 8,
            channelMultipliers: new[] { 1, 2 }, numResBlocksPerLevel: 1, seed: seed),
        seed: seed);

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<double>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble() * 2.0 - 1.0;
        return tensor;
    }

    private static double[] Values(Vector<double> v)
    {
        var values = new double[v.Length];
        for (int i = 0; i < v.Length; i++) values[i] = v[i];
        return values;
    }

    [Fact(Timeout = 120000)]
    public async Task Train_OnAnImage_MovesTheDenoiserAndLeavesTheAutoencoderFixed()
    {
        await Task.Yield();
        var model = CreateModel(seed: 8);
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
    public async Task PredictNoiseBatched_MatchesPerElementPredictions()
    {
        await Task.Yield();
        var model = CreateModel(seed: 11);
        var batch = Random(new[] { 2, LatentChannelCount, LatentPixels, LatentPixels }, 5);
        int[] timesteps = { 100, 700 };

        var batched = model.PredictNoiseBatched(batch, timesteps);

        int perElement = batch.Length / 2;
        for (int b = 0; b < 2; b++)
        {
            var element = new Tensor<double>(new[] { 1, LatentChannelCount, LatentPixels, LatentPixels });
            for (int j = 0; j < perElement; j++) element[j] = batch[b * perElement + j];
            var single = model.PredictNoise(element, timesteps[b]);
            for (int j = 0; j < perElement; j++)
                Assert.Equal(single[j], batched[b * perElement + j], 12);
        }
    }
}
