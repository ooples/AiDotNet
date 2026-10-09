using System;
using System.Collections.Generic;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Enums;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Latent diffusion trains its denoiser on z = E(x), the scaled VAE latent of an image (Rombach et al. 2022,
/// section 3.3). LatentDiffusionModelBase used to hand the training sample to the scheduler unchanged, so a
/// caller who passed images trained the denoiser on pixels while Generate ran it on latents (#2157).
/// </summary>
public class LatentDiffusionTrainingSampleTests
{
    /// <summary>Records the sample the denoiser is trained on.</summary>
    private sealed class ProbeModel : UniVSTModel<double>
    {
        // UniVST's own topology at test width: a 3 -> 4 channel VAE downsampling by eight, as the paper-scale one
        // does, and a small latent U-Net. At paper scale (U-Net width 320, VAE width 128) this class peaked at 26 GB
        // and killed the 16 GB CI runner of the shard it runs in.
        public ProbeModel()
            : base(
                predictor: new AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<double>(
                    inputChannels: 4, outputChannels: 4, baseChannels: 8, channelMultipliers: [1, 2],
                    numResBlocks: 1, attentionResolutions: [], contextDim: 16, seed: 42),
                vae: new AiDotNet.Diffusion.VAE.StandardVAE<double>(
                    inputChannels: 3, latentChannels: 4, baseChannels: 8, channelMultipliers: [1, 2, 4, 4],
                    numResBlocksPerLevel: 1, seed: 42),
                seed: 42)
        {
        }

        public int[]? NoisedShape { get; private set; }

        public List<int[]> NoisePredictionShapes { get; } = new();

        public override Tensor<double> PredictNoise(Tensor<double> noisySample, int timestep)
        {
            NoisePredictionShapes.Add(noisySample.Shape.ToArray());
            return base.PredictNoise(noisySample, timestep);
        }

        protected override Tensor<double> PredictTrainingNoise(
            Tensor<double> noisySample, int[] timesteps, bool isBatched, Tensor<double> input, Tensor<double> expectedOutput)
        {
            NoisedShape = noisySample.Shape.ToArray();
            return base.PredictTrainingNoise(noisySample, timesteps, isBatched, input, expectedOutput);
        }
    }

    private static Tensor<double> Random(int[] shape, int seed)
    {
        var t = new Tensor<double>(shape);
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < t.Length; i++) t[i] = (rng.NextDouble() * 2.0) - 1.0;
        return t;
    }

    [Fact]
    public void Train_OnAnImage_TrainsTheDenoiserOnItsLatent()
    {
        using var model = new ProbeModel();
        int channels = model.VAE.InputChannels;
        int factor = model.VAE.DownsampleFactor;
        var image = Random(new[] { 1, channels, 64, 64 }, 1);

        model.Train(image, image);

        Assert.Equal(new[] { 1, model.LatentChannels, 64 / factor, 64 / factor }, model.NoisedShape);
    }

    [Fact]
    public void ImageIsTheDefaultSampleSpace()
    {
        using var model = new ProbeModel();

        Assert.Equal(DiffusionTrainingSampleSpace.Image, model.TrainingSampleSpace);
    }

    [Fact]
    public void Train_OnALatent_UsesItAsItIs()
    {
        using var model = new ProbeModel { TrainingSampleSpace = DiffusionTrainingSampleSpace.Latent };
        int factor = model.VAE.DownsampleFactor;
        var latent = Random(new[] { 1, model.LatentChannels, 64 / factor, 64 / factor }, 2);

        model.Train(latent, latent);

        Assert.Equal(latent.Shape.ToArray(), model.NoisedShape);
    }

    /// <summary>
    /// The case a channel-count guess got wrong: image and latent of equal depth. Stated as an image, the sample
    /// is still encoded, so the denoiser trains at the latent resolution Generate samples at.
    /// </summary>
    [Fact]
    public void Train_OnAnImage_WhoseDepthEqualsTheLatentDepth_StillEncodesIt()
    {
        using var model = new ImagenProbeModel();
        Assert.Equal(model.VAE.InputChannels, model.LatentChannels);
        int factor = model.VAE.DownsampleFactor;
        Assert.True(factor > 1);
        var image = Random(new[] { 1, model.VAE.InputChannels, 64, 64 }, 3);

        model.Train(image, image);

        Assert.Equal(new[] { 1, model.LatentChannels, 64 / factor, 64 / factor }, model.NoisedShape);
    }

    [Fact]
    public void Train_OnAFlattenedLatent_UsesItAsItIs()
    {
        using var model = new ProbeModel { TrainingSampleSpace = DiffusionTrainingSampleSpace.Latent };
        int factor = model.VAE.DownsampleFactor;
        int side = 64 / factor;
        var flattened = Random(new[] { 1, model.LatentChannels * side * side }, 5);

        model.Train(flattened, flattened);

        Assert.Equal(new[] { 1, model.LatentChannels * side * side }, model.NoisedShape);
    }

    /// <summary>
    /// A flattened batch stays [B, C*H*W] for the scheduler, and the batched predictor hands each row to the denoiser on
    /// its own as [1, C*H*W], which the latent model reshapes to a latent per row.
    /// </summary>
    [Fact]
    public void Train_OnTwoFlattenedLatents_PredictsEachRowOnItsOwn()
    {
        using var model = new ProbeModel { TrainingSampleSpace = DiffusionTrainingSampleSpace.Latent };
        int side = 64 / model.VAE.DownsampleFactor;
        int width = model.LatentChannels * side * side;
        var flattened = Random(new[] { 2, width }, 7);

        model.Train(flattened, flattened);

        Assert.Equal(new[] { 2, width }, model.NoisedShape);
        Assert.Equal(2, model.NoisePredictionShapes.Count);
        Assert.All(model.NoisePredictionShapes, shape => Assert.Equal(new[] { 1, width }, shape));
    }
    [Fact]
    public void Train_WithAnUndefinedSampleSpace_IsRefused()
    {
        using var model = new ProbeModel { TrainingSampleSpace = (DiffusionTrainingSampleSpace)7 };
        var image = Random(new[] { 1, model.VAE.InputChannels, 64, 64 }, 6);

        Assert.Throws<InvalidOperationException>(() => model.Train(image, image));
    }

    [Fact]
    public void Train_OnAnImage_WithTheWrongDepth_IsRefused()
    {
        using var model = new ProbeModel();
        var notAnImage = Random(new[] { 1, model.VAE.InputChannels + 1, 64, 64 }, 4);

        var error = Assert.Throws<ArgumentException>(() => model.Train(notAnImage, notAnImage));
        Assert.Contains(nameof(DiffusionTrainingSampleSpace.Latent), error.Message);
    }

    /// <summary>Records the sample the denoiser is trained on, for a model whose image and latent depths match.</summary>
    private sealed class ImagenProbeModel : AiDotNet.Diffusion.TextToImage.ImagenModel<double>
    {
        // Imagen's own topology at test width: its pixel-depth VAE (three channels in and out, downsampling by
        // four) and a small base U-Net, so the case runs in a unit test.
        public ImagenProbeModel()
            : base(
                baseUnet: SmallUnet(), superRes1Unet: SmallUnet(),
                vae: new AiDotNet.Diffusion.VAE.StandardVAE<double>(
                    inputChannels: 3, latentChannels: 3, baseChannels: 8, channelMultipliers: [1, 2, 4],
                    numResBlocksPerLevel: 1, latentScaleFactor: 1.0, seed: 42),
                seed: 42)
        {
        }

        private static AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<double> SmallUnet() => new(
            inputChannels: 3, outputChannels: 3, baseChannels: 8, channelMultipliers: [1, 2],
            numResBlocks: 1, attentionResolutions: [], contextDim: 16, seed: 42);

        public int[]? NoisedShape { get; private set; }

        protected override Tensor<double> PredictTrainingNoise(
            Tensor<double> noisySample, int[] timesteps, bool isBatched, Tensor<double> input, Tensor<double> expectedOutput)
        {
            NoisedShape = noisySample.Shape.ToArray();
            return base.PredictTrainingNoise(noisySample, timesteps, isBatched, input, expectedOutput);
        }
    }
}
