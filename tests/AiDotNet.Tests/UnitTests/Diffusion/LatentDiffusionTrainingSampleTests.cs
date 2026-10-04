using System;
using AiDotNet.Diffusion.StyleTransfer;
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
        public ProbeModel() : base(seed: 42) { }

        public int[]? NoisedShape { get; private set; }

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
        var rng = new Random(seed);
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
    public void Train_OnALatent_UsesItAsItIs()
    {
        using var model = new ProbeModel();
        int factor = model.VAE.DownsampleFactor;
        var latent = Random(new[] { 1, model.LatentChannels, 64 / factor, 64 / factor }, 2);

        model.Train(latent, latent);

        Assert.Equal(latent.Shape.ToArray(), model.NoisedShape);
    }
}
