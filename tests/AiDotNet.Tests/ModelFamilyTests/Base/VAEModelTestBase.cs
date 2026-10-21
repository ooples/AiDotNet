using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Family invariants for latent-diffusion VAEs: the contract <see cref="IVAEModel{T}"/> states about shapes, the
/// latent distribution and latent scaling (#2139).
/// </summary>
/// <remarks>
/// <para>
/// VAEs were excluded from generation because they implement <see cref="IVAEModel{T}"/>, not IDiffusionModel, so the
/// diffusion families could not construct them. Writing this base found three defects at once: two VAEs reported
/// DownsampleFactor 8 while encoding a 64x64 image to 4x4, and AutoencoderKL could not report its parameters after a
/// full encode and decode. Each invariant is a promise a latent-diffusion pipeline relies on: it sizes the denoiser's
/// latent from <c>LatentChannels</c> and <c>DownsampleFactor</c>, reads deterministic latents from the mean, and moves
/// between the VAE's and the denoiser's latent scales with <c>ScaleLatent</c>/<c>UnscaleLatent</c>.
/// </para>
/// </remarks>
public abstract class VAEModelTestBase
{
    /// <summary>Subclasses return their concrete VAE.</summary>
    protected abstract IVAEModel<double> CreateVAE();

    /// <summary>Height and width of the test image; a multiple of every VAE's DownsampleFactor.</summary>
    protected virtual int ImageSize => 64;

    private Tensor<double> CreateImage(IVAEModel<double> vae, int seed)
    {
        int size = Math.Max(ImageSize, 2 * vae.DownsampleFactor);
        var image = new Tensor<double>(new[] { 1, vae.InputChannels, size, size });
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < image.Length; i++) image[i] = rng.NextDouble() * 2 - 1;
        return image;
    }

    private static void AssertFinite(Tensor<double> tensor, string what)
    {
        for (int i = 0; i < tensor.Length; i++)
            Assert.False(double.IsNaN(tensor[i]) || double.IsInfinity(tensor[i]), $"{what}[{i}] is {tensor[i]}.");
    }

    [Fact(Timeout = 120000)]
    public async Task Encode_ProducesTheDeclaredLatentShape()
    {
        await Task.Yield();
        var vae = CreateVAE();
        var image = CreateImage(vae, 1);
        int size = image.Shape[2], f = vae.DownsampleFactor;

        var latent = vae.Encode(image, sampleMode: false);

        Assert.Equal(new[] { 1, vae.LatentChannels, size / f, size / f }, latent.Shape.ToArray());
        AssertFinite(latent, "latent");
    }

    [Fact(Timeout = 120000)]
    public async Task Decode_RestoresTheImageShape()
    {
        await Task.Yield();
        var vae = CreateVAE();
        var image = CreateImage(vae, 2);

        var reconstruction = vae.Decode(vae.Encode(image, sampleMode: false));

        Assert.Equal(image.Shape.ToArray(), reconstruction.Shape.ToArray());
        AssertFinite(reconstruction, "reconstruction");
    }

    [Fact(Timeout = 120000)]
    public async Task DeterministicEncode_IsTheDistributionMean()
    {
        await Task.Yield();
        var vae = CreateVAE();
        var image = CreateImage(vae, 3);

        var latent = vae.Encode(image, sampleMode: false);
        var (mean, logVariance) = vae.EncodeWithDistribution(image);

        Assert.Equal(mean.Shape.ToArray(), logVariance.Shape.ToArray());
        Assert.Equal(mean.Shape.ToArray(), latent.Shape.ToArray());
        for (int i = 0; i < latent.Length; i++)
            Assert.Equal(mean[i], latent[i]);
    }

    [Fact(Timeout = 120000)]
    public async Task Sample_IsReproducibleForASeed()
    {
        await Task.Yield();
        var vae = CreateVAE();
        var (mean, logVariance) = vae.EncodeWithDistribution(CreateImage(vae, 4));

        var first = vae.Sample(mean, logVariance, seed: 7);
        var second = vae.Sample(mean, logVariance, seed: 7);

        Assert.Equal(mean.Shape.ToArray(), first.Shape.ToArray());
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(first[i], second[i]);
    }

    [Fact(Timeout = 120000)]
    public async Task UnscaleLatent_InvertsScaleLatent()
    {
        await Task.Yield();
        var vae = CreateVAE();
        var latent = vae.Encode(CreateImage(vae, 5), sampleMode: false);

        var roundTrip = vae.UnscaleLatent(vae.ScaleLatent(latent));

        for (int i = 0; i < latent.Length; i++)
            Assert.True(Math.Abs(roundTrip[i] - latent[i]) <= 1e-12 * Math.Max(1.0, Math.Abs(latent[i])),
                $"Element {i}: {latent[i]} came back as {roundTrip[i]}.");
    }

    [Fact(Timeout = 120000)]
    public async Task Parameters_AreReadableAfterAForward()
    {
        await Task.Yield();
        var vae = CreateVAE();
        vae.Decode(vae.Encode(CreateImage(vae, 6), sampleMode: false));

        var parameters = vae.GetParameters();

        Assert.True(parameters.Length > 0, "A VAE has learnable parameters.");
        Assert.Equal(vae.ParameterCount, parameters.Length);
    }
}
