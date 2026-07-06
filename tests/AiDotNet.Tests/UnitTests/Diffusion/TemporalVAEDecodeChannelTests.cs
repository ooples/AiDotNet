using System;
using System.Threading.Tasks;
using AiDotNet.Diffusion.VAE;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Regression tests for the TemporalVAE video decode path. The decoder temporal blocks were built
/// high→low channels and applied as one monolithic block BEFORE the spatial decoder, so they ended at
/// <c>baseChannels</c> instead of <c>lastChannels</c> — the spatial decoder's first GroupNorm then rejected
/// the input with "Input has N channels but layer expects M channels" for ANY config with temporal layers
/// (<c>numTemporalLayers &gt;= 1</c>), a hard crash on every <see cref="TemporalVAE{T}.Decode"/> call.
/// (Surfaced while investigating DIAMOND #1764; the model's own latent-space Predict never decodes, so the
/// broken decode path was untested.)
/// </summary>
public class TemporalVAEDecodeChannelTests
{
    private static TemporalVAE<float> MakeVae(int numTemporalLayers) => new TemporalVAE<float>(
        inputChannels: 3, latentChannels: 16, baseChannels: 32,
        channelMultipliers: new[] { 1, 2, 4, 4 }, numTemporalLayers: numTemporalLayers,
        temporalKernelSize: 3, causalMode: true, latentScaleFactor: 0.13025);

    private static Tensor<float> MakeVideo(int seed)
    {
        var rng = new Random(seed);
        // 8x spatial downsample (2^3) → latent spatial 4 (survives without collapsing to 1x1).
        var video = new Tensor<float>(new[] { 1, 3, 4, 32, 32 });
        for (int i = 0; i < video.Length; i++) video[i] = (float)(rng.NextDouble() * 2 - 1);
        return video;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public async Task Decode_WithTemporalLayers_RoundTripsToVideoShape(int numTemporalLayers)
    {
        await Task.Yield();
        var vae = MakeVae(numTemporalLayers);
        var video = MakeVideo(1764);

        var latent = vae.Encode(video, sampleMode: false);
        // This threw "Input has 32 channels but layer expects 128 channels." before the fix.
        var decoded = vae.Decode(latent);

        // Decoder must reconstruct the full [batch, channels, frames, H, W] video shape.
        Assert.Equal(video.Shape.Length, decoded.Shape.Length);
        for (int d = 0; d < video.Shape.Length; d++)
            Assert.Equal(video.Shape[d], decoded.Shape[d]);
    }

    [Fact]
    public async Task Decode_IsDeterministic_AcrossRepeatedCalls()
    {
        await Task.Yield();
        var vae = MakeVae(numTemporalLayers: 3);
        var latent = vae.Encode(MakeVideo(7), sampleMode: false);

        var a = vae.Decode(latent);
        var b = vae.Decode(latent);

        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i], b[i]);
    }
}
