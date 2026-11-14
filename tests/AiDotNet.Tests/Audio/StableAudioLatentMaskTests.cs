using AiDotNet.Audio.StableAudio;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Stable Audio inpaints in the VAE's latent space (Evans et al. 2024): the waveform mask is carried to the latent
/// time grid and applied to every latent channel.
/// </summary>
public class StableAudioLatentMaskTests
{
    [Fact]
    public void LatentMask_DownsamplesTheWaveformMaskToTheLatentFrames()
    {
        // Eight samples on four latent frames of two samples each, two channels. Samples 4 and 5 are masked, so only
        // frame 2 is regenerated, in both channels. Reading the waveform mask element by element over the latents
        // (the old blend) would mask flat latents 4 and 5, frames 0 and 1 of the second channel.
        var mask = new Tensor<double>(new[] { 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0 }, new[] { 8 });

        var latent = StableAudioModel<double>.LatentMask(AiDotNetEngine.Current, mask, new[] { 2, 4 });

        Assert.Equal(new[] { 2, 4 }, latent.Shape.ToArray());
        Assert.Equal(new[] { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 }, latent.ToArray());
    }

    [Fact]
    public void LatentMask_MasksAFrameWhenAnyOfItsSamplesIsMasked()
    {
        var mask = new Tensor<double>(new[] { 0.0, 0.0, 0.0, 0.9, 0.0, 0.0 }, new[] { 6 });

        var latent = StableAudioModel<double>.LatentMask(AiDotNetEngine.Current, mask, new[] { 1, 2 });

        Assert.Equal(new[] { 0.0, 1.0 }, latent.ToArray());
    }
}
