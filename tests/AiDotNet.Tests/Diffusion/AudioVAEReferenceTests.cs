using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Diffusion.VAE;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Diffusion;

/// <summary>
/// AudioVAE reproduces AudioLDM's VAE, diffusers' AutoencoderKL in AudioLDM's configuration
/// (<c>tools/reference-data/audioldm_vae_reference.py</c>): with the reference's weights loaded through
/// <see cref="AudioVAE{T}.LoadDiffusersWeights"/>, a tiny configuration gives the reference's posterior moments and
/// decoded spectrograms.
/// </summary>
/// <remarks>
/// The previous model was a 1D network over mel bins as channels, with a time downsample its hand-written loops did not
/// learn: it could not load this state dict and its latents did not fit AudioLDM's 2D UNet.
/// </remarks>
public class AudioVAEReferenceTests
{
    private const double Tolerance = 1e-9;

    private static JsonElement Fixture()
    {
        const string name = "audioldm_vae_reference.json";
        string relative = Path.Combine("Diffusion", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static AudioVAE<double> LoadReference(JsonElement f, out IReadOnlyCollection<string> read)
    {
        var config = f.GetProperty("config");
        var vae = new AudioVAE<double>(
            inputChannels: config.GetProperty("in_channels").GetInt32(),
            latentChannels: config.GetProperty("latent_channels").GetInt32(),
            blockOutChannels: Ints(config.GetProperty("block_out_channels")),
            layersPerBlock: config.GetProperty("layers_per_block").GetInt32(),
            normGroups: config.GetProperty("norm_num_groups").GetInt32(),
            melBins: 8,
            seed: 1);
        var state = f.GetProperty("state_dict");
        var shapes = f.GetProperty("shapes");
        read = vae.LoadDiffusersWeights((name, shape) =>
        {
            Assert.True(state.TryGetProperty(name, out var values), $"The reference has no '{name}'.");
            Assert.Equal(Ints(shapes.GetProperty(name)), shape);
            return Doubles(values);
        });
        return vae;
    }

    private static void Close(double[] expected, Tensor<double> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= Tolerance,
                $"{what}[{i}]: {actual.GetFlat(i):R} vs the reference's {expected[i]:R}");
    }

    [Fact(Timeout = 120000)]
    public async Task Every_reference_weight_has_a_place_and_every_weight_is_read()
    {
        await Task.Yield();
        var f = Fixture();
        using var vae = LoadReference(f, out var read);

        var expected = f.GetProperty("state_dict").EnumerateObject().Select(p => p.Name).OrderBy(n => n, StringComparer.Ordinal);
        Assert.Equal(expected, read.OrderBy(n => n, StringComparer.Ordinal));
        Assert.Equal(f.GetProperty("state_dict").EnumerateObject().Sum(p => p.Value.GetArrayLength()), vae.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task Encode_and_decode_match_the_reference()
    {
        await Task.Yield();
        var f = Fixture();
        using var vae = LoadReference(f, out _);
        var input = new Tensor<double>(Ints(f.GetProperty("input_shape")), new Vector<double>(Doubles(f.GetProperty("input"))));

        var (mean, logVariance) = vae.EncodeWithDistribution(input);
        var decoded = vae.Decode(mean);

        // moments = [mean | log-variance] along channels, per batch item.
        var momentShape = Ints(f.GetProperty("moments_shape"));
        var moments = Doubles(f.GetProperty("moments"));
        int latent = vae.LatentChannels, plane = momentShape[2] * momentShape[3];
        var expectedMean = new List<double>();
        var expectedLogVariance = new List<double>();
        for (int b = 0; b < momentShape[0]; b++)
        {
            int offset = b * 2 * latent * plane;
            expectedMean.AddRange(moments.Skip(offset).Take(latent * plane));
            expectedLogVariance.AddRange(moments.Skip(offset + latent * plane).Take(latent * plane));
        }
        Assert.Equal(new[] { momentShape[0], latent, momentShape[2], momentShape[3] }, mean.Shape.ToArray());
        Close(expectedMean.ToArray(), mean, "mean");
        Close(expectedLogVariance.ToArray(), logVariance, "log-variance");
        Assert.Equal(Ints(f.GetProperty("decoded_shape")), decoded.Shape.ToArray());
        Close(Doubles(f.GetProperty("decoded")), decoded, "decoded");
    }

    [Fact]
    public void The_defaults_are_AudioLDMs()
    {
        // cvssp/audioldm-s-full-v2's vae/config.json: in 1, latent 8, widths 128/256/512, 2 layers, 32 groups.
        using var vae = new AudioVAE<double>(seed: 1);
        Assert.Equal(1, vae.InputChannels);
        Assert.Equal(8, vae.LatentChannels);
        Assert.Equal(4, vae.DownsampleFactor);
        Assert.Equal(64, vae.MelBins);
        Assert.Equal(AudioVAE<double>.AudioLdmLatentScale, vae.LatentScaleFactor);
        Assert.Equal(0.9392935633659363, AudioVAE<double>.AudioLdmLatentScale);
        Assert.Equal(0.4110932946205139, AudioVAE<double>.AudioLdm2LatentScale);
    }
}
