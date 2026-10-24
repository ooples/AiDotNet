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
/// TemporalInterpolationVAE reproduces Stable Video Diffusion's VAE, diffusers' AutoencoderKLTemporalDecoder
/// (<c>tools/reference-data/svd_vae_reference.py</c>): with the reference's weights loaded through
/// <see cref="TemporalInterpolationVAE{T}.LoadDiffusersWeights"/>, a tiny configuration gives the reference's posterior
/// moments and decoded clip.
/// </summary>
/// <remarks>
/// The reference sets every weight by a closed formula of its state-dict name and index (<see cref="WeightValue"/>),
/// so the fixture carries shapes rather than 340k values. The previous model was dense layers over the channel axis,
/// with no convolution and no spatial or temporal compression, although it reported a downsample factor of 8.
/// </remarks>
public class TemporalInterpolationVAEReferenceTests
{
    private const double Tolerance = 1e-9;

    private static JsonElement Fixture()
    {
        const string name = "svd_vae_reference.json";
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

    // The reference's weight_value: a bounded, irregular value for element index of the named tensor.
    private static double WeightValue(string name, int index)
    {
        int offset = 0;
        for (int k = 0; k < name.Length; k++) offset += name[k] * (k + 1);
        double c = (offset % 1000) / 100.0;
        return 0.2 * Math.Sin(0.37 * index + c) * Math.Cos(0.11 * index + 0.5 * c);
    }

    private static TemporalInterpolationVAE<double> LoadReference(JsonElement f, out IReadOnlyCollection<string> read)
    {
        var config = f.GetProperty("config");
        var vae = new TemporalInterpolationVAE<double>(
            inputChannels: config.GetProperty("in_channels").GetInt32(),
            latentChannels: config.GetProperty("latent_channels").GetInt32(),
            blockOutChannels: Ints(config.GetProperty("block_out_channels")),
            layersPerBlock: config.GetProperty("layers_per_block").GetInt32(),
            seed: 1);
        var shapes = f.GetProperty("shapes");
        read = vae.LoadDiffusersWeights((name, shape) =>
        {
            Assert.True(shapes.TryGetProperty(name, out var expected), $"The reference has no '{name}'.");
            Assert.Equal(Ints(expected), shape);
            int count = shape.Aggregate(1, (a, b) => a * b);
            return Enumerable.Range(0, count).Select(i => WeightValue(name, i)).ToArray();
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

        var expected = f.GetProperty("shapes").EnumerateObject().Select(p => p.Name).OrderBy(n => n, StringComparer.Ordinal);
        Assert.Equal(expected, read.OrderBy(n => n, StringComparer.Ordinal));
        Assert.Equal(f.GetProperty("shapes").EnumerateObject().Sum(p => Ints(p.Value).Aggregate(1L, (a, b) => a * b)),
            vae.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task Encode_and_decode_a_clip_match_the_reference()
    {
        await Task.Yield();
        var f = Fixture();
        using var vae = LoadReference(f, out _);
        var input = new Tensor<double>(Ints(f.GetProperty("input_shape")), new Vector<double>(Doubles(f.GetProperty("input"))));

        var (mean, logVariance) = vae.EncodeWithDistribution(input);
        var decoded = vae.Decode(mean);

        // moments = [mean | log-variance] along channels, laid out [batch, 2 * latent, frames, height, width].
        var momentShape = Ints(f.GetProperty("moments_shape"));
        var moments = Doubles(f.GetProperty("moments"));
        int latent = vae.LatentChannels, volume = momentShape[2] * momentShape[3] * momentShape[4];
        var expectedMean = new List<double>();
        var expectedLogVariance = new List<double>();
        for (int b = 0; b < momentShape[0]; b++)
        {
            int offset = b * 2 * latent * volume;
            expectedMean.AddRange(moments.Skip(offset).Take(latent * volume));
            expectedLogVariance.AddRange(moments.Skip(offset + latent * volume).Take(latent * volume));
        }
        Assert.Equal(new[] { momentShape[0], latent, momentShape[2], momentShape[3], momentShape[4] }, mean.Shape.ToArray());
        Close(expectedMean.ToArray(), mean, "mean");
        Close(expectedLogVariance.ToArray(), logVariance, "log-variance");
        Assert.Equal(Ints(f.GetProperty("decoded_shape")), decoded.Shape.ToArray());
        Close(Doubles(f.GetProperty("decoded")), decoded, "decoded");
    }

    [Fact]
    public void The_defaults_are_Stable_Video_Diffusions()
    {
        // stabilityai/stable-video-diffusion-img2vid's vae/config.json: in 3, latent 4, widths 128/256/512/512, 2 layers;
        // diffusers counts 97,742,847 parameters. The count is known without allocating the weights.
        using var vae = new TemporalInterpolationVAE<double>(seed: 1);
        Assert.Equal(3, vae.InputChannels);
        Assert.Equal(4, vae.LatentChannels);
        Assert.Equal(8, vae.DownsampleFactor);
        Assert.Equal(1, vae.TemporalCompression);
        Assert.Equal(0.18215, vae.LatentScaleFactor);
        Assert.Equal(97_742_847, vae.ParameterCount);
    }
}