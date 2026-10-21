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
/// Causal3DVAE reproduces diffusers' AutoencoderKLCogVideoX (<c>tools/reference-data/cogvideox_vae_reference.py</c>):
/// with the reference's weights loaded through <see cref="Causal3DVAE{T}.LoadCogVideoXWeights"/>, a tiny configuration
/// gives the reference's posterior moments and decoded clips for an odd clip (1 + 4k frames, the causal layout), an even
/// clip and a single frame.
/// </summary>
/// <remarks>
/// The previous model was three dense layers over the channel axis: it had no convolution, so it could not load this
/// state dict, compressed nothing although it reported 8x, and mixed no frames.
/// </remarks>
public class Causal3DVAEReferenceTests
{
    private const double Tolerance = 1e-9;

    private static JsonElement Fixture()
    {
        const string name = "cogvideox_vae_reference.json";
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

    private static Causal3DVAE<double> LoadReference(JsonElement f, out IReadOnlyCollection<string> read)
    {
        var config = f.GetProperty("config");
        var vae = new Causal3DVAE<double>(
            inputChannels: config.GetProperty("in_channels").GetInt32(),
            latentChannels: config.GetProperty("latent_channels").GetInt32(),
            blockOutChannels: Ints(config.GetProperty("block_out_channels")),
            layersPerBlock: config.GetProperty("layers_per_block").GetInt32(),
            normGroups: config.GetProperty("norm_num_groups").GetInt32(),
            temporalCompression: config.GetProperty("temporal_compression_ratio").GetInt32(),
            seed: 1);
        var state = f.GetProperty("state_dict");
        var shapes = f.GetProperty("shapes");
        read = vae.LoadCogVideoXWeights((name, shape) =>
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

    [Theory(Timeout = 300000)]
    [InlineData("clip9")]
    [InlineData("clip8")]
    [InlineData("frame1")]
    public async Task Encode_and_decode_match_the_reference(string clip)
    {
        await Task.Yield();
        var f = Fixture();
        using var vae = LoadReference(f, out _);
        var c = f.GetProperty("cases").GetProperty(clip);
        var input = new Tensor<double>(Ints(c.GetProperty("input_shape")), new Vector<double>(Doubles(c.GetProperty("input"))));
        int latentChannels = vae.LatentChannels;

        var (mean, logVariance) = vae.EncodeWithDistribution(input);
        var decoded = vae.Decode(mean);

        // moments = [mean | log-variance] along channels.
        var momentShape = Ints(c.GetProperty("moments_shape"));
        var moments = Doubles(c.GetProperty("moments"));
        int perChannelBlock = momentShape[2] * momentShape[3] * momentShape[4];
        var expectedMean = moments.Take(latentChannels * perChannelBlock).ToArray();
        var expectedLogVariance = moments.Skip(latentChannels * perChannelBlock).ToArray();
        Assert.Equal(new[] { 1, latentChannels, momentShape[2], momentShape[3], momentShape[4] }, mean.Shape.ToArray());
        Assert.Equal(vae.GetLatentFrameCount(input.Shape[2]), mean.Shape[2]);
        Close(expectedMean, mean, "mean");
        Close(expectedLogVariance, logVariance, "log-variance");
        Assert.Equal(Ints(c.GetProperty("decoded_shape")), decoded.Shape.ToArray());
        Close(Doubles(c.GetProperty("decoded")), decoded, "decoded");
    }

    [Fact(Timeout = 120000)]
    public async Task An_image_encodes_as_a_one_frame_clip()
    {
        await Task.Yield();
        var f = Fixture();
        using var vae = LoadReference(f, out _);
        var c = f.GetProperty("cases").GetProperty("frame1");
        var shape = Ints(c.GetProperty("input_shape"));
        var image = new Tensor<double>(new[] { shape[0], shape[1], shape[3], shape[4] }, new Vector<double>(Doubles(c.GetProperty("input"))));

        var latent = vae.Encode(image, sampleMode: false);
        var decoded = vae.Decode(latent);

        Assert.Equal(new[] { 1, vae.LatentChannels, shape[3] / vae.DownsampleFactor, shape[4] / vae.DownsampleFactor }, latent.Shape.ToArray());
        Assert.Equal(image.Shape.ToArray(), decoded.Shape.ToArray());
        Close(Doubles(c.GetProperty("decoded")), decoded, "decoded image");
    }
}
