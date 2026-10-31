using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Codecs;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Mimi, CSM's codec, against moshi 0.2.2's MimiModel with the released configuration (attention context shortened to 12
/// frames; tools/reference-data/mimi_reference.py): every checkpoint name has a place, the unquantized latent of two
/// seconds of audio matches, all 32 codebooks' codes match, and decoding them matches.
/// </summary>
public class MimiReferenceTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    // The fixture's closed form: s = sin(0.37 i + 0.11 len(name) + 0.0013 (i² mod 9973)).
    internal static double[] ClosedForm(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new double[count];
        for (int i = 0; i < count; i++)
        {
            long k = i;
            double s = Math.Sin(0.37 * i + 0.11 * name.Length + 0.0013 * (k * k % 9973));
            values[i] = name.EndsWith("norm1.weight", StringComparison.Ordinal) || name.EndsWith("norm2.weight", StringComparison.Ordinal) ? 1 + 0.2 * s
                : name.EndsWith("cluster_usage", StringComparison.Ordinal) ? 1 + 0.5 * s
                : name.EndsWith("embedding_sum", StringComparison.Ordinal) ? 0.5 * s
                : name.EndsWith("_initialized", StringComparison.Ordinal) ? 1
                : name.EndsWith(".scale", StringComparison.Ordinal) ? 0.05 * s
                : shape.Length >= 2 ? s * Math.Sqrt(2.0 * shape[0] / count)
                : 0.05 * s;
        }
        return values;
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    [Fact(Timeout = 600000)]
    public async Task EncodeAndDecode_MatchMoshi()
    {
        await Task.Yield();
        var reference = Load("mimi_reference.json");
        var shapes = reference.GetProperty("shapes").EnumerateObject()
            .ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(v => v.GetInt32()).ToArray(), StringComparer.Ordinal);
        var mimi = new Mimi<double>(new CpuEngine(), new MimiConfiguration { Context = reference.GetProperty("context").GetInt32() }, new List<LayerBase<double>>());
        var used = mimi.Load(name => shapes.TryGetValue(name, out var shape) ? ClosedForm(name, shape) : null);
        Assert.Empty(shapes.Keys.Where(n => !used.Contains(n)));
        Compare(mimi, reference);
    }

    /// <summary>The released codec (<c>MIMI_PATH</c>: <c>kyutai/moshiko-pytorch-bf16</c>'s
    /// <c>tokenizer-e351c8d8-checkpoint125.safetensors</c>) on three seconds of speech
    /// (tools/reference-data/mimi_realweights_reference.py).</summary>
    [Fact(Timeout = 1800000)]
    public async Task ReleasedWeights_MatchMoshi_WhenPresent()
    {
        await Task.Yield();
        string? path = Environment.GetEnvironmentVariable("MIMI_PATH");
        if (path is null || !File.Exists(path)) return;
        var state = new Dictionary<string, double[]>(StringComparer.Ordinal);
        foreach (var entry in new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(path))
            state[entry.Key] = entry.Value.ToVector().Select(v => (double)v).ToArray();
        var mimi = new Mimi<double>(new CpuEngine(), new MimiConfiguration(), new List<LayerBase<double>>());
        var used = mimi.Load(name => state.TryGetValue(name, out var v) ? v : null);
        Assert.Empty(state.Keys.Where(n => !used.Contains(n)));
        Compare(mimi, Load("mimi_realweights_reference.json"));
    }

    private static void Compare(Mimi<double> mimi, JsonElement reference)
    {
        var audio = Doubles(reference.GetProperty("audio"));
        var input = new Tensor<double>(new[] { 1, 1, audio.Length });
        for (int i = 0; i < audio.Length; i++) input[0, 0, i] = audio[i];
        var latent = mimi.Latent(input);
        var latentShape = reference.GetProperty("latent_shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        var expectedLatent = Doubles(reference.GetProperty("latent"));
        Assert.Equal(latentShape[0], latent.Shape[2]);
        double scale = expectedLatent.Max(Math.Abs), worst = 0;
        for (int t = 0; t < latentShape[0]; t++)
            for (int c = 0; c < latentShape[1]; c++)
                worst = Math.Max(worst, Math.Abs(expectedLatent[t * latentShape[1] + c] - latent[0, c, t]));
        Assert.True(worst <= 1e-9 * scale, $"latent differs by {worst} (scale {scale})");

        var expectedCodes = reference.GetProperty("codes").EnumerateArray().Select(r => r.EnumerateArray().Select(v => v.GetInt32()).ToArray()).ToArray();
        var codes = mimi.Encode(audio);
        Assert.Equal(expectedCodes.Length, codes.GetLength(0));
        for (int t = 0; t < expectedCodes.Length; t++)
            Assert.Equal(expectedCodes[t], Enumerable.Range(0, codes.GetLength(1)).Select(q => codes[t, q]).ToArray());

        var given = new int[expectedCodes.Length, expectedCodes[0].Length];
        for (int t = 0; t < expectedCodes.Length; t++)
            for (int q = 0; q < expectedCodes[t].Length; q++) given[t, q] = expectedCodes[t][q];
        var decoded = mimi.Decode(given);
        var expected = Doubles(reference.GetProperty("decoded"));
        Assert.Equal(expected.Length, decoded.Length);
        double audioScale = expected.Max(Math.Abs);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - decoded[i]) <= 1e-9 * audioScale, $"sample {i}: {expected[i]} vs {decoded[i]}");
    }
}
