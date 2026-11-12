using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// NaturalSpeech's modules against heatz123/naturalspeech in float64 (tools/reference-data/naturalspeech_reference.py):
/// the learnable upsampling (its variant), the duration predictor, the VAE memory bank and soft-DTW over the pairwise KL.
/// </summary>
public class NaturalSpeechReferenceTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "naturalspeech_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static Tensor<double> T(JsonElement e, params int[] shape) => new(shape, new Vector<double>(Doubles(e)));

    private static void Close(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * (1 + Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual.GetFlat(i)}");
    }

    private static Func<string, double[]?> Reader(JsonElement state) => n => state.TryGetProperty(n, out var v) ? Doubles(v) : null;

    [Fact(Timeout = 120000)]
    public async Task Heatz123Upsampling_MatchesLearnableUpsampling()
    {
        await Task.Yield();
        var u = Reference.Value.GetProperty("upsampling");
        var engine = AiDotNetEngine.Current;
        var up = new NaturalSpeechUpsampler<double>(engine, new List<LayerBase<double>>(), new Random(1), NaturalSpeechUpsampling.Heatz123, 192, 3, 8, 4, 2, 192);
        up.LoadHeatz123(Reader(u.GetProperty("state")), "");
        var hidden = engine.Reshape(engine.TensorTranspose(T(u.GetProperty("hidden"), 5, 192)), new[] { 1, 192, 5 });
        var (mean, logs, frames) = up.Forward(hidden, T(u.GetProperty("durations"), 5), false, new Random(0));
        Assert.Equal(u.GetProperty("frames").GetInt32(), frames);
        // The port returns proj_o's output as [T, 384]; ours is the mean and log-scale halves, channels first.
        var expected = Doubles(u.GetProperty("output"));
        for (int t = 0; t < frames; t++)
            for (int c = 0; c < 192; c++)
            {
                Assert.True(Math.Abs(expected[t * 384 + c] - mean[0, c, t]) <= 1e-9 * (1 + Math.Abs(expected[t * 384 + c])), $"mean[{t},{c}]");
                Assert.True(Math.Abs(expected[t * 384 + 192 + c] - logs[0, c, t]) <= 1e-9 * (1 + Math.Abs(expected[t * 384 + 192 + c])), $"logs[{t},{c}]");
            }
    }

    [Fact(Timeout = 120000)]
    public async Task DurationPredictorAndMemory_MatchTheRelease()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var d = Reference.Value.GetProperty("duration");
        var dp = new NaturalSpeechDurationPredictor<double>(engine, new List<LayerBase<double>>(), 16, 12, 3, 2, 0.5);
        var state = d.GetProperty("state");
        dp.Load(n => Doubles(state.GetProperty(n)), "");
        Close(Doubles(d.GetProperty("logw")), dp.Forward(T(d.GetProperty("hidden"), 1, 16, 7), false, new Random(0)), 1e-10, "logw");

        var m = Reference.Value.GetProperty("memory");
        var memory = new NaturalSpeechMemoryBank<double>(engine, new List<LayerBase<double>>(), new Random(1), 16, 10, 2);
        var ms = m.GetProperty("state");
        memory.Load(n => Doubles(ms.GetProperty(n)), "");
        Close(Doubles(m.GetProperty("output")), memory.Forward(T(m.GetProperty("z"), 1, 16, 6)), 1e-10, "memory");
    }

    [Fact(Timeout = 120000)]
    public async Task SoftDtw_MatchesTheRelease()
    {
        await Task.Yield();
        var engine = AiDotNetEngine.Current;
        var s = Reference.Value.GetProperty("softdtw");
        var matrix = NaturalSpeechOps.KlMatrix(engine, T(s.GetProperty("z"), 1, 4, 7), T(s.GetProperty("logs_q"), 1, 4, 7), T(s.GetProperty("m_p"), 1, 4, 5),
            T(s.GetProperty("logs_p"), 1, 4, 5));
        // The port's matrix is [prior frames, posterior frames] and computed in float32 (get_sdtw_kl_matrix casts with
        // .float()); ours is [posterior, prior] in the model's precision.
        var expected = Doubles(s.GetProperty("matrix"));
        for (int j = 0; j < 5; j++)
            for (int i = 0; i < 7; i++)
                Assert.True(Math.Abs(expected[j * 7 + i] - matrix[i, j]) <= 1e-6 * (1 + Math.Abs(expected[j * 7 + i])), $"kl[{i},{j}]: {expected[j * 7 + i]} vs {matrix[i, j]}");
        // Soft-DTW on the port's own matrix.
        var value = NaturalSpeechOps.SoftDtw(engine, T(s.GetProperty("matrix"), 5, 7), 0.01, 0.07).GetFlat(0);
        double reference = s.GetProperty("value").GetDouble();
        Assert.True(Math.Abs(reference - value) <= 1e-9 * (1 + Math.Abs(reference)), $"soft-DTW: {reference} vs {value}");
    }
}
