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
/// GPT-SoVITS's flow-matching synthesizer against the release's SynthesizerTrnV3, CFM and CFMV5 in float64 for v3, v4 and
/// v5dev (tools/reference-data/gptsovits_cfm_reference.py): the style vector, decode_encp at speed 1 and 1.25, one CFM
/// inference, the chunked synthesis of a two-chunk target, and the training loss with its draws fixed. The CFM's noise is
/// the fixture's closed-form pattern.
/// </summary>
public class GptSoVitsCfmReferenceTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "gptsovits_cfm_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();
    private static Tensor<double> T(JsonElement e, params int[] shape) => new(shape, new Vector<double>(Doubles(e)));

    private static void Close(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * (1 + Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual.GetFlat(i)}");
    }

    // The fixture's noise: [B, C, T] = sin(1.3·t + 0.7·c + 0.11·call), here as rows [T, C].
    private static Func<int[], Tensor<double>> Noise()
    {
        int call = 0;
        return shape =>
        {
            var x = new Tensor<double>(shape);
            for (int t = 0; t < shape[0]; t++)
                for (int c = 0; c < shape[1]; c++) x[t, c] = Math.Sin(1.3 * t + 0.7 * c + 0.11 * call);
            call++;
            return x;
        };
    }

    private static Tensor<double> Transpose(Tensor<double> channelsFirst)
    {
        int c = channelsFirst.Shape[1], t = channelsFirst.Shape[2];
        var rows = new Tensor<double>(new[] { t, c });
        for (int i = 0; i < t; i++)
            for (int j = 0; j < c; j++) rows[i, j] = channelsFirst[0, j, i];
        return rows;
    }

    private static (GptSoVitsCfmSynthesizer<double> Model, JsonElement Entry) Model(string version)
    {
        var r = Reference.Value;
        var e = r.GetProperty("versions").GetProperty(version);
        int ssl = r.GetProperty("ssl_dim").GetInt32(), hidden = r.GetProperty("hidden").GetInt32(), gin = r.GetProperty("gin").GetInt32();
        int bridge = r.GetProperty("bridge").GetInt32();
        var c = GptSoVitsSynthesizerConfiguration.For(version) with
        {
            InterChannels = 8, Hidden = hidden, Filter = 32, Heads = 2, Layers = 2, Kernel = 3, Gin = gin, Symbols = e.GetProperty("symbols").GetInt32(),
            SslDim = ssl, Bins = 64, MrteChannels = gin, StyleHidden = 16,
        };
        var dit = new GptSoVitsDiTConfiguration(Dim: 32, Depth: 2, Heads: 2, HeadDim: 16, FfMult: 2, TextDim: bridge);
        var model = new GptSoVitsCfmSynthesizer<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), c, dit, new Random(1), bridge);
        var state = e.GetProperty("state");
        model.Load(name => state.TryGetProperty(name, out var v) ? Doubles(v) : null);
        return (model, e);
    }

    [Theory(Timeout = 600000)]
    [InlineData("v3")]
    [InlineData("v4")]
    [InlineData("v5dev")]
    public async Task StyleAndDecodeEncp_MatchSynthesizerTrnV3(string version)
    {
        await Task.Yield();
        var (model, e) = Model(version);
        // The style encoder pools in float32 (temporal_avg_pool), so the style vector matches to that precision.
        var style = model.Style(T(e.GetProperty("reference"), 1, 1025, 11));
        Close(Doubles(e.GetProperty("ge")), style, 1e-6, "ge");
        var ge = T(e.GetProperty("ge"), 1, Reference.Value.GetProperty("gin").GetInt32(), 1);
        var codes = Ints(e.GetProperty("codes"));
        var text = Ints(e.GetProperty("text"));
        foreach (var d in e.GetProperty("decodes").EnumerateArray())
        {
            var fea = model.DecodeFeatures(codes, text, ge, d.GetProperty("speed").GetDouble());
            Assert.Equal(Ints(d.GetProperty("shape")), fea.Shape.ToArray());
            Close(Doubles(d.GetProperty("fea")), fea, 1e-9, $"fea@{d.GetProperty("speed").GetDouble()}");
        }
    }

    [Theory(Timeout = 600000)]
    [InlineData("v3")]
    [InlineData("v4")]
    [InlineData("v5dev")]
    public async Task Inference_MatchesCfm(string version)
    {
        await Task.Yield();
        var (model, e) = Model(version);
        var inference = e.GetProperty("inference");
        int bridge = Reference.Value.GetProperty("bridge").GetInt32();
        var mu = T(inference.GetProperty("mu"), 9, bridge);
        var prompt = Transpose(T(inference.GetProperty("prompt"), 1, 100, 3));
        var mel = model.Cfm.Inference(mu, prompt, inference.GetProperty("steps").GetInt32(), 1.0, inference.GetProperty("cfg").GetDouble(),
            Noise()(new[] { 9, 100 }));
        // The text positions are a float32 table (precompute_freqs_cis) whose cos/sin torch rounds within an ulp of .NET's.
        Close(Doubles(inference.GetProperty("mel")), Channels(mel), 1e-6, "mel");
    }

    // [T, C] rows to [1, C, T].
    private static Tensor<double> Channels(Tensor<double> rows)
    {
        int t = rows.Shape[0], c = rows.Shape[1];
        var x = new Tensor<double>(new[] { 1, c, t });
        for (int i = 0; i < t; i++)
            for (int j = 0; j < c; j++) x[0, j, i] = rows[i, j];
        return x;
    }

    [Theory(Timeout = 900000)]
    [InlineData("v3")]
    [InlineData("v4")]
    [InlineData("v5dev")]
    public async Task ChunkedSynthesis_MatchesTheRelease(string version)
    {
        await Task.Yield();
        var (model, e) = Model(version);
        var chunked = e.GetProperty("chunked");
        int bridge = Reference.Value.GetProperty("bridge").GetInt32(), stride = Reference.Value.GetProperty("stride").GetInt32();
        int target = chunked.GetProperty("target").GetInt32();
        var features = new Tensor<double>(new[] { 1, bridge, target });
        for (int c = 0; c < bridge; c++)
            for (int t = 0; t < target; t++) features[0, c, t] = Math.Sin(t * 0.05 + c * 0.3);
        var mel = model.SynthesizeMel(T(chunked.GetProperty("fea_ref"), 1, bridge, 32), T(chunked.GetProperty("mel2"), 1, 100, 30), features,
            chunked.GetProperty("steps").GetInt32(), chunked.GetProperty("cfg").GetDouble(), Noise());
        Assert.Equal(Ints(chunked.GetProperty("shape")), mel.Shape.ToArray());
        var expected = Doubles(chunked.GetProperty("mel"));
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - mel.GetFlat(i * stride)) <= 1e-6 * (1 + Math.Abs(expected[i])), $"mel[{i * stride}]: {expected[i]} vs {mel.GetFlat(i * stride)}");
        double sum = 0;
        for (int i = 0; i < mel.Length; i++) sum += mel.GetFlat(i);
        Assert.True(Math.Abs(chunked.GetProperty("sum").GetDouble() - sum) <= 1e-6 * (1 + Math.Abs(sum)), $"sum {sum}");
    }

    [Theory(Timeout = 600000)]
    [InlineData("v3")]
    [InlineData("v4")]
    [InlineData("v5dev")]
    public async Task TrainingLoss_MatchesForward(string version)
    {
        await Task.Yield();
        var (model, e) = Model(version);
        var l = e.GetProperty("loss");
        int ssl = Reference.Value.GetProperty("ssl_dim").GetInt32(), frames = l.GetProperty("mel_frames").GetInt32();
        var text = Ints(e.GetProperty("text"));
        foreach (var c in l.GetProperty("cases").EnumerateArray())
        {
            var draws = new GptSoVitsCfmSynthesizer<double>.LossDraws(c.GetProperty("prompt_u").GetDouble(), c.GetProperty("t").GetDouble(),
                Noise()(new[] { frames, 100 }), c.GetProperty("bootstrap").GetBoolean() ? c.GetProperty("k").GetInt32() : null);
            var loss = model.Loss(T(l.GetProperty("ssl"), 1, ssl, 8), T(l.GetProperty("spec"), 1, 1025, 8), T(l.GetProperty("mel"), 1, 100, frames), text,
                freezeQuantizer: true, new Random(0), training: false, draws);
            double expected = c.GetProperty("loss").GetDouble();
            // The style vector's float32 pooling bounds the agreement.
            Assert.True(Math.Abs(expected - loss.GetFlat(0)) <= 1e-5 * (1 + Math.Abs(expected)), $"loss (bootstrap {c.GetProperty("bootstrap")}, k {c.GetProperty("k")}): {expected} vs {loss.GetFlat(0)}");
        }
    }

    [Fact(Timeout = 600000)]
    public async Task Lora_MatchesPeftBeforeAndAfterMerging()
    {
        await Task.Yield();
        var (model, e) = Model("v3");
        var l = e.GetProperty("lora");
        int rank = l.GetProperty("rank").GetInt32(), bridge = Reference.Value.GetProperty("bridge").GetInt32();
        var dit = model.Cfm.Estimator;
        dit.AddLora(new List<LayerBase<double>>(), rank, rank, new Random(3));
        var state = l.GetProperty("state");
        dit.LoadLora(name => state.TryGetProperty(name, out var v) ? Doubles(v) : null, "estimator.", rank);
        var mu = T(l.GetProperty("mu"), 9, bridge);
        var prompt = Transpose(T(l.GetProperty("prompt"), 1, 100, 3));
        var adapted = model.Cfm.Inference(mu, prompt, 3, 1.0, 0.0, Noise()(new[] { 9, 100 }));
        Close(Doubles(l.GetProperty("adapted")), Channels(adapted), 1e-6, "adapted");
        dit.MergeLora();
        Assert.False(dit.HasLora);
        var merged = model.Cfm.Inference(mu, prompt, 3, 1.0, 0.0, Noise()(new[] { 9, 100 }));
        Close(Doubles(l.GetProperty("merged")), Channels(merged), 1e-6, "merged");
    }
}
