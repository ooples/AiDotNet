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
/// GPT-SoVITS's SoVITS synthesizer against the release's SynthesizerTrn in float64 for v1, v2 and v2ProPlus
/// (tools/reference-data/gptsovits_sovits_reference.py): semantic codes, the style vector, the prior, the flow, the
/// generator, decode with the noise fixed (speed 1 and 1.25) and the posterior.
/// </summary>
public class GptSoVitsSynthesizerReferenceTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "gptsovits_sovits_reference.json";
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

    private static (GptSoVitsSynthesizer<double> Model, JsonElement Entry) Model(string version)
    {
        var r = Reference.Value;
        var e = r.GetProperty("versions").GetProperty(version);
        int ssl = r.GetProperty("ssl_dim").GetInt32(), hidden = r.GetProperty("hidden").GetInt32();
        var c = new GptSoVitsSynthesizerConfiguration
        {
            Version = version, InterChannels = 8, Hidden = hidden, Filter = 32, Heads = 2, Layers = 2, Kernel = 3, Dropout = 0.1,
            ResblockKernels = new[] { 3, 5 }, ResblockDilations = new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 } }, UpsampleRates = new[] { 4, 2 },
            UpsampleInitialChannels = 32, UpsampleKernels = new[] { 8, 4 }, Gin = e.GetProperty("gin").GetInt32(), Symbols = e.GetProperty("symbols").GetInt32(),
            SslDim = ssl, Bins = 64, SvDim = 32, MrteChannels = e.GetProperty("mrte_hidden").GetInt32(), StyleHidden = 16,
        };
        var model = new GptSoVitsSynthesizer<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(), c, new Random(1));
        var state = e.GetProperty("state");
        model.Load(name => state.TryGetProperty(name, out var v) ? Doubles(v) : null);
        return (model, e);
    }

    [Theory(Timeout = 300000)]
    [InlineData("v1")]
    [InlineData("v2")]
    [InlineData("v2ProPlus")]
    public async Task Modules_MatchSynthesizerTrn(string version)
    {
        await Task.Yield();
        var (model, e) = Model(version);
        int ssl = Reference.Value.GetProperty("ssl_dim").GetInt32();
        Assert.Equal(Ints(e.GetProperty("codes")), model.ExtractLatent(T(e.GetProperty("ssl"), 1, ssl, 14)));
        var reference = T(e.GetProperty("reference"), 1, 1025, 11);
        Tensor<double>? sv = e.GetProperty("sv").ValueKind == JsonValueKind.Null ? null : T(e.GetProperty("sv"), 32);
        var style = model.Style(reference, sv);
        // The release pools the style encoder's frames in float32 (temporal_avg_pool casts with x.float()), so the style
        // vector, and everything it conditions, matches to float32 precision.
        Close(Doubles(e.GetProperty("style")), style, 1e-6, "style");
        var codes = Ints(e.GetProperty("decode_codes"));
        var text = Ints(e.GetProperty("text"));
        var (mean, logs) = model.Encode(model.Dequantize(codes), text, style);
        Close(Doubles(e.GetProperty("prior_mean")), mean, 1e-6, "prior mean");
        Close(Doubles(e.GetProperty("prior_logs")), logs, 1e-6, "prior logs");
        var z = T(e.GetProperty("flow_input"), 1, 8, 12);
        Close(Doubles(e.GetProperty("flow_forward")), model.Flow(z, style, reverse: false), 1e-6, "flow forward");
        Close(Doubles(e.GetProperty("flow_reverse")), model.Flow(z, style, reverse: true), 1e-6, "flow reverse");
        Close(Doubles(e.GetProperty("generator")), model.Generator.Forward(z, style), 1e-6, "generator");
        foreach (var d in e.GetProperty("decodes").EnumerateArray())
        {
            var noise = T(d.GetProperty("noise"), Ints(d.GetProperty("noise_shape")));
            var audio = model.Decode(codes, text, new[] { reference }, new[] { sv }, 0.6, d.GetProperty("speed").GetDouble(), shape =>
            {
                Assert.Equal(noise.Shape.ToArray(), shape);
                return noise;
            });
            Close(Doubles(d.GetProperty("audio")), audio, 1e-6, $"decode at speed {d.GetProperty("speed").GetDouble()}");
        }
        var p = e.GetProperty("posterior");
        var (zq, mq, logsq) = model.Posterior.Forward(T(p.GetProperty("spec"), 1, 1025, 9), style, T(p.GetProperty("noise"), 1, 8, 9));
        Close(Doubles(p.GetProperty("z")), zq, 1e-6, "posterior z");
        Close(Doubles(p.GetProperty("mean")), mq, 1e-6, "posterior mean");
        Close(Doubles(p.GetProperty("logs")), logsq, 1e-6, "posterior logs");
    }
}
