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
/// MARS5's pieces against the release's code in float64 (tools/reference-data/mars5_reference.py): the minbpe text and
/// speech tokenizers, the AR CodecLM and the NAR ResidualTransformer at tiny sizes, the multinomial diffusion (schedule,
/// posterior, loss, and RePaint inference with closed-form draws), the logit processors and the silence trimmer.
/// </summary>
public class Mars5ReferenceTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "mars5_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.ValueKind == JsonValueKind.Null ? double.NegativeInfinity : v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static int[,] Grid(JsonElement e)
    {
        var rows = e.EnumerateArray().Select(Ints).ToArray();
        var g = new int[rows.Length, rows.Length == 0 ? 0 : rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[i].Length; j++) g[i, j] = rows[i][j];
        return g;
    }

    private static void Close(double[] expected, Func<int, double> actual, int count, double tolerance, string what)
    {
        Assert.Equal(expected.Length, count);
        for (int i = 0; i < expected.Length; i++)
        {
            double a = actual(i);
            if (double.IsNegativeInfinity(expected[i])) { Assert.True(double.IsNegativeInfinity(a), $"{what}[{i}]: -inf vs {a}"); continue; }
            Assert.True(Math.Abs(expected[i] - a) <= tolerance * (1 + Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {a}");
        }
    }

    private static Func<string, double[]?> Reader(JsonElement state) => name => state.TryGetProperty(name, out var v) ? Doubles(v) : null;

    [Fact(Timeout = 120000)]
    public async Task Tokenizers_MatchMinBpe()
    {
        await Task.Yield();
        var t = Reference.Value.GetProperty("tokenizers");
        var text = new MinBpeRegexTokenizer(MinBpeModel.Parse(t.GetProperty("text_model").GetString()!));
        var speech = new MinBpeCodebookTokenizer(MinBpeModel.Parse(t.GetProperty("speech_model").GetString()!));
        Assert.Equal(t.GetProperty("text_vocab").GetInt32(), text.VocabSize);
        Assert.Equal(t.GetProperty("speech_vocab").GetInt32(), speech.VocabSize);
        foreach (var c in t.GetProperty("texts").EnumerateArray())
            Assert.Equal(Ints(c.GetProperty("ids")), text.Encode(c.GetProperty("text").GetString()!).ToArray());
        foreach (var c in t.GetProperty("codes").EnumerateArray())
            Assert.Equal(Ints(c.GetProperty("ids")), speech.Encode(Ints(c.GetProperty("codes"))).ToArray());
        foreach (var c in t.GetProperty("decode").EnumerateArray())
            Assert.Equal(Ints(c.GetProperty("codes")), speech.DecodeCodes(Ints(c.GetProperty("ids"))).ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task CodecLm_MatchesTheRelease()
    {
        await Task.Yield();
        var a = Reference.Value.GetProperty("ar");
        var model = new Mars5CodecLm<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(),
            new Mars5ArConfiguration(40, Dim: 32, Heads: 4, Layers: 2, SpeakerLayers: 1, Hidden: (int)(32 * (7.0 / 3))), new Random(1));
        model.Load(Reader(a.GetProperty("state")));
        var speaker = model.Speaker(Grid(a.GetProperty("reference")));
        // The sine position tables (and the NAR's timestep embedding) are float32 in the release.
        Close(Doubles(a.GetProperty("speaker")), i => speaker.GetFlat(i), speaker.Length, 1e-6, "speaker");
        var logits = model.Logits(Ints(a.GetProperty("tokens")), speaker);
        Close(Doubles(a.GetProperty("logits")), i => logits.GetFlat(i), logits.Length, 1e-6, "logits");
        // The kv-cached path gives the last position's logits.
        var tokens = Ints(a.GetProperty("tokens"));
        var cache = new AiDotNet.NeuralNetworks.LlamaCache<double>();
        model.NextLogits(tokens.Take(tokens.Length - 1).ToList(), speaker, cache);
        var next = model.NextLogits(tokens, speaker, cache);
        var expected = Doubles(a.GetProperty("logits")).Skip((tokens.Length - 1) * 40).ToArray();
        Close(expected, i => next.GetFlat(i), next.Length, 1e-6, "cached");
    }

    [Fact(Timeout = 300000)]
    public async Task ResidualTransformer_MatchesTheRelease()
    {
        await Task.Yield();
        var n = Reference.Value.GetProperty("nar");
        var model = new Mars5ResidualTransformer<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(),
            new Mars5NarConfiguration(30, Dim: 32, Heads: 4, EncoderLayers: 2, DecoderLayers: 2, SpeakerLayers: 1, TimeDim: 16, CondDrop: 0), new Random(1));
        model.Load(Reader(n.GetProperty("state")));
        foreach (var c in n.GetProperty("cases").EnumerateArray())
        {
            var logits = model.Forward(Ints(n.GetProperty("text")), Grid(n.GetProperty("reference")), Grid(n.GetProperty("x")), c.GetProperty("t").GetInt32(),
                c.GetProperty("drop").GetBoolean());
            Close(Doubles(c.GetProperty("logits")), i => logits.GetFlat(i), logits.Length, 1e-6, $"nar t={c.GetProperty("t").GetInt32()}");
        }
    }

    [Fact(Timeout = 600000)]
    public async Task Diffusion_MatchesTheRelease()
    {
        await Task.Yield();
        var d = Reference.Value.GetProperty("diffusion");
        foreach (var s in d.GetProperty("schedules").EnumerateArray())
            Assert.Equal(Ints(s.GetProperty("ts")), Mars5Diffusion.Schedule(s.GetProperty("T").GetInt32(), s.GetProperty("jl").GetInt32(), s.GetProperty("js").GetInt32()).ToArray());
        var diffusion = new Mars5Diffusion(1025, d.GetProperty("timesteps").GetInt32());
        var x0 = Grid(d.GetProperty("x0"));
        var xt = Grid(d.GetProperty("xt"));
        int Flat(int[,] g, int i) => g[i / 8, i % 8];
        var logX0 = diffusion.OneHotRows(Enumerable.Range(0, 32).Select(i => Flat(x0, i)).ToArray());
        var logXt = diffusion.OneHotRows(Enumerable.Range(0, 32).Select(i => Flat(xt, i)).ToArray());
        var pred = Doubles(d.GetProperty("pred"));
        var logPred = new double[pred.Length];
        for (int r = 0; r < 32; r++)
        {
            double max = Enumerable.Range(0, 1025).Max(k => pred[r * 1025 + k]);
            double norm = max + Math.Log(Enumerable.Range(0, 1025).Sum(k => Math.Exp(pred[r * 1025 + k] - max)));
            for (int k = 0; k < 1025; k++) logPred[r * 1025 + k] = pred[r * 1025 + k] - norm;
        }
        var engine = AiDotNetEngine.Current;
        var predTensor = new Tensor<double>(new[] { 32, 1025 }, new Vector<double>(logPred));
        foreach (var l in d.GetProperty("losses").EnumerateArray())
        {
            int t = l.GetProperty("t").GetInt32();
            var posterior = diffusion.QPosterior(logX0, logXt, t);
            Close(Doubles(l.GetProperty("posterior_head")), i => posterior[i], 12, 1e-6, $"posterior t={t}");
            double loss = diffusion.Loss(engine, logX0, logXt, predTensor, t).GetFlat(0);
            Assert.True(Math.Abs(l.GetProperty("loss").GetDouble() - loss) <= 1e-6 * (1 + Math.Abs(loss)), $"loss t={t}: {l.GetProperty("loss").GetDouble()} vs {loss}");
            double noPrior = diffusion.Loss(engine, logX0, logXt, predTensor, t, includeKlPrior: false).GetFlat(0);
            Assert.True(Math.Abs(l.GetProperty("loss_no_prior").GetDouble() - noPrior) <= 1e-6 * (1 + Math.Abs(noPrior)), $"loss (no prior) t={t}");
        }
        var scaled = diffusion.QPredOneScaled(logXt, 2, 0.6, 1, 4, 8);
        var head = Doubles(d.GetProperty("scaled_head"));
        Close(head, i => scaled[(i / 6) * 1025 + i % 6], head.Length, 1e-6, "scaled");
    }

    [Fact(Timeout = 900000)]
    public async Task Inference_MatchesPerformSimpleInference()
    {
        await Task.Yield();
        var r = Reference.Value;
        var n = r.GetProperty("nar");
        var model = new Mars5ResidualTransformer<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(),
            new Mars5NarConfiguration(30, Dim: 32, Heads: 4, EncoderLayers: 2, DecoderLayers: 2, SpeakerLayers: 1, TimeDim: 16, CondDrop: 0), new Random(1));
        // The fixture's NAR without its float32 inputs, so its logits match exactly and the sampled codes can be compared.
        model.Load(Reader(r.GetProperty("diffusion").GetProperty("inference_state")));
        var text = Ints(n.GetProperty("text"));
        var reference = Grid(n.GetProperty("reference"));
        foreach (var c in r.GetProperty("diffusion").GetProperty("inference").EnumerateArray())
        {
            long u = 0, ri = 0;
            double[] Uniform(int count)
            {
                var values = new double[count];
                for (int i = 0; i < count; i++, u++)
                {
                    double v = Math.Sin(u * 12.9898) * 43758.5453;
                    values[i] = Math.Abs(v - Math.Floor(v));
                }
                return values;
            }
            int RandomCode() => (int)((ri++ * 7919 + 13) % 1025);
            double[] Predict(int[,] x, int t, bool drop)
            {
                var logits = model.Forward(text, reference, x, t, drop);
                return Enumerable.Range(0, logits.Length).Select(logits.GetFlat).ToArray();
            }
            bool deep = c.GetProperty("deep").GetBoolean();
            var dsh = new Mars5DiffusionSampling(c.GetProperty("jump_len").GetInt32(), c.GetProperty("jump_n_sample").GetInt32(), LastGreedy: false,
                X0Temperature: 0.7, GuidanceWeight: c.GetProperty("guidance").GetDouble(), ScaledInference: true, DeepClone: deep,
                Q0OverrideSteps: c.GetProperty("q0_override").GetInt32());
            var diffusion = new Mars5Diffusion(1025, c.GetProperty("timesteps").GetInt32());
            var codes = diffusion.Infer(Ints(c.GetProperty("quant0")), 8, deep ? reference : null, Predict, dsh, Uniform, RandomCode);
            var expected = Grid(c.GetProperty("codes"));
            Assert.Equal(expected.GetLength(0), codes.GetLength(0));
            for (int i = 0; i < expected.GetLength(0); i++)
                for (int q = 0; q < 8; q++) Assert.True(expected[i, q] == codes[i, q], $"deep={deep} jl={dsh.JumpLength}: codes[{i},{q}] {expected[i, q]} vs {codes[i, q]}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task SamplersAndTrim_MatchTheRelease()
    {
        await Task.Yield();
        var s = Reference.Value.GetProperty("samplers");
        var logits = Doubles(s.GetProperty("logits"));
        var previous = Ints(s.GetProperty("previous"));
        var a = (double[])logits.Clone();
        Mars5Sampler.FrequencyPenalty(a, previous, 3, 0.4, 6);
        Close(Doubles(s.GetProperty("penalized")), i => a[i], a.Length, 1e-12, "penalized");
        var b = (double[])a.Clone();
        Mars5Sampler.EosPenalty(b, 4, 10, 0.5, 1.0, 12);
        Close(Doubles(s.GetProperty("eos")), i => b[i], b.Length, 1e-12, "eos");
        var c = b.Select(v => v / 0.7).ToArray();
        Mars5Sampler.TopKTopP(c, 20, 0.6);
        Close(Doubles(s.GetProperty("topkp")), i => c[i], c.Length, 1e-12, "top-k/top-p");
        var d = (double[])b.Clone();
        Mars5Sampler.TypicalP(d, 0.5);
        Close(Doubles(s.GetProperty("typical")), i => d[i], d.Length, 1e-12, "typical");
        var t = Reference.Value.GetProperty("trim");
        var (_, start, end) = Mars5Trim.Trim(Doubles(t.GetProperty("audio")), 27);
        Assert.Equal(t.GetProperty("start").GetInt32(), start);
        Assert.Equal(t.GetProperty("end").GetInt32(), end);
    }
}
