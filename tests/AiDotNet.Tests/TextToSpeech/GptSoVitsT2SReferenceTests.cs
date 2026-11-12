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
/// GPT-SoVITS's text-to-semantic model against the release's Text2SemanticDecoder in float64
/// (tools/reference-data/gptsovits_t2s_reference.py): the text rows, the teacher-forced logits, the cross-entropy and DPO
/// losses, generation with the sampling noise fixed, and logits_to_probs.
/// </summary>
public class GptSoVitsT2SReferenceTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "gptsovits_t2s_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static GptSoVitsT2S<double> Model()
    {
        var r = Reference.Value;
        var c = r.GetProperty("config");
        var model = new GptSoVitsT2S<double>(AiDotNetEngine.Current, new List<LayerBase<double>>(),
            new GptSoVitsT2SConfiguration(c.GetProperty("hidden_dim").GetInt32(), c.GetProperty("head").GetInt32(), c.GetProperty("n_layer").GetInt32(),
                c.GetProperty("vocab_size").GetInt32(), c.GetProperty("phoneme_vocab_size").GetInt32(), 0, 0.1), new Random(1));
        var state = r.GetProperty("state");
        model.Load((name, shape) =>
        {
            var values = Doubles(state.GetProperty(name));
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            return values;
        });
        return model;
    }

    private static Tensor<double> Bert(double[] values, int rows) => new(new[] { rows, 1024 }, new Vector<double>(values));

    private static void Close(double[] expected, Tensor<double> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual.GetFlat(i)) <= tolerance * (1 + Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual.GetFlat(i)}");
    }

    [Fact(Timeout = 300000)]
    public async Task TextRowsLogitsAndLosses_MatchTheRelease()
    {
        await Task.Yield();
        var r = Reference.Value;
        var model = Model();
        var a = Ints(r.GetProperty("phonemes_a"));
        var b = Ints(r.GetProperty("phonemes_b"));
        var bertA = Bert(Doubles(r.GetProperty("bert_a")), a.Length);
        var bertB = Bert(Doubles(r.GetProperty("bert_b")), b.Length);
        // The sinusoidal table is built in float32 (as the release builds it), where torch's sin and exp and .NET's may
        // differ in the last bit.
        Close(Doubles(r.GetProperty("text_rows")), model.TextRows(a, bertA), 1e-6, "text rows");

        var batch = new List<GptSoVitsT2S<double>.Example>
        {
            new(a, bertA, Ints(r.GetProperty("semantic_a"))),
            new(b, bertB, Ints(r.GetProperty("semantic_b"))),
        };
        var (logits, targets) = model.TeacherForced(batch, batch.Select(e => e.Semantic).ToList(), false, null);
        Close(Doubles(r.GetProperty("logits")), logits, 1e-5, "logits");
        var expectedTargets = r.GetProperty("targets").EnumerateArray().Select(Ints).ToArray();
        for (int i = 0; i < 2; i++)
            for (int t = 0; t < expectedTargets[i].Length; t++) Assert.Equal(expectedTargets[i][t], targets[i, t]);
        var ce = model.Loss(batch, dpo: false, training: false, new Random(2));
        Assert.True(Math.Abs(r.GetProperty("ce").GetDouble() - ce.GetFlat(0)) < 1e-5, $"ce {r.GetProperty("ce").GetDouble()} vs {ce.GetFlat(0)}");
        var rejected = r.GetProperty("rejected").EnumerateArray().Select(Ints).ToList();
        var dpo = model.Loss(batch, dpo: true, training: false, new Random(2), rejectedSequences: rejected);
        Assert.True(Math.Abs(r.GetProperty("dpo").GetDouble() - dpo.GetFlat(0)) < 1e-5, $"dpo {r.GetProperty("dpo").GetDouble()} vs {dpo.GetFlat(0)}");
    }

    [Fact(Timeout = 300000)]
    public async Task GenerationWithFixedNoise_MatchesInferPanel()
    {
        await Task.Yield();
        var r = Reference.Value;
        var model = Model();
        var a = Ints(r.GetProperty("phonemes_a"));
        var rows = model.TextRows(a, Bert(Doubles(r.GetProperty("bert_a")), a.Length));
        foreach (var g in r.GetProperty("generations").EnumerateArray())
        {
            var noise = g.GetProperty("noise").EnumerateArray().Select(Doubles).ToList();
            int[]? prompt = g.GetProperty("prompt").ValueKind == JsonValueKind.Null ? null : Ints(g.GetProperty("prompt"));
            var sampling = new GptSoVitsSampling(g.GetProperty("top_k").GetInt32(), g.GetProperty("top_p").GetDouble(),
                g.GetProperty("temperature").GetDouble(), g.GetProperty("repetition_penalty").GetDouble());
            var tokens = model.Generate(rows, prompt, sampling, new Random(3), (step, count) =>
            {
                Assert.Equal(noise[step].Length, count);
                return noise[step];
            });
            Assert.Equal(Ints(g.GetProperty("tokens")), tokens);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Probabilities_MatchLogitsToProbs()
    {
        await Task.Yield();
        foreach (var p in Reference.Value.GetProperty("probabilities").EnumerateArray())
        {
            var sampling = new GptSoVitsSampling(p.GetProperty("top_k").GetInt32(), p.GetProperty("top_p").GetDouble(),
                p.GetProperty("temperature").GetDouble(), p.GetProperty("repetition_penalty").GetDouble());
            var actual = GptSoVitsT2S<double>.Probabilities(Doubles(p.GetProperty("logits")), Ints(p.GetProperty("previous")), sampling, out _);
            var expected = Doubles(p.GetProperty("probabilities"));
            for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], actual[i], 12);
        }
    }
}
