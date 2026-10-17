using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Audio.Generation;
using AiDotNet.Audio.MusicGen;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// MusicGen's language model reproduces audiocraft's <c>LMModel</c> (<c>tools/reference-data/musicgen_reference.py</c>):
/// with the reference's weights loaded, a tiny configuration gives the same text conditioning, the same logits at every
/// position the delay pattern predicts, the same training cross-entropy, and the same codes generated greedily with
/// classifier-free guidance 3.0.
/// </summary>
public class MusicGenReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "musicgen_reference.json";
        string relative = Path.Combine("Audio", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    internal static MusicGenOptions TinyOptions() => new()
    {
        ModelSize = MusicGenModelSize.Small, ModelDim = 16, NumHeads = 2, NumLayers = 2,
        TextVocabularySize = 64, TextModelDim = 8, TextLayers = 2, TextHeads = 2, TextKeyValueDim = 4, TextFeedForwardDim = 16,
        Codec = new EnCodecOptions
        {
            SampleRate = 32000, Filters = 4, Ratios = [8, 5, 4, 4], Dimension = 8, ResidualKernelSizes = [3, 1], TrueSkip = true,
            Causal = false, ReflectPadding = false, NumQuantizers = 4, CodebookSize = 16, TargetBandwidths = [0.8],
            TargetBandwidthKbps = 0.8,
        },
        UseSampling = false, Seed = 3,
    };

    private static MusicGenModel<double> LoadReference(JsonElement f)
    {
        var model = new MusicGenModel<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 5 }, TinyOptions());
        double[] Read(JsonElement table, string name, int[] shape)
        {
            var values = table.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            return values;
        }
        var lm = f.GetProperty("lm");
        var projection = f.GetProperty("projection");
        model.LoadLanguageModelWeights((name, shape) => name switch
        {
            "condition_provider.conditioners.description.output_proj.weight" => Read(projection, "weight", shape),
            "condition_provider.conditioners.description.output_proj.bias" => Read(projection, "bias", shape),
            _ => Read(lm, name, shape),
        });
        var t5 = f.GetProperty("t5");
        model.LoadTextEncoderWeights((name, shape) => Read(t5, name, shape));
        return model;
    }

    private static Tensor<double> Tokens(JsonElement f)
    {
        var ids = f.GetProperty("tokens").EnumerateArray().Select(e => (double)e.GetInt32()).ToArray();
        return new Tensor<double>(new[] { ids.Length }, new Vector<double>(ids));
    }

    private static int[,] Codes(JsonElement f, string name)
    {
        var rows = f.GetProperty(name).EnumerateArray().Select(r => r.EnumerateArray().Select(e => e.GetInt32()).ToArray()).ToArray();
        var codes = new int[rows.Length, rows[0].Length];
        for (int q = 0; q < rows.Length; q++)
            for (int t = 0; t < rows[q].Length; t++) codes[q, t] = rows[q][t];
        return codes;
    }

    [Fact(Timeout = 120000)]
    public async Task Conditioning_Logits_AndLoss_MatchTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        using var model = LoadReference(f);
        var computed = model.ConditionForTests(Tokens(f));
        // Hugging Face's T5LayerNorm takes the mean square in float32 even in a float64 model, so the conditioning is
        // compared with the reference's run with that one step in float64; the language model is then checked on
        // the stock reference conditioning.
        var expectedCondition = f.GetProperty("condition_float64_norm").EnumerateArray().Select(r => r.EnumerateArray().Select(e => e.GetDouble()).ToArray()).ToArray();
        Assert.Equal(new[] { expectedCondition.Length, 16 }, computed.Shape.ToArray());
        for (int i = 0; i < expectedCondition.Length; i++)
            for (int j = 0; j < 16; j++) Assert.True(Math.Abs(expectedCondition[i][j] - computed[i, j]) < 1e-9, $"condition[{i}, {j}]: {computed[i, j]} vs {expectedCondition[i][j]}");
        var condition = ReferenceCondition(f);

        var core = model.CoreForTests;
        var codes = Codes(f, "codes");
        int k = codes.GetLength(0), frames = codes.GetLength(1);
        var logits = core.Logits(MusicGenDelayPattern.TrainingSequence(codes, core.Configuration.Delays, core.Special), condition, false, new Random(0));
        var mask = f.GetProperty("mask").EnumerateArray().Select(r => r.EnumerateArray().Select(e => e.GetInt32()).ToArray()).ToArray();
        var expected = f.GetProperty("logits");
        int compared = 0;
        for (int q = 0; q < k; q++)
            for (int t = 0; t < frames; t++)
            {
                Assert.Equal(mask[q][t] == 1, MusicGenDelayPattern.Predicted(t, q, frames, core.Configuration.Delays));
                if (mask[q][t] == 0) continue;
                var row = expected[q][t].EnumerateArray().Select(e => e.GetDouble()).ToArray();
                for (int v = 0; v < row.Length; v++) Assert.Equal(row[v], logits[q, t + core.Configuration.Delays[q], v], 9);
                compared++;
            }
        Assert.True(compared > frames, $"Only {compared} positions compared.");
        Assert.Equal(f.GetProperty("loss").GetDouble(), core.Loss(codes, condition, false, new Random(0))[0], 9);
    }

    private static Tensor<double> ReferenceCondition(JsonElement f)
    {
        var rows = f.GetProperty("condition").EnumerateArray().Select(r => r.EnumerateArray().Select(e => e.GetDouble()).ToArray()).ToArray();
        var condition = new Tensor<double>(new[] { rows.Length, rows[0].Length });
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[i].Length; j++) condition[i, j] = rows[i][j];
        return condition;
    }

    [Fact(Timeout = 120000)]
    public async Task GreedyGuidedGeneration_MatchesTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        using var model = LoadReference(f);
        var core = model.CoreForTests;
        var condition = ReferenceCondition(f);
        var expected = Codes(f, "generated");
        int frames = expected.GetLength(1);
        var generated = core.Generate(condition, model.ConditionForTests(null), 3.0, frames, logits =>
        {
            int best = 0;
            for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
            return best;
        });
        for (int q = 0; q < expected.GetLength(0); q++)
            for (int t = 0; t < frames; t++) Assert.Equal(expected[q, t], generated[q, t]);
        // The model's own greedy path (no sampling) is the same generation.
        var viaModel = model.GenerateCodes(Tokens(f), frames, seed: 0);
        for (int q = 0; q < expected.GetLength(0); q++)
            for (int t = 0; t < frames; t++) Assert.Equal(expected[q, t], viaModel[q, t]);
    }

    [Fact(Timeout = 120000)]
    public async Task MelodyModel_Chroma_PrefixLogits_Loss_AndGeneration_MatchTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        var m = f.GetProperty("melody");
        // The chromagram of the C4-then-A4 tone, one-hot per frame.
        var tone = m.GetProperty("tone");
        int sr = tone.GetProperty("sample_rate").GetInt32(), n = tone.GetProperty("samples").GetInt32();
        var frequencies = tone.GetProperty("frequencies").EnumerateArray().Select(e => e.GetDouble()).ToArray();
        double amplitude = tone.GetProperty("amplitude").GetDouble();
        var samples = Enumerable.Range(0, n).Select(i => amplitude * Math.Sin(2 * Math.PI * (i < n / 2 ? frequencies[0] : frequencies[1]) * i / sr)).ToArray();
        var chroma = MusicGenChroma.Extract(samples, sr);
        var expectedChroma = m.GetProperty("chroma").EnumerateArray().Select(r => r.EnumerateArray().Select(e => e.GetDouble()).ToArray()).ToArray();
        Assert.Equal(expectedChroma.Length, chroma.GetLength(0));
        for (int t = 0; t < expectedChroma.Length; t++)
            for (int c = 0; c < 12; c++) Assert.Equal(expectedChroma[t][c], chroma[t, c]);

        var options = TinyOptions();
        options.MelodyConditioning = true;
        using var model = new MusicGenModel<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 5 }, options);
        double[] Read(JsonElement table, string name, int[] shape) => table.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();
        var lm = m.GetProperty("lm");
        model.LoadLanguageModelWeights((name, shape) => name switch
        {
            "condition_provider.conditioners.description.output_proj.weight" => Read(f.GetProperty("projection"), "weight", shape),
            "condition_provider.conditioners.description.output_proj.bias" => Read(f.GetProperty("projection"), "bias", shape),
            "condition_provider.conditioners.self_wav.output_proj.weight" => Read(m.GetProperty("projection"), "weight", shape),
            "condition_provider.conditioners.self_wav.output_proj.bias" => Read(m.GetProperty("projection"), "bias", shape),
            _ => Read(lm, name, shape),
        });
        model.LoadTextEncoderWeights((name, shape) => Read(f.GetProperty("t5"), name, shape));
        // The prefix [text; chroma]: the chroma rows exactly, the text rows to T5LayerNorm's float32 rounding.
        var rows = m.GetProperty("prefix").EnumerateArray().Select(r => r.EnumerateArray().Select(e => e.GetDouble()).ToArray()).ToArray();
        var prefix = new Tensor<double>(new[] { rows.Length, 16 });
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < 16; j++) prefix[i, j] = rows[i][j];
        var melodyTensor = new Tensor<double>(new[] { n }, new Vector<double>(samples));
        var computed = model.MelodyConditionForTests(Tokens(f), melodyTensor);
        Assert.Equal(prefix.Shape.ToArray(), computed.Shape.ToArray());
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < 16; j++) Assert.True(Math.Abs(prefix[i, j] - computed[i, j]) < 1e-6, $"prefix[{i}, {j}]");

        var core = model.CoreForTests;
        Assert.False(core.Configuration.CrossAttention);
        var codes = Codes(f, "codes");
        int k = codes.GetLength(0), frames = codes.GetLength(1);
        var logits = core.Logits(MusicGenDelayPattern.TrainingSequence(codes, core.Configuration.Delays, core.Special), prefix, false, new Random(0));
        Assert.Equal(frames + 1, logits.Shape[1]);
        var mask = m.GetProperty("mask").EnumerateArray().Select(r => r.EnumerateArray().Select(e => e.GetInt32()).ToArray()).ToArray();
        var expected = m.GetProperty("logits");
        for (int q = 0; q < k; q++)
            for (int t = 0; t < frames; t++)
            {
                if (mask[q][t] == 0) continue;
                var row = expected[q][t].EnumerateArray().Select(e => e.GetDouble()).ToArray();
                for (int v = 0; v < row.Length; v++) Assert.Equal(row[v], logits[q, t + core.Configuration.Delays[q], v], 9);
            }
        Assert.Equal(m.GetProperty("loss").GetDouble(), core.Loss(codes, prefix, false, new Random(0))[0], 9);
        var expectedCodes = Codes(m, "generated");
        var generated = core.Generate(prefix, new Tensor<double>(prefix.Shape.ToArray()), 3.0, expectedCodes.GetLength(1), l =>
        {
            int best = 0;
            for (int i = 1; i < l.Length; i++) if (l[i] > l[best]) best = i;
            return best;
        });
        for (int q = 0; q < k; q++)
            for (int t = 0; t < expectedCodes.GetLength(1); t++) Assert.Equal(expectedCodes[q, t], generated[q, t]);
    }

    [Fact(Timeout = 120000)]
    public async Task StereoPartialDelayModel_Logits_Loss_AndGeneration_MatchTheReference()
    {
        await Task.Yield();
        var f = Fixture();
        var st = f.GetProperty("stereo");
        var options = TinyOptions();
        options.Stereo = true;
        Assert.Equal(st.GetProperty("delays").EnumerateArray().Select(e => e.GetInt32()).ToArray(), options.ResolveDelays());
        using var model = new MusicGenModel<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
            NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1) { RandomSeed = 5 }, options);
        double[] Read(JsonElement table, string name) => table.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();
        model.LoadLanguageModelWeights((name, shape) => name switch
        {
            "condition_provider.conditioners.description.output_proj.weight" => Read(f.GetProperty("projection"), "weight"),
            "condition_provider.conditioners.description.output_proj.bias" => Read(f.GetProperty("projection"), "bias"),
            _ => Read(st.GetProperty("lm"), name),
        });
        var core = model.CoreForTests;
        Assert.Equal(8, core.Configuration.Codebooks);
        var condition = ReferenceCondition(f);
        var codes = Codes(st, "codes");
        int frames = codes.GetLength(1);
        var logits = core.Logits(MusicGenDelayPattern.TrainingSequence(codes, core.Configuration.Delays, core.Special), condition, false, new Random(0));
        var mask = st.GetProperty("mask").EnumerateArray().Select(r => r.EnumerateArray().Select(e => e.GetInt32()).ToArray()).ToArray();
        var expected = st.GetProperty("logits");
        for (int q = 0; q < 8; q++)
            for (int t = 0; t < frames; t++)
            {
                Assert.Equal(mask[q][t] == 1, MusicGenDelayPattern.Predicted(t, q, frames, core.Configuration.Delays));
                if (mask[q][t] == 0) continue;
                var row = expected[q][t].EnumerateArray().Select(e => e.GetDouble()).ToArray();
                for (int v = 0; v < row.Length; v++) Assert.Equal(row[v], logits[q, t + core.Configuration.Delays[q], v], 9);
            }
        Assert.Equal(st.GetProperty("loss").GetDouble(), core.Loss(codes, condition, false, new Random(0))[0], 9);
        var expectedCodes = Codes(st, "generated");
        var generated = core.Generate(condition, model.ConditionForTests(null), 3.0, expectedCodes.GetLength(1), l =>
        {
            int best = 0;
            for (int i = 1; i < l.Length; i++) if (l[i] > l[best]) best = i;
            return best;
        });
        for (int q = 0; q < 8; q++)
            for (int t = 0; t < expectedCodes.GetLength(1); t++) Assert.Equal(expectedCodes[q, t], generated[q, t]);
        // Stereo output is [2, samples]: each channel's codes decoded separately.
        Assert.Equal(new[] { 2, 5 * 640 }, model.GenerateMusic("a", durationSeconds: 0.1).Shape.ToArray());
    }
}
