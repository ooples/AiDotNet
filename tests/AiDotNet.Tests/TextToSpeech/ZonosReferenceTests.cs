using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Zonos against Zyphra/Zonos's model at a tiny size for both backbones (tools/reference-data/zonos_reference.py): the
/// transformer (zonos/backbone/_torch.py) and the hybrid (mamba_ssm 2.2.4's Mamba2 and attention blocks). Every release
/// tensor has a place; the phoneme ids, the guided conditioning prefix, every guided logit of a greedy generation after an
/// audio prefix (through the end-of-audio countdown) and the returned codes match; the training objective matches the
/// same computation on the release's modules.
/// </summary>
public class ZonosReferenceTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    internal static double[] ClosedForm(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new double[count];
        bool unitScale = name.EndsWith("norm.weight", StringComparison.Ordinal) || name.EndsWith("norm_f.weight", StringComparison.Ordinal)
            || name.EndsWith("norm2.weight", StringComparison.Ordinal) || name.EndsWith(".D", StringComparison.Ordinal);
        for (int i = 0; i < count; i++)
        {
            long k = i;
            double s = Math.Sin(0.37 * i + 0.11 * name.Length + 0.0013 * (k * k % 9973));
            values[i] = unitScale ? 1 + 0.2 * s
                : name.EndsWith("A_log", StringComparison.Ordinal) ? 0.3 * s
                : shape.Length == 2 ? 0.5 * s / Math.Sqrt(shape[1])
                : 0.5 * s;
        }
        return values;
    }

    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();
    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    private static int[,] Matrix(JsonElement rows)
    {
        var values = rows.EnumerateArray().Select(Ints).ToArray();
        var matrix = new int[values.Length, values.Length == 0 ? 0 : values[0].Length];
        for (int i = 0; i < values.Length; i++)
            for (int j = 0; j < values[i].Length; j++) matrix[i, j] = values[i][j];
        return matrix;
    }

    private static void Close(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(1e-300, expected.Where(v => !double.IsInfinity(v)).Max(Math.Abs)), worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            if (double.IsInfinity(expected[i]))
            {
                Assert.Equal(expected[i], actual[i]);
                continue;
            }
            worst = Math.Max(worst, Math.Abs(expected[i] - actual[i]));
        }
        Assert.True(worst <= tolerance * scale, $"{what} differs by {worst} (scale {scale})");
    }

    internal static ZonosOptions TinyOptions(bool hybrid)
    {
        var options = new ZonosOptions
        {
            HiddenDim = 32, Layers = hybrid ? 3 : 2, AttentionLayers = hybrid ? new[] { 1 } : null, Heads = 4, KeyValueHeads = 2,
            AttentionIntermediate = 128, RotaryDim = 8, RotaryInterleaved = !hybrid, StateDim = 8, MambaHeadDim = 8, Expand = 2,
            // The reference scans in chunks of 4; the SSD algorithm's chunking changes only the order of the arithmetic.
            ChunkSize = 3, Conditioners = (hybrid ? ZonosConditioner.Hybrid : ZonosConditioner.Transformer).ToList(),
            Backbone = hybrid ? ZonosBackbone.Hybrid : ZonosBackbone.Transformer, Temperature = 0, SampleRate = 16000, HopSize = 8,
            Codec = new AiDotNet.Audio.Effects.DACOptions
            {
                SampleRate = 16000, EncoderDim = 4, EncoderRates = [2, 4], DecoderDim = 16, DecoderRates = [4, 2], NumQuantizers = 9,
                CodebookSize = 1024, CodebookDim = 2, TargetBandwidthKbps = 180.0,
            },
        };
        return options;
    }

    internal static Zonos<double> Tiny(JsonElement reference, bool hybrid, double? endFactor = null)
    {
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new Zonos<double>(architecture, TinyOptions(hybrid));
        var shapes = reference.GetProperty("shapes").EnumerateObject().ToDictionary(p => p.Name, p => Ints(p.Value), StringComparer.Ordinal);
        double factor = endFactor ?? reference.GetProperty("end_factor").GetDouble();
        var used = model.LoadTorchWeights((name, shape) =>
        {
            var values = ClosedForm(name, shapes[name]);
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), values.Length);
            if (name == "heads.0.weight")
                for (int j = 0; j < shape[1]; j++) values[1024 * shape[1] + j] *= factor;
            return values;
        });
        Assert.Empty(shapes.Keys.Where(n => !used.Contains(n)));
        return model;
    }

    [Theory(Timeout = 600000)]
    [InlineData("transformer")]
    [InlineData("hybrid")]
    public async Task ConditioningGenerationAndLoss_MatchTheRelease(string backbone)
    {
        await Task.Yield();
        var reference = Load("zonos_reference.json").GetProperty(backbone);
        bool hybrid = backbone == "hybrid";
        var model = Tiny(reference, hybrid);

        var ids = ZonosPrefixConditioner<double>.PhonemeIds(reference.GetProperty("phonemes").GetString()!);
        Assert.Equal(Ints(reference.GetProperty("phoneme_ids")), ids);
        Assert.Equal(reference.GetProperty("language_id").GetInt32(), Zonos<double>.Languages.ToList().IndexOf("en-us"));

        var inputs = model.ConditionInputs(ids, Doubles(reference.GetProperty("speaker")));
        var prefix = model.PrepareConditioning(inputs);
        Assert.Equal(reference.GetProperty("prefix_length").GetInt32(), prefix.Shape[1]);
        Close(Doubles(reference.GetProperty("prefix")), prefix.ToArray(), 1e-12, "conditioning prefix");

        var observed = new List<double[]>();
        model.GuidedLogits = observed.Add;
        var codes = model.GenerateCodes(prefix, Matrix(reference.GetProperty("prefix_codes")), 10);
        model.GuidedLogits = null;
        var expectedLogits = reference.GetProperty("logits").EnumerateArray().Select(Doubles).ToArray();
        Assert.Equal(expectedLogits.Length, observed.Count);
        for (int i = 0; i < expectedLogits.Length; i++) Close(expectedLogits[i], observed[i], 1e-10, $"step {i}'s guided logits");
        Assert.Equal(Matrix(reference.GetProperty("generated")), codes);

        double loss = model.Loss(inputs, Matrix(reference.GetProperty("frames"))).ToArray()[0];
        double target = reference.GetProperty("loss").GetDouble();
        Assert.True(Math.Abs(loss - target) <= 1e-10 * target, $"loss {loss} vs {target}");

        // With codebook 0's end row zeroed the generation never ends: it runs to the length limit and through the end of
        // the delay pattern.
        var endless = Tiny(reference, hybrid, endFactor: 0);
        var seen = new List<double[]>();
        endless.GuidedLogits = seen.Add;
        var full = endless.GenerateCodes(endless.PrepareConditioning(endless.ConditionInputs(ids, Doubles(reference.GetProperty("speaker")))),
            Matrix(reference.GetProperty("prefix_codes")), 10);
        var expectedFull = reference.GetProperty("full_logits").EnumerateArray().Select(Doubles).ToArray();
        Assert.Equal(expectedFull.Length, seen.Count);
        for (int i = 0; i < expectedFull.Length; i++) Close(expectedFull[i], seen[i], 1e-10, $"step {i}'s guided logits to the limit");
        Assert.Equal(Matrix(reference.GetProperty("full_generated")), full);
    }

    private static double[] SpeakerClosedForm(string name, int[] shape)
    {
        int count = shape.Aggregate(1, (a, b) => a * b);
        var values = new double[count];
        for (int i = 0; i < count; i++)
        {
            long k = i;
            double s = Math.Sin(0.37 * i + 0.11 * name.Length + 0.0013 * (k * k % 9973));
            values[i] = name.EndsWith("running_var", StringComparison.Ordinal) ? 1 + 0.5 * Math.Abs(s)
                : name.EndsWith("running_mean", StringComparison.Ordinal) || name.EndsWith(".bias", StringComparison.Ordinal) ? 0.1 * s
                : shape.Length == 1 ? 1 + 0.2 * s
                : 0.5 * s / Math.Sqrt(count / (double)shape[0]);
        }
        return values;
    }

    private static AiDotNet.Tensors.LinearAlgebra.Tensor<double> Wave(double[] samples) =>
        new(new[] { samples.Length }, new AiDotNet.Tensors.LinearAlgebra.Vector<double>(samples));

    /// <summary>The speaker model (zonos/speaker_cloning.py, tools/reference-data/zonos_speaker_reference.py) at a tiny
    /// size: the resampled waveform's log mel features, the embedding and its LDA projection match.</summary>
    [Fact(Timeout = 600000)]
    public async Task SpeakerModel_MatchesTheRelease()
    {
        await Task.Yield();
        var reference = Load("zonos_speaker_reference.json").GetProperty("tiny");
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new Zonos<double>(architecture, TinyOptions(false));
        var shapes = reference.GetProperty("shapes").EnumerateObject().ToDictionary(p => p.Name, p => Ints(p.Value), StringComparer.Ordinal);
        model.LoadSpeakerWeights((name, shape) => SpeakerClosedForm(name, shapes[name]), shapes.Keys, new[] { 1, 1, 2, 1 }, 4);
        var audio = Wave(Doubles(reference.GetProperty("audio")));
        int rate = reference.GetProperty("rate").GetInt32();
        var encoder = model.SpeakerEncoder!;
        var prepared = encoder.Prepare(audio, rate);
        Assert.Equal(reference.GetProperty("samples16k").GetInt32(), prepared.Length);
        Close(Doubles(reference.GetProperty("features")), encoder.Features(prepared).ToArray(), 1e-10, "log mel features");
        var (embedding, lda) = encoder.Forward(prepared);
        Close(Doubles(reference.GetProperty("embedding")), embedding, 1e-9, "speaker embedding");
        Close(Doubles(reference.GetProperty("lda")), lda, 1e-9, "LDA projection");
        Close(Doubles(reference.GetProperty("lda")), model.SpeakerEmbedding(audio, rate), 1e-9, "SpeakerEmbedding");
    }

    /// <summary>With the released speaker model (<c>ZONOS_SPEAKER_DIR</c>: Zyphra/Zonos-v0.1-speaker-embedding), on three
    /// seconds of speech: the embedding and its LDA projection match.</summary>
    [Fact(Timeout = 1800000)]
    public async Task ReleasedSpeakerModel_MatchesTheRelease_WhenPresent()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("ZONOS_SPEAKER_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "ResNet293_SimAM_ASP_base.pt"))) return;
        var reference = Load("zonos_speaker_reference.json").GetProperty("released");
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new Zonos<double>(architecture, TinyOptions(false));
        model.LoadSpeakerModel(dir);
        var audio = Wave(Doubles(reference.GetProperty("audio")));
        var encoder = model.SpeakerEncoder!;
        var (embedding, lda) = encoder.Forward(encoder.Prepare(audio, reference.GetProperty("rate").GetInt32()));
        Close(Doubles(reference.GetProperty("embedding")), embedding, 1e-8, "speaker embedding");
        Close(Doubles(reference.GetProperty("lda")), lda, 1e-8, "LDA projection");
    }

    /// <summary>normalize_numbers (from VITS2, through inflect) for the languages other than Japanese
    /// (tools/reference-data/zonos_frontend_reference.py).</summary>
    [Fact(Timeout = 60000)]
    public async Task NumberNormalization_MatchesTheRelease()
    {
        await Task.Yield();
        foreach (var c in Load("zonos_frontend_reference.json").GetProperty("english").EnumerateArray())
            Assert.Equal(c.GetProperty("normalized").GetString(), ZonosFrontEnd.NormalizeNumbers(c.GetProperty("text").GetString()!));
    }

    /// <summary>kanjize's number2kanji (tools/reference-data/zonos_frontend_reference.py).</summary>
    [Fact(Timeout = 60000)]
    public async Task Kanjize_MatchesKanjize()
    {
        await Task.Yield();
        foreach (var c in Load("zonos_frontend_reference.json").GetProperty("kanjize").EnumerateArray())
            Assert.Equal(c.GetProperty("kanji").GetString(), Zonos<double>.Kanjize(System.Numerics.BigInteger.Parse(c.GetProperty("number").GetString()!)));
    }

    /// <summary>With Sudachi's full dictionary (<c>SUDACHI_DICT</c>: sudachidict_full's system.dic, <c>SUDACHI_RESOURCES</c>:
    /// SudachiPy's resources), Zonos's Japanese normalization gives SudachiPy's readings for every sentence.</summary>
    [Fact(Timeout = 600000)]
    public async Task JapaneseReadings_MatchSudachiPy_WhenTheDictionaryIsPresent()
    {
        await Task.Yield();
        string? dictionary = Environment.GetEnvironmentVariable("SUDACHI_DICT"), resources = Environment.GetEnvironmentVariable("SUDACHI_RESOURCES");
        if (dictionary is null || resources is null || !File.Exists(dictionary)) return;
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        var model = new Zonos<double>(architecture, TinyOptions(false));
        var tokenizer = new AiDotNet.TextToSpeech.FrontEnd.SudachiTokenizer(AiDotNet.TextToSpeech.FrontEnd.SudachiDictionary.Open(dictionary, resources));
        foreach (var c in Load("zonos_frontend_reference.json").GetProperty("cases").EnumerateArray())
        {
            string sentence = c.GetProperty("sentence").GetString()!;
            var expected = c.GetProperty("readings").EnumerateArray().Select(e => e.GetString()!).ToArray();
            Assert.Equal(expected, tokenizer.Readings(c.GetProperty("prepared").GetString()!).ToArray());
            Assert.Equal(c.GetProperty("normalized").GetString(), model.JapaneseReadings(sentence));
        }
    }
}
