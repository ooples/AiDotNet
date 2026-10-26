using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech.FlowDiffusion;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MaskGCT's networks against Amphion's own classes built small (tools/reference-data/maskgct_reference.py): the T2S
/// and S2A training forwards to 1e-9, their iterative decoding token for token (replaying the reference's Gumbel draws),
/// the semantic codec's codes, and the acoustic codec's codes and decoded audio.
/// </summary>
public class MaskGctReferenceTests
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
        double fanIn = shape.Length >= 2 ? (double)count / shape[0] : 1;
        bool Ends(string suffix) => name.EndsWith(suffix, StringComparison.Ordinal);
        return Enumerable.Range(0, count).Select(i =>
        {
            long k = i;
            double s = Math.Sin(0.37 * i + 0.11 * name.Length + 0.0013 * (k * k % 9973));
            if (Ends("to_weight.bias") || Ends("norm.weight")) return 1 + 0.2 * s;
            if (Ends("weight_g")) return 0.5 + 0.2 * s;
            if (Ends(".alpha")) return 1 + 0.5 * s;
            if (Ends("in_project.weight_v")) return 8 * s / Math.Sqrt(fanIn);
            if (shape.Length >= 2) return 0.6 * s / Math.Sqrt(fanIn);
            return 0.05 * s;
        }).ToArray();
    }

    private static Dictionary<string, int[]> Shapes(JsonElement e) =>
        e.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(v => v.GetInt32()).ToArray(), StringComparer.Ordinal);

    // A reader over closed-form weights that records what it served.
    private sealed class Reader
    {
        private readonly Dictionary<string, int[]> _shapes;
        public Reader(Dictionary<string, int[]> shapes) => _shapes = shapes;
        public HashSet<string> Used { get; } = new(StringComparer.Ordinal);
        public double[]? Read(string name)
        {
            if (!_shapes.TryGetValue(name, out var shape)) return null;
            Used.Add(name);
            return ClosedForm(name, shape);
        }
        public void AssertAllUsed(Func<string, bool>? exempt = null) =>
            Assert.Empty(_shapes.Keys.Where(n => !Used.Contains(n) && !(exempt?.Invoke(n) ?? false)));
    }

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();
    private static double[] Doubles(JsonElement e) => e.EnumerateArray().Select(v => v.GetDouble()).ToArray();

    private static int[,] Grid(JsonElement e)
    {
        var rows = e.EnumerateArray().Select(Ints).ToArray();
        var g = new int[rows.Length, rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[0].Length; j++) g[i, j] = rows[i][j];
        return g;
    }

    private static Func<int, double[]> Replay(JsonElement draws)
    {
        var queue = new Queue<double[]>(draws.EnumerateArray().Select(Doubles));
        return count =>
        {
            var next = queue.Dequeue();
            Assert.Equal(count, next.Length);
            return next;
        };
    }

    private static void AssertClose(JsonElement expected, Tensor<double> actual, double tolerance)
    {
        var rows = expected.EnumerateArray().Select(Doubles).ToArray();
        Assert.Equal(new[] { rows.Length, rows[0].Length }, actual.Shape.ToArray());
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[0].Length; j++)
                Assert.True(Math.Abs(rows[i][j] - actual[i, j]) <= tolerance, $"[{i}, {j}]: {rows[i][j]} vs {actual[i, j]}");
    }

    private static MaskGctTransformerConfiguration Backbone(JsonElement config) => new()
    {
        Hidden = config.GetProperty("hidden").GetInt32(), Layers = config.GetProperty("layers").GetInt32(), Heads = config.GetProperty("heads").GetInt32(),
    };

    [Fact(Timeout = 300000)]
    public async Task TextToSemantic_MatchesTheRelease()
    {
        await Task.Yield();
        var reference = Load("maskgct_reference.json");
        var config = reference.GetProperty("config");
        var model = new MaskGctTextToSemantic<double>(new CpuEngine(), new List<LayerBase<double>>(), Backbone(config),
            config.GetProperty("semantic_codes").GetInt32(), 1024, new Random(1));
        var reader = new Reader(Shapes(reference.GetProperty("t2s_shapes")));
        model.Load((name, _) => reader.Read(name));
        reader.AssertAllUsed();

        var training = reference.GetProperty("t2s_training");
        var mask = Doubles(training.GetProperty("mask")).Select(v => v == 1).ToArray();
        var logits = model.Logits(Ints(training.GetProperty("phones")), Array.Empty<int>(), Ints(training.GetProperty("x0")), mask,
            training.GetProperty("t").GetDouble());
        AssertClose(training.GetProperty("logits"), logits, 1e-9);

        var generation = reference.GetProperty("t2s_generation");
        var tokens = model.Generate(Ints(generation.GetProperty("phones")), Ints(generation.GetProperty("prompt")), generation.GetProperty("length").GetInt32(),
            new MaskGctDecoding { Steps = generation.GetProperty("steps").GetInt32(), Temperature = 0.9, Guidance = 2.5, GuidanceRescale = 0.75 },
            Replay(generation.GetProperty("draws")));
        Assert.Equal(Ints(generation.GetProperty("tokens")), tokens);
    }

    [Fact(Timeout = 300000)]
    public async Task SemanticToAcoustic_MatchesTheRelease()
    {
        await Task.Yield();
        var reference = Load("maskgct_reference.json");
        var config = reference.GetProperty("config");
        int semanticCodes = config.GetProperty("semantic_codes").GetInt32(), codes = config.GetProperty("acoustic_codes").GetInt32();
        int quantizers = config.GetProperty("quantizers").GetInt32();
        var shapes = reference.GetProperty("s2a_shapes");
        var first = new MaskGctSemanticToAcoustic<double>(new CpuEngine(), new List<LayerBase<double>>(), Backbone(config), 1, codes, semanticCodes, true, new Random(1));
        var full = new MaskGctSemanticToAcoustic<double>(new CpuEngine(), new List<LayerBase<double>>(), Backbone(config), quantizers, codes, semanticCodes, false, new Random(2));
        var firstReader = new Reader(Shapes(shapes.GetProperty("s2a_1layer")));
        var fullReader = new Reader(Shapes(shapes.GetProperty("s2a_full")));
        first.Load((name, _) => firstReader.Read(name));
        full.Load((name, _) => fullReader.Read(name));
        firstReader.AssertAllUsed();
        fullReader.AssertAllUsed();

        var training = reference.GetProperty("s2a_training");
        var mask = Doubles(training.GetProperty("mask")).Select(v => v == 1).ToArray();
        var logits = full.TrainingLogits(Ints(training.GetProperty("semantic")), Grid(training.GetProperty("codes")), training.GetProperty("layer").GetInt32(),
            training.GetProperty("prompt").GetInt32(), mask, training.GetProperty("t").GetDouble());
        AssertClose(training.GetProperty("logits"), logits, 1e-9);

        var generation = reference.GetProperty("s2a_generation");
        var semantic = Ints(generation.GetProperty("semantic"));
        var prompt = Grid(generation.GetProperty("prompt"));
        var draws = Replay(generation.GetProperty("draws"));
        var decoding = new MaskGctDecoding { Temperature = 1.5, Guidance = 2.5, GuidanceRescale = 0.75 };
        var layer1 = first.Generate(semantic, prompt, null, Ints(generation.GetProperty("steps_first")), decoding, draws);
        var expectedFirst = Grid(generation.GetProperty("first"));
        Assert.Equal(expectedFirst, layer1);
        var all = full.Generate(semantic, prompt, layer1, Ints(generation.GetProperty("steps_full")), decoding, draws);
        Assert.Equal(Grid(generation.GetProperty("full")), all);
    }

    [Fact(Timeout = 300000)]
    public async Task Codecs_MatchTheRelease()
    {
        await Task.Yield();
        var reference = Load("maskgct_reference.json");
        var config = reference.GetProperty("config");
        var sc = config.GetProperty("semantic_codec");
        var semanticCodec = new MaskGctSemanticCodec<double>(new CpuEngine(), new List<LayerBase<double>>(), sc.GetProperty("hidden").GetInt32(),
            sc.GetProperty("dim").GetInt32(), sc.GetProperty("intermediate").GetInt32(), sc.GetProperty("layers").GetInt32(), sc.GetProperty("codes").GetInt32(),
            sc.GetProperty("code_dim").GetInt32(), new Random(1));
        var semanticReader = new Reader(Shapes(reference.GetProperty("semantic_codec_shapes")));
        semanticCodec.Load((name, _) => semanticReader.Read(name));
        // The decoder only trains the codec.
        semanticReader.AssertAllUsed(n => n.StartsWith("decoder.", StringComparison.Ordinal));
        var s = reference.GetProperty("semantic_codec");
        semanticCodec.LoadStatistics(Doubles(s.GetProperty("mean")), Doubles(s.GetProperty("var")));
        var rows = s.GetProperty("features").EnumerateArray().Select(Doubles).ToArray();
        var features = new Tensor<double>(new[] { rows.Length, rows[0].Length });
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[0].Length; j++) features[i, j] = rows[i][j];
        Assert.Equal(Ints(s.GetProperty("codes")), semanticCodec.Encode(features));

        var ac = config.GetProperty("acoustic_codec");
        var acousticCodec = new MaskGctAcousticCodec<double>(new CpuEngine(), new List<LayerBase<double>>(), ac.GetProperty("encoder_dim").GetInt32(),
            Ints(ac.GetProperty("strides")), ac.GetProperty("latent").GetInt32(), config.GetProperty("quantizers").GetInt32(), config.GetProperty("acoustic_codes").GetInt32(),
            ac.GetProperty("code_dim").GetInt32(), ac.GetProperty("vocos_dim").GetInt32(), ac.GetProperty("vocos_intermediate").GetInt32(),
            ac.GetProperty("vocos_layers").GetInt32(), ac.GetProperty("fft").GetInt32(), ac.GetProperty("hop").GetInt32(), new Random(1));
        var encoderReader = new Reader(Shapes(reference.GetProperty("acoustic_encoder_shapes")));
        var decoderReader = new Reader(Shapes(reference.GetProperty("acoustic_decoder_shapes")));
        var a = reference.GetProperty("acoustic_codec");
        var window = Doubles(a.GetProperty("window"));
        acousticCodec.Load((name, _) => encoderReader.Read(name), (name, _) => name == "model.head.istft.window" ? window : decoderReader.Read(name));
        encoderReader.AssertAllUsed();
        decoderReader.AssertAllUsed();
        var expectedCodes = Grid(a.GetProperty("codes"));
        Assert.Equal(expectedCodes, acousticCodec.Encode(Doubles(a.GetProperty("audio"))));
        var expected = Doubles(a.GetProperty("decoded"));
        var decoded = acousticCodec.Decode(expectedCodes);
        Assert.Equal(expected.Length, decoded.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - decoded[i]) <= 1e-9, $"sample {i}: {expected[i]} vs {decoded[i]}");
    }
}
