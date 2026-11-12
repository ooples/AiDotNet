using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.CodecBased;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Fish Speech 1.4 and 1.5 against the releases' code on tiny sizes (tools/reference-data/fishspeech_reference.py): the
/// training sample as each release packs it, the Dual-AR transformer's logits and cross-entropies on it, both DPO
/// objectives, the voice-cloning prompt as generate_long assembles it, the release's decoding at the most likely token, the
/// codes kept, and (1.5) Firefly-GAN's log mel, codes and decoded audio.
/// </summary>
public class FishSpeechReferenceTests
{
    private const double ModelScale = 3.0;

    private static JObject Fixture(string name)
    {
        string fileName = $"fishspeech{name}_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    /// <summary>The fixture's closed-form weights (fishspeech_reference.py's closed_form).</summary>
    internal static double[] ClosedForm(string name, int[] shape, double scale)
    {
        long count = shape.Aggregate(1L, (a, b) => a * b);
        var values = new double[count];
        double fanIn = shape.Length >= 2 ? count / (double)shape[0] : 1;
        for (long k = 0; k < count; k++)
        {
            double s = Math.Sin(0.37 * k + 0.11 * name.Length + 0.0013 * (k * k % 9973));
            values[k] = scale * (name.EndsWith("to_weight.bias", StringComparison.Ordinal) || name.EndsWith("norm.weight", StringComparison.Ordinal) ? 1 + 0.2 * s
                : name.EndsWith("project_in.weight", StringComparison.Ordinal) ? 8 * s / Math.Sqrt(fanIn)
                : name.EndsWith("original0", StringComparison.Ordinal) ? 0.5 + 0.2 * s
                : shape.Length >= 2 ? 0.6 * s / Math.Sqrt(fanIn)
                : 0.05 * s);
        }
        return values;
    }

    private static int[,] Grid(JToken t)
    {
        var rows = t.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        var grid = new int[rows.Length, rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[0].Length; j++) grid[i, j] = rows[i][j];
        return grid;
    }

    private static void Same(int[,] expected, int[,] actual, string what)
    {
        Assert.True(expected.GetLength(0) == actual.GetLength(0) && expected.GetLength(1) == actual.GetLength(1),
            $"{what}: shape [{expected.GetLength(0)}, {expected.GetLength(1)}] vs [{actual.GetLength(0)}, {actual.GetLength(1)}]");
        for (int i = 0; i < expected.GetLength(0); i++)
            for (int j = 0; j < expected.GetLength(1); j++)
                Assert.True(expected[i, j] == actual[i, j], $"{what}[{i}, {j}]: {expected[i, j]} vs {actual[i, j]}");
    }

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance = 1e-9)
    {
        var e = expected.Select(v => (double)v).ToArray();
        var a = actual.ToArray();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - a[i]) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {a[i]}");
    }

    /// <summary>The fixture's stub tokenizer: characters to ids 0-19, the chat and semantic tokens after them.</summary>
    private static (Func<string, int[]> Encode, Func<string, int> Id) Stub(bool v15)
    {
        var special = v15
            ? new Dictionary<string, int> { ["<|im_start|>"] = 20, ["<|im_end|>"] = 21, ["<|voice|>"] = 22, ["<|end_of_text|>"] = 23 }
            : new Dictionary<string, int> { ["<|im_start|>"] = 20, ["<|im_end|>"] = 21, ["<|semantic|>"] = 22 };
        if (v15) for (int i = 0; i < 16; i++) special[$"<|semantic:{i}|>"] = 24 + i;
        int[] Encode(string text)
        {
            var ids = new List<int>();
            foreach (var piece in Regex.Split(text, @"(<\|[^|]*\|>)"))
            {
                if (piece.Length == 0) continue;
                if (special.TryGetValue(piece, out int id)) ids.Add(id);
                else ids.AddRange(piece.Select(c => c % 20));
            }
            return ids.ToArray();
        }
        return (Encode, token => special[token]);
    }

    [Theory(Timeout = 600000)]
    [InlineData("15")]
    [InlineData("14")]
    public async Task DualAr_MatchesTheRelease(string version)
    {
        await Task.Yield();
        bool v15 = version == "15";
        var f = Fixture(version);
        var shapes = (JObject)f["shapes"]!;
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var (encode, id) = Stub(v15);
        var format = new FishSpeechFormat(v15 ? FishSpeechVersion.V15 : FishSpeechVersion.V14, encode, id, 3, 16);
        var config = new FishDualArConfiguration(v15 ? 40 : 24, 32, 2, 4, 2, 48, 2, 16, 3, 192, 1e6, 1e-6, 0.0);
        FishDualAr<double> Build(double scale)
        {
            var net = new FishDualAr<double>(engine, new List<LayerBase<double>>(), config, new Random(1));
            net.Load((name, _) => ClosedForm(name, shapes[name]!.Select(v => (int)v).ToArray(), scale));
            return net;
        }
        var net = Build(ModelScale);
        var reference = Build(0.9 * ModelScale);

        var codes = new int[3, 6];
        for (int b = 0; b < 3; b++)
            for (int j = 0; j < 6; j++) codes[b, j] = (3 * j + 5 * b) % 13;
        var (tokens, labels) = format.TrainingSample("Hi there.", codes);
        Same(Grid(f["train_tokens"]!), tokens, "training tokens");
        Same(Grid(f["train_labels"]!), labels, "training labels");

        using var _ = new NoGradScope<double>();
        var (tokenLogits, codebookLogits) = net.Forward(tokens, format.IsSemantic);
        Close(f["token_logits"]!, tokenLogits, "token logits");
        Close(f["codebook_logits"]!, codebookLogits, "codebook logits");
        var (tokenLoss, codebookLoss) = FishSpeechObjective<double>.Losses(engine, net, format, tokens, labels);
        Close(f["base_loss"]!, tokenLoss, "token loss");
        Close(f["semantic_loss"]!, codebookLoss, "codebook loss");

        var rejectedCodes = new int[3, 6];
        for (int b = 0; b < 3; b++)
            for (int j = 0; j < 6; j++) rejectedCodes[b, j] = (7 * j + 2 * b + 1) % 11;
        var (rejectedTokens, rejectedLabels) = format.TrainingSample("Hi there.", rejectedCodes);
        Same(Grid(f["rejected_tokens"]!), rejectedTokens, "rejected tokens");
        var chosen = FishSpeechObjective<double>.SequenceLogProbability(engine, net, format, tokens, labels);
        var rejected = FishSpeechObjective<double>.SequenceLogProbability(engine, net, format, rejectedTokens, rejectedLabels);
        Close(f["policy_logps"]!, engine.TensorConcatenate(new[] { chosen, rejected }, 0), "policy log-probabilities");
        double referenceChosen = FishSpeechObjective<double>.SequenceLogProbability(engine, reference, format, tokens, labels)[0];
        double referenceRejected = FishSpeechObjective<double>.SequenceLogProbability(engine, reference, format, rejectedTokens, rejectedLabels)[0];
        Close(f["reference_logps"]!, new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { referenceChosen, referenceRejected })), "reference log-probabilities");
        Close(f["dpo_loss"]!, FishSpeechObjective<double>.Dpo(engine, chosen, rejected, referenceChosen, referenceRejected, 0.1), "DPO loss");
        Close(f["reference_free_dpo_loss"]!, FishSpeechObjective<double>.ReferenceFreeDpo(engine, net, format, tokens, labels, rejectedTokens, rejectedLabels, 0.2),
            "reference-free DPO loss");

        var parts = new List<int[,]>();
        if (format.SystemPrompt() is { } system) parts.Add(system);
        parts.Add(format.Segment("Hello.", codes));
        parts.Add(format.Segment("Good day.", null));
        int total = parts.Sum(p => p.GetLength(1));
        var prompt = new int[4, total];
        int at = 0;
        foreach (var p in parts)
        {
            for (int r = 0; r < 4; r++)
                for (int t = 0; t < p.GetLength(1); t++) prompt[r, at + t] = p[r, t];
            at += p.GetLength(1);
        }
        Same(Grid(f["prompt"]!), prompt, "prompt");
        var generated = FishSpeechGenerator<double>.Generate(engine, net, format, prompt, 10, 0.7, 1e-9, 1.5, new Random(1));
        Same(Grid(f["generated"]!), generated, "generated");
        var kept = Grid(f["kept_codes"]!);
        if (kept.Cast<int>().Any(c => c < 0))
            // generate_long asserts the codes are non-negative; a non-semantic frame's zeroed codes fail it, as here.
            Assert.Throws<InvalidOperationException>(() => format.KeptCodes(generated, total));
        else
            Same(kept, format.KeptCodes(generated, total), "kept codes");
    }

    [Fact(Timeout = 600000)]
    public async Task Firefly_MatchesTheRelease()
    {
        await Task.Yield();
        var f = Fixture("15");
        var shapes = (JObject)f["shapes"]!;
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var config = new FishFireflyConfiguration(8000, 64, 16, 12, new[] { 1, 1, 1, 1 }, new[] { 8, 8, 8, 16 }, 16, 2, new[] { 8, 5, 5, 5 }, new[] { 2, 2 },
            new[] { 2, 2, 2, 2 }, new[] { 4, 4, 4, 4 }, new[] { 3, 7 }, new[] { new[] { 1, 3, 5 }, new[] { 1, 3, 5 } }, 16, 13, 13);
        var codec = new FishFireflyCodec<double>(engine, new List<LayerBase<double>>(), config, new Random(1));
        var used = new HashSet<string>();
        codec.Load(name =>
        {
            if (shapes["codec." + name] is not JArray shape) return null;
            used.Add("codec." + name);
            return ClosedForm(name, shape.Select(v => (int)v).ToArray(), 1.0);
        });
        Assert.All(shapes.Properties().Select(p => p.Name).Where(n => n.StartsWith("codec.", StringComparison.Ordinal)), n => Assert.Contains(n, used));

        using var _ = new NoGradScope<double>();
        var audio = f["audio"]!.Select(v => (double)v).ToArray();
        var wave = new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio));
        var mel = codec.Mel(wave);
        Close(f["mel"]!, mel, "mel");
        var codes = codec.Encode(wave);
        Same(Grid(f["codec_codes"]!), codes, "codes");
        Close(f["decoded"]!, codec.Decode(codes), "decoded");
    }

    /// <summary>
    /// Fish Speech 1.5 with the released weights against the release (tools/reference-data/fishspeech_realweights_reference.py),
    /// where the checkpoint is present (<c>FISHSPEECH15_DIR</c>: fishaudio/fish-speech-1.5 with model.pth, the Firefly-GAN
    /// generator and tokenizer.tiktoken): Firefly-GAN's mel, codes and audio of a chirp, the voice-cloning prompt with the real
    /// tokenizer, the transformer's logits on it, and generation at the most likely token.
    /// </summary>
    [Fact(Timeout = 3600000)]
    public async Task ReleasedCheckpoint_MatchesTheRelease_WhenPresent()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("FISHSPEECH15_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "model.pth"))) return;
        var f = Fixture("15_realweights");
        var architecture = new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 1, outputSize: 1);
        var model = new FishSpeechV15<double>(architecture, new FishSpeechV15Options());
        model.DisableAutoStreaming();
        model.LoadReferenceWeights(dir);
        GC.Collect();
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;

        using var _ = new NoGradScope<double>();
        var audio = f["audio"]!.Select(v => (double)v).ToArray();
        var wave = new Tensor<double>(new[] { audio.Length }, new Vector<double>(audio));
        Close(f["mel"]!, model.Codec.Mel(wave), "mel", 1e-9);
        var codes = model.EncodeAudio(wave);
        Same(Grid(f["codec_codes"]!), codes, "codec codes");
        Close(f["decoded"]!, model.DecodeAudio(codes), "decoded", 1e-7);

        var format = model.Format;
        var parts = new List<int[,]> { format.SystemPrompt()!, format.Segment(model.CleanText("Hello there, this is a test."), codes),
            format.Segment(model.CleanText("Fish speech speaks."), null) };
        int total = parts.Sum(p => p.GetLength(1)), rows = parts[0].GetLength(0), at = 0;
        var prompt = new int[rows, total];
        foreach (var p in parts)
        {
            for (int r = 0; r < rows; r++)
                for (int t = 0; t < p.GetLength(1); t++) prompt[r, at + t] = p[r, t];
            at += p.GetLength(1);
        }
        Same(Grid(f["prompt"]!), prompt, "prompt");
        var (tokenLogits, codebookLogits) = model.Transformer.Forward(prompt, format.IsSemantic);
        Close(f["token_logits"]!, engine.TensorNarrow(tokenLogits, 0, total - 1, 1), "token logits", 1e-7);
        var positions = f["positions"]!.Select(v => (int)v).ToArray();
        Close(f["codebook_logits"]!, engine.TensorConcatenate(positions.Select(t => engine.TensorNarrow(codebookLogits, 0, t, 1)).ToArray(), 0),
            "codebook logits", 1e-7);
        var generated = FishSpeechGenerator<double>.Generate(engine, model.Transformer, format, prompt, 4, 0.7, 1e-9, 1.2, new Random(1));
        var expected = Grid(f["generated"]!);
        var tail = new int[rows, generated.GetLength(1) - total];
        for (int r = 0; r < rows; r++)
            for (int t = 0; t < tail.GetLength(1); t++) tail[r, t] = generated[r, total + t];
        Same(expected, tail, "generated");
    }
}
