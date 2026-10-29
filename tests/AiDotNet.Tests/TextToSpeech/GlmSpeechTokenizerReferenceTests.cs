using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// GLM-4-Voice's speech tokenizer against THUDM's speech_tokenizer (tools/reference-data/glm_speech_tokenizer_reference.py):
/// WhisperFeatureExtractor's features as extract_speech_token pads them; a tiny WhisperVQ model's codes and encoder states;
/// and one training step's ASR + quantize loss, EMA codebook update and dead-code restart.
/// </summary>
public class GlmSpeechTokenizerReferenceTests
{
    private static JObject Fixture()
    {
        const string fileName = "glm_speech_tokenizer_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();

    private static void Close(double[] expected, IReadOnlyList<double> actual, string what, double tolerance = 1e-9)
    {
        Assert.Equal(expected.Length, actual.Count);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])), $"{what}[{i}]: {expected[i]} vs {actual[i]}");
    }

    internal static GlmSpeechTokenizerConfiguration Tiny() => new()
    {
        Mels = 128, Dim = 16, Heads = 2, FeedForward = 32, EncoderLayers = 4, MaxSourcePositions = 64, PoolingKernel = 4, PoolingPosition = 2,
        QuantizePosition = 2, CodebookSize = 12, CausalBlockSize = 8, EmaDecay = 0.99, CommitCoefficient = 0.25, LossScale = 10.0,
        RestartInterval = 1, DecoderLayers = 2, Vocabulary = 40, MaxTargetPositions = 16,
    };

    [Fact(Timeout = 1800000)]
    public async System.Threading.Tasks.Task ReleasedTokenizer_MatchesRelease()
    {
        await System.Threading.Tasks.Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("GLM4VOICE_TOKENIZER_DIR");
        string checkpoint = dir is null ? "" : Path.Combine(dir, "model.safetensors");
        if (!File.Exists(checkpoint)) return;
        const string fileName = "glm_speech_tokenizer_realweights_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && d is not null; d = d.Parent)
            path = Path.Combine(d.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        var weights = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(checkpoint);
        var tokenizer = new GlmSpeechTokenizer<double>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, new List<LayerBase<double>>(),
            new GlmSpeechTokenizerConfiguration(), new Random(1), withTrainingStack: false);
        var used = new HashSet<string>();
        tokenizer.LoadEncoder((name, _) =>
        {
            used.Add(name);
            return weights[name].ToVector().Select(v => (double)v).ToArray();
        }, name => weights.ContainsKey(name));
        Assert.All(weights.Keys, k => Assert.Contains(k, used));
        weights.Clear();

        var audio = Doubles(f["audio"]!);
        using var _ = new NoGradScope<double>();
        Assert.Equal(f["tokens"]!.Select(v => (int)v).ToArray(), tokenizer.Tokenize(audio));
        var (features, valid) = tokenizer.Features(audio);
        var output = tokenizer.Encode(features, valid);
        Assert.Equal(f["codes"]!.Select(v => (int)v).ToArray(), output.Codes);
        Close(Doubles(f["hidden"]!), output.Hidden.ToArray(), "hidden", 1e-8);
    }

    [Fact]
    public void Tokenizer_MatchesRelease()
    {
        var f = Fixture();
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        var tokenizer = new GlmSpeechTokenizer<double>(AiDotNet.Tensors.Engines.AiDotNetEngine.Current, new List<LayerBase<double>>(), Tiny(), new Random(1), withTrainingStack: true);
        double[] Read(string name, int[] _)
        {
            used.Add(name);
            return Doubles(state[name] ?? throw new InvalidDataException(name));
        }
        tokenizer.LoadEncoder(Read, name => state[name] is not null, "model.encoder.");
        tokenizer.LoadDecoder(Read, "model.decoder.");
        Assert.All(state.Properties(), p => Assert.Contains(p.Name, used));
        // The fixture's initial EMA counts, set rare for some codes so the first update restarts them.
        var counts = Doubles(f["ema_count_initial"]!);
        Array.Copy(counts, tokenizer.EmaCount, counts.Length);

        // Features.
        var (features, valid) = tokenizer.Features(Doubles(f["audio"]!));
        Close(Doubles(f["features"]!), features.ToArray(), "features");
        Assert.Equal(136, valid.Length);
        Assert.Equal(130, valid.Count(v => v));

        // Eval: codes and the full encoder's states.
        var inputs = new Tensor<double>(new[] { 128, 64 }, new Vector<double>(Doubles(f["inputs"]!)));
        var mask = f["mask"]!.Select(v => (int)v == 1).ToArray();
        using (new NoGradScope<double>())
        {
            var eval = tokenizer.Encode(inputs, mask);
            Assert.Equal(f["eval_codes"]!.Select(v => (int)v).ToArray(), eval.Codes);
            Close(Doubles(f["eval_hidden"]!), eval.Hidden.ToArray(), "eval hidden");
        }

        // One training forward: the ASR cross-entropy plus the quantize loss, then the EMA update with its restart.
        var labels = f["labels"]!.Select(v => (int)v).ToArray();
        var decoderInput = new[] { 1 }.Concat(labels.Take(labels.Length - 1)).ToArray();
        using (new NoGradScope<double>())
        {
            var train = tokenizer.Encode(inputs, mask, training: true);
            var quantize = tokenizer.CommitmentLoss(train);
            var total = tokenizer.AsrLoss(decoderInput, labels, train.Hidden, train.Valid, false, null)[0] + quantize[0];
            Close(Doubles(f["quantize_loss"]!), new[] { quantize[0] }, "quantize loss");
            Close(Doubles(f["eval_ce"]!), new[] { total - quantize[0] }, "ASR cross-entropy");
            Close(Doubles(f["total_loss"]!), new[] { total }, "total loss");
            tokenizer.UpdateCodebook(train, (rows, count) => Enumerable.Range(0, count).ToArray());
        }
        Close(Doubles(f["ema_count"]!), tokenizer.EmaCount, "ema count");
        Close(Doubles(f["ema_weight"]!), tokenizer.EmaWeight, "ema weight");
        Close(Doubles(f["codebook"]!), tokenizer.Codebook.GetParameters().ToArray(), "codebook");
        using (new NoGradScope<double>())
            Assert.Equal(f["codes_after_update"]!.Select(v => (int)v).ToArray(), tokenizer.Encode(inputs, mask).Codes);
    }
}
