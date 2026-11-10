using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Moshi's RQ-Transformer against the release's LMModel (<c>tools/reference-data/moshi_lm_reference.py</c>): the training
/// text and audio logits with their masks for the paper's all-stream Depth Transformer and the release's Moshi-only one,
/// and the Eq. 7 loss.
/// </summary>
public class MoshiLmReferenceTests
{
    private static readonly Lazy<JObject> Data = new(() =>
    {
        const string name = "moshi_lm_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", name);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", name);
        return JObject.Parse(File.ReadAllText(path));
    });

    internal static Dictionary<string, double[]> State(string base64)
    {
        var bytes = Convert.FromBase64String(base64);
        long headerLength = BitConverter.ToInt64(bytes, 0);
        var header = JObject.Parse(Encoding.UTF8.GetString(bytes, 8, (int)headerLength));
        var result = new Dictionary<string, double[]>();
        int start = 8 + (int)headerLength;
        foreach (var entry in header.Properties())
        {
            if (entry.Name == "__metadata__" || (string)entry.Value["dtype"]! != "F64") continue;
            var offsets = entry.Value["data_offsets"]!.Select(v => (int)v).ToArray();
            result[entry.Name] = Enumerable.Range(0, (offsets[1] - offsets[0]) / 8).Select(i => BitConverter.ToDouble(bytes, start + offsets[0] + 8 * i)).ToArray();
        }
        return result;
    }

    private static MoshiLanguageModel<double> Model(int depthStreams, string state,
        List<LayerBase<double>>? layers = null)
    {
        var f = Data.Value;
        var c = f["config"]!;
        int depthFeedForward = (int)c["depformer_dim_feedforward"]!;
        var configuration = new MoshiLanguageModelConfiguration
        {
            Codebooks = 3, AudioStreams = 6, DepthStreams = depthStreams, AudioCardinality = (int)c["card"]!, TextCardinality = (int)c["text_card"]!,
            Delays = f["delays"]!.Select(v => (int)v).ToArray(),
            Temporal = new MoshiTransformerConfiguration
            {
                Dim = (int)c["dim"]!, Heads = (int)c["num_heads"]!, Layers = (int)c["num_layers"]!, FeedForward = (int)(4.125 * (int)c["dim"]!),
                Context = (int)c["context"]!,
            },
            Depth = new MoshiTransformerConfiguration
            {
                Dim = (int)c["depformer_dim"]!, Heads = (int)c["depformer_num_heads"]!, Layers = (int)c["depformer_num_layers"]!,
                FeedForward = depthFeedForward, Context = 0, Rotary = false, WeightsPerStep = depthStreams,
            },
        };
        var model = new MoshiLanguageModel<double>(AiDotNetEngine.Current, layers ?? new List<LayerBase<double>>(), configuration, new Random(0));
        var weights = State((string)f[state]!);
        model.LoadTorchWeights(name => weights.TryGetValue(name, out var v) ? v : null);
        return model;
    }

    private static int[,] Codes()
    {
        var rows = Data.Value["codes"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        var codes = new int[rows.Length, rows[0].Length];
        for (int k = 0; k < rows.Length; k++) for (int t = 0; t < rows[0].Length; t++) codes[k, t] = rows[k][t];
        return codes;
    }

    private static void Close(JToken expected, Tensor<double> actual, string what, Func<int, bool> scored)
    {
        var values = expected.Select(v => (double)v).ToArray();
        Assert.Equal(values.Length, actual.Length);
        for (int i = 0; i < values.Length; i++)
            if (scored(i)) Assert.True(Math.Abs(values[i] - actual[i]) <= 1e-10 * Math.Max(1, Math.Abs(values[i])), $"{what}[{i}]: {values[i]:R} vs {actual[i]:R}");
    }

    [Theory(Timeout = 120000)]
    [InlineData("paper", 6)]
    [InlineData("release", 3)]
    public async Task TrainingLogits_MatchTheRelease(string name, int depthStreams)
    {
        await Task.Yield();
        var f = Data.Value;
        var model = Model(depthStreams, $"{name}_state");
        var codes = Codes();
        int length = codes.GetLength(1), text = (int)f["config"]!["text_card"]!, card = (int)f["config"]!["card"]!;
        using var _ = new NoGradScope<double>();
        var (textLogits, textMask, audioLogits, audioMask) = model.Forward(codes);
        var expectedTextMask = f[$"{name}_text_mask"]!.Select(v => (bool)v).ToArray();
        Assert.Equal(expectedTextMask, textMask);
        var expectedAudioMask = f[$"{name}_audio_mask"]!.Select(r => r.Select(v => (bool)v).ToArray()).ToArray();
        for (int s = 0; s < depthStreams; s++)
            Assert.Equal(expectedAudioMask[s], Enumerable.Range(0, length).Select(t => audioMask[s, t]).ToArray());
        // Logits are compared where the release defines them (NaN, written as 0, elsewhere).
        Close(f[$"{name}_text_logits"]!, textLogits, "text logits", i => i / text < length - f["delays"]![0]!.Value<int>());
        Close(f[$"{name}_audio_logits"]!, audioLogits, "audio logits", i =>
        {
            int s = i / (length * card), t = i / card % length;
            int delay = f["delays"]![1 + s]!.Value<int>();
            return t < length - delay;
        });
    }

    [Theory(Timeout = 120000)]
    [InlineData("paper", 6)]
    [InlineData("release", 3)]
    public async Task GreedyStreaming_MatchesTheReleaseLmGen(string name, int depthStreams)
    {
        await Task.Yield();
        var f = Data.Value;
        var model = Model(depthStreams, $"{name}_state");
        var streamer = new MoshiStreamer<double>(model, new MoshiSampling { UseSampling = false }, new Random(0));
        var user = f["user"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        var expectedFrames = f[$"{name}_stream_frames"]!.ToArray();
        var expectedLogits = f[$"{name}_stream_text_logits"]!.Select(v => (double)v).ToArray();
        int text = (int)f["config"]!["text_card"]!;
        using var _ = new NoGradScope<double>();
        for (int step = 0; step < expectedFrames.Length; step++)
        {
            int[]? input = depthStreams == 3 ? new[] { user[0][step], user[1][step], user[2][step] } : null;
            var frame = streamer.Step(input);
            for (int i = 0; i < text; i++)
            {
                double want = expectedLogits[step * text + i];
                Assert.True(Math.Abs(want - streamer.LastTextLogits![i]) <= 1e-10 * Math.Max(1, Math.Abs(want)), $"step {step} text logit {i}");
            }
            if (expectedFrames[step].Type == JTokenType.Null) Assert.Null(frame);
            else Assert.Equal(expectedFrames[step].Select(v => (int)v).ToArray(), frame);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Eq7Loss_MatchesTheReleaseLoss()
    {
        await Task.Yield();
        var model = Model(6, "paper_state");
        using var _ = new NoGradScope<double>();
        var loss = model.Loss(Codes())[0];
        double expected = (double)Data.Value["paper_loss"]!;
        Assert.True(Math.Abs(loss - expected) <= 1e-10 * Math.Max(1, Math.Abs(expected)), $"{expected:R} vs {loss:R}");
    }

    [Theory(Timeout = 120000)]
    [InlineData("paper", 6)]
    [InlineData("release", 3)]
    public async Task EveryParameterDerivative_MatchesPinnedAuthorAutograd(string name, int depthStreams)
    {
        await Task.Yield();
        const string filename = "moshi_lm_gradient_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", filename);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", filename);
        var reference = JObject.Parse(File.ReadAllText(path));
        Assert.Equal("e6a55d2722a65870ef52a6c9f6ecfc0e90f38362", (string)reference["author_commit"]!);
        var item = reference["cases"]![name]!;
        Assert.Empty(item["unused_parameters"]!);
        var layers = new List<LayerBase<double>>();
        var model = Model(depthStreams, $"{name}_state", layers);
        var parameters = layers.SelectMany(layer => layer.GetTrainableParameters()).ToArray();
        Assert.Equal((int)item["parameter_scalars"]!, parameters.Sum(p => p.Length));
        var expectedLayers = new List<LayerBase<double>>();
        var authorGradients = State((string)item["gradient_state"]!);
        Assert.Equal((int)item["parameter_tensors"]!, authorGradients.Count);
        var consumed = new HashSet<string>();
        var expectedModel = Model(depthStreams, $"{name}_state", expectedLayers);
        expectedModel.LoadTorchWeights(key =>
        {
            Assert.True(authorGradients.TryGetValue(key, out var value), $"Missing author derivative {key}");
            consumed.Add(key);
            return value;
        });
        Assert.Equal(authorGradients.Keys.OrderBy(key => key), consumed.OrderBy(key => key));
        var expectedParameters = expectedLayers.SelectMany(layer => layer.GetTrainableParameters()).ToArray();
        Assert.Equal(parameters.Length, expectedParameters.Length);
        using var tape = new GradientTape<double>();
        var loss = model.Loss(Codes());
        double expectedLoss = (double)item["loss"]!;
        Assert.True(Math.Abs(loss[0] - expectedLoss) <= 1e-10 * Math.Max(1, Math.Abs(expectedLoss)));
        var gradients = tape.ComputeGradients(loss, sources: parameters);
        int nonzero = 0;
        for (int p = 0; p < parameters.Length; p++)
        {
            Assert.Equal(parameters[p].Shape, expectedParameters[p].Shape);
            Assert.True(gradients.TryGetValue(parameters[p], out var actual), $"Native parameter {p} has no derivative");
            Assert.NotNull(actual);
            if (actual!.ToArray().Any(value => value != 0)) nonzero++;
            for (int i = 0; i < actual.Length; i++)
            {
                double expected = expectedParameters[p].GetFlat(i);
                double value = actual.GetFlat(i);
                Assert.True(Math.Abs(value - expected) <= 1e-10 * Math.Max(1, Math.Abs(expected)),
                    $"{name} parameter {p}[{i}]: author {expected:R}, native {value:R}");
            }
        }
        Assert.True(nonzero > 0);
    }
}
