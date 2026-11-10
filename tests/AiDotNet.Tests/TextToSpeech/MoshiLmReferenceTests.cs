using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
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

    private static MoshiOptions TinyCodecOptions(bool includeTraining = false) => new()
        {
            HiddenDim = 16, NumHeads = 2, Layers = 1, FeedForward = 48, Context = 96,
            DepthDim = 16, DepthHeads = 2, DepthLayers = 1, DepthFeedForward = 48,
            Codebooks = 3, CodebookSize = 12, VocabSize = 20, SampleRate = 1600,
            MimiFilters = 4, MimiDimension = 16, MimiTransformerLayers = 1, MimiCodebooks = 4,
            MimiRatios = new[] { 4, 2 }, MimiHeads = 2, MimiFeedForward = 32, MimiContext = 6, MimiCodebookDim = 8,
            IncludeMimiTraining = includeTraining, WavLmWidth = 16, WavLmLayers = 1, WavLmHeads = 2, WavLmConvChannels = 8,
            MimiDiscriminatorWindows = new[] { 256, 128 }, MimiDiscriminatorFilters = 4,
        };

    [Fact]
    public void StoredCodecQuantizers_ChangeFullCapacityTokensWithoutChangingLmSubset()
    {
        var audio = Enumerable.Range(0, 512).Select(i => .4 * Math.Sin(2 * Math.PI * 90 * i / 1600)).ToArray();
        foreach (int stored in new[] { 4, 5 })
        {
            var options = TinyCodecOptions(); options.MimiCodebooks = stored;
            var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
                AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
            using var model = new Moshi<double>(architecture, options);
            var wave = new Tensor<double>(new[] { 1, 1, audio.Length }, new Vector<double>(audio));
            var random = new Random(3);
            using (var scope = new NoGradScope<double>())
                for (int pass = 0; pass <= stored; pass++)
                    model.Core.Mimi.TrainingForward(wave, stored - 1, true, true, random);
            var full = model.Core.Mimi.Encode(audio);
            var active = model.Core.Mimi.Encode(audio, options.Codebooks);
            Assert.Equal(stored, full.GetLength(1));
            Assert.Equal(3, active.GetLength(1));
            Assert.Equal(full.GetLength(0), active.GetLength(0));
            for (int frame = 0; frame < active.GetLength(0); frame++)
                for (int quantizer = 0; quantizer < active.GetLength(1); quantizer++)
                    Assert.Equal(full[frame, quantizer], active[frame, quantizer]);
        }
    }

    [Theory]
    [InlineData(true, 0)]
    [InlineData(true, 2)]
    [InlineData(false, 0)]
    [InlineData(false, 2)]
    public void SharedCodecDepth_ChangesBothActualMimiTransformers(bool encoder, int depth)
    {
        var options = TinyCodecOptions();
        AiDotNet.TextToSpeech.TtsModelOptions shared = options;
        if (encoder) shared.NumEncoderLayers = depth;
        else shared.NumDecoderLayers = depth;
        using var model = new Moshi<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        var codec = model.Core.Mimi;
        foreach (string field in new[] { "_encoderTransformer", "_decoderTransformer" })
        {
            var transformer = codec.GetType().GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(codec)!;
            var layers = (Array)transformer.GetType().GetProperty("Layers")!.GetValue(transformer)!;
            Assert.Equal(depth, layers.Length);
        }
        Assert.Equal(depth, options.MimiTransformerLayers);
        Assert.Equal((depth, depth), (shared.NumEncoderLayers, shared.NumDecoderLayers));
    }

    [Theory]
    [InlineData("{\"MimiTransformerLayers\":1,\"NumEncoderLayers\":6,\"NumDecoderLayers\":9}", 1)]
    [InlineData("{\"NumEncoderLayers\":6,\"NumDecoderLayers\":9,\"MimiTransformerLayers\":2}", 2)]
    [InlineData("{\"NumEncoderLayers\":2}", 2)]
    [InlineData("{\"NumDecoderLayers\":3}", 3)]
    public void SavedNativeCodecDepth_WinsLegacySharedCounts(string json, int depth)
    {
        var options = Newtonsoft.Json.JsonConvert.DeserializeObject<MoshiOptions>(json)!;
        Assert.Equal(depth, MoshiCore<double>.MimiConfiguration(options).TransformerLayers);
        Assert.Equal((depth, depth), (options.NumEncoderLayers, options.NumDecoderLayers));
        var restored = Newtonsoft.Json.JsonConvert.DeserializeObject<MoshiOptions>(Newtonsoft.Json.JsonConvert.SerializeObject(options))!;
        Assert.Equal((depth, depth, depth), (restored.MimiTransformerLayers, restored.NumEncoderLayers, restored.NumDecoderLayers));
    }

    [Fact(Timeout = 120000)]
    public async Task MimiTraining_UpdatesBothNetworksWithoutUnrequestedClipping()
    {
        await Task.Yield();
        var options = TinyCodecOptions(includeTraining: true);
        using var model = new Moshi<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        Tensor<double>[] Group(string field) => ((IEnumerable<LayerBase<double>>)typeof(Moshi<double>)
            .GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!)
            .SelectMany(layer => layer.GetTrainableParameters()).ToArray();
        var generator = Group("_mimiLayers"); var critic = Group("_discriminatorLayers");
        var generatorBefore = generator.SelectMany(p => p.ToArray()).ToArray();
        var criticBefore = critic.SelectMany(p => p.ToArray()).ToArray();
        var audio = new Tensor<double>(new[] { 1600 }, new Vector<double>(Enumerable.Range(0, 1600)
            .Select(i => .4 * Math.Sin(2 * Math.PI * 90 * i / 1600) + .2 * Math.Sin(2 * Math.PI * 330 * i / 1600)).ToArray()));
        for (int step = 0; step < 2; step++)
        {
            double loss = model.TrainMimi(audio);
            Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), "Mimi training produced a nonfinite loss.");
        }
        Assert.False(generatorBefore.SequenceEqual(generator.SelectMany(p => p.ToArray())), "Mimi generator did not update.");
        Assert.False(criticBefore.SequenceEqual(critic.SelectMany(p => p.ToArray())), "Mimi discriminator did not update.");
        Assert.Equal(0, model.MaxGradNormValue);
        var optimizers = (Dictionary<string, AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)
            typeof(Moshi<double>).GetField("_optimizers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        foreach (string component in new[] { "mimi", "mimi discriminator" })
        {
            var optimizer = Assert.IsType<AiDotNet.Optimizers.AdamWOptimizer<double, Tensor<double>, Tensor<double>>>(optimizers[component]);
            var actual = Assert.IsType<AiDotNet.Models.Options.AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>>(optimizer.GetOptions());
            Assert.Equal((.5, .9, 1e-8), (actual.Beta1, actual.Beta2, actual.Epsilon));
            Assert.False(actual.UseAdaptiveBetas, $"{component} uses adaptive betas.");
            Assert.False(actual.EnableGradientClipping, $"{component} applies unrequested optimizer clipping.");
            Assert.Equal(0, actual.MaxGradientNorm);
        }
    }

    [Theory]
    [InlineData("MelChannels", -1)]
    [InlineData("MelChannels", 1)]
    [InlineData("MelChannels", 80)]
    [InlineData("HopSize", -1)]
    [InlineData("HopSize", 1)]
    [InlineData("HopSize", 256)]
    [InlineData("FftSize", -1)]
    [InlineData("FftSize", 1)]
    [InlineData("FftSize", 1024)]
    [InlineData("MaxMelLength", -1)]
    [InlineData("MaxMelLength", 1)]
    [InlineData("MaxMelLength", 2048)]
    public void NativeCodecTasks_RejectNonzeroMelFrontendSettings(string option, int value)
    {
        foreach (bool recognizer in new[] { false, true })
        {
            var options = TinyCodecOptions();
            options.MelChannels = 0; options.HopSize = 0; options.FftSize = 0; options.MaxMelLength = 0;
            typeof(MoshiOptions).GetProperty(option)!.SetValue(options, value);
            var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
                AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
            Assert.Throws<ArgumentException>(() =>
            {
                using IDisposable model = recognizer ? new AiDotNet.SpeechRecognition.Streaming.Moshi<double>(architecture, options)
                    : new Moshi<double>(architecture, options);
            });
        }
    }

    [Theory]
    [InlineData(.1)]
    [InlineData(-.1)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void NativeCodecTasks_RejectTransformerDropout(double rate)
    {
        foreach (bool recognizer in new[] { false, true })
        {
            var options = TinyCodecOptions(); options.DropoutRate = rate;
            var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
                AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
            Assert.Throws<ArgumentException>(() =>
            {
                using IDisposable model = recognizer ? new AiDotNet.SpeechRecognition.Streaming.Moshi<double>(architecture, options)
                    : new Moshi<double>(architecture, options);
            });
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void NativeTextLimit_RequiresAPositiveTokenCount(int maximum)
    {
        var options = TinyCodecOptions(); options.MaxTextLength = maximum;
        Assert.Throws<ArgumentException>(() =>
        {
            using var model = new Moshi<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
                AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        });
    }

    [Fact]
    public void TimedTranscript_EnforcesActualInputTokenLimit()
    {
        var options = TinyCodecOptions(); options.MaxTextLength = 1;
        using var model = new Moshi<double>(new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        Assert.Throws<ArgumentException>(() => model.Core.TextStream(new[] { new MoshiTokenSpan(new[] { 5, 7 }, 0, .1) }, 32, 0));
    }

    [Theory(Timeout = 120000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ActualCodedTraining_MatchesTorchMomentsWeightsAndCosineRates(bool recognizer)
    {
        await Task.Yield();
        const string filename = "moshi_lm_gradient_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", filename);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", filename);
        var reference = JObject.Parse(File.ReadAllText(path))["optimizer"]!;
        var settings = reference["settings"]!;
        var options = new MoshiOptions
        {
            HiddenDim = 16, NumHeads = 2, Layers = 2, FeedForward = 66, Context = 5,
            DepthDim = 8, DepthHeads = 2, DepthLayers = 2, DepthFeedForward = 33,
            Codebooks = 3, CodebookSize = 12, VocabSize = 20, TextPadding = 3, TextEndPadding = 0,
            SampleRate = 1600, MimiFilters = 4, MimiDimension = 16, MimiTransformerLayers = 1,
            MimiCodebooks = 4, MimiRatios = new[] { 4, 2 }, MimiHeads = 2, MimiFeedForward = 32,
            MimiContext = 6, MimiCodebookDim = 8, WarmupSteps = 0,
            LearningRate = (double)settings["temporal_rate"]!, DepthLearningRate = (double)settings["depth_rate"]!,
            TextRateMultiplier = (double)settings["text_rate_multiplier"]!, WeightDecay = (double)settings["weight_decay"]!,
            MaxGradientNorm = (double)settings["max_norm"]!, TotalSteps = (int)settings["cosine_steps"]!,
            Variant = recognizer ? MoshiVariant.Asr : MoshiVariant.Dialogue, TextAudioDelaySeconds = 0,
        };
        var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);
        object model = recognizer ? new AiDotNet.SpeechRecognition.Streaming.Moshi<double>(architecture, options)
            : new Moshi<double>(architecture, options);
        using var lifetime = (IDisposable)model;
        var modelType = model.GetType();
        var core = model is Moshi<double> speaker ? speaker.Core : ((AiDotNet.SpeechRecognition.Streaming.Moshi<double>)model).Core;
        var initial = State((string)Data.Value["paper_state"]!);
        core.LanguageModel.LoadTorchWeights(key => initial.TryGetValue(key, out var value) ? value : null);
        var layers = (List<LayerBase<double>>)modelType.GetField("_lmLayers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
        var parameters = layers.SelectMany(layer => layer.GetTrainableParameters()).ToArray();
        Assert.Equal(19728, parameters.Sum(p => p.Length));
        var expectedLayers = new List<LayerBase<double>>();
        var expectedModel = Model(6, "paper_state", expectedLayers);
        var expectedParameters = expectedLayers.SelectMany(layer => layer.GetTrainableParameters()).ToArray();
        var train = modelType.GetMethod(recognizer ? "TrainWithCustomObjective" : "TrainCodes", BindingFlags.Instance | BindingFlags.NonPublic)!;
        foreach (var step in reference["steps"]!)
        {
            // This is the real LM training path on the author's valid codec-token sequence.
            // The waveform codec is frozen and outside this component comparison.
            object[] arguments;
            if (recognizer)
            {
                var marker = new Tensor<double>(new[] { 1 });
                var actualOptimizer = modelType.GetMethod("Optimizer", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(model, null)!;
                arguments = new object[] { marker, marker,
                    (Func<Tensor<double>, Tensor<double>, Tensor<double>>)((_, _) => core.Loss(Codes())), actualOptimizer };
            }
            else arguments = new object[] { Codes(), false };
            double loss = (double)train.Invoke(model, arguments)!;
            Assert.True(Math.Abs(loss - (double)step["loss"]!) <= 1e-10);
            object optimizerInstance;
            if (recognizer) optimizerInstance = modelType.GetField("_optimizer", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
            else
            {
                var optimizers = (Dictionary<string, AiDotNet.Interfaces.IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>)
                    modelType.GetField("_optimizers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;
                optimizerInstance = optimizers["audio:PreTraining"];
            }
            var optimizer = Assert.IsType<AiDotNet.Optimizers.AdamWOptimizer<double, Tensor<double>, Tensor<double>>>(optimizerInstance);
            var optimizerType = optimizer.GetType();
            foreach (var component in new[] { ("first_moment", "_tapeM"), ("second_moment", "_tapeV"), ("weights", "") })
            {
                var values = State((string)step[component.Item1]!);
                expectedModel.LoadTorchWeights(key => values.TryGetValue(key, out var value) ? value : null);
                var moments = component.Item2.Length == 0 ? null :
                    (IReadOnlyDictionary<Tensor<double>, Tensor<double>>)optimizerType.GetField(component.Item2, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(optimizer)!;
                for (int p = 0; p < parameters.Length; p++)
                {
                    var actual = moments is null ? parameters[p] : moments[parameters[p]];
                    for (int i = 0; i < actual.Length; i++)
                    {
                        double expected = expectedParameters[p].GetFlat(i), value = actual.GetFlat(i);
                        Assert.True(Math.Abs(value - expected) <= 1e-10 * Math.Max(1, Math.Abs(expected)),
                            $"{component.Item1} parameter {p}[{i}]: author {expected:R}, native {value:R}");
                    }
                }
            }
            Assert.True(Math.Abs(optimizer.GetCurrentLearningRate() - (double)step["next_rates"]![0]!) <= 1e-14);
        }
    }
}
