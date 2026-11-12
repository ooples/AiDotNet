using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MeloTTS as a model: its BERT features against MeloTTS's get_bert_feature (tools/reference-data/melo_tts_bert_reference.py),
/// synthesis from text (sentences joined by 50 ms of silence), and a training step that updates the generator and both
/// discriminators but never the frozen BERT.
/// </summary>
public class MeloTTSModelTests
{
    private static MeloTTSOptions TinyOptions() => new()
    {
        SampleRate = 16000, FftSize = 64, WindowSize = 64, HopSize = 16, MelChannels = 16, SegmentSize = 64,
        HiddenDim = 16, InterChannels = 8, FilterChannels = 32, NumHeads = 2, NumEncoderLayers = 3, FlowTransformerLayers = 3,
        NumFlowSteps = 1, NumSpeakers = 2, SpeakerChannels = 8, UpsampleRates = new[] { 4, 4 }, UpsampleKernelSizes = new[] { 8, 8 },
        UpsampleInitialChannels = 16, ResblockKernelSizes = new[] { 3 }, ResblockDilationSizes = new[] { new[] { 1, 3, 5 } },
        BertChannels = 16, JaBertChannels = 16, BertLayers = 2, BertHeads = 2, BertIntermediate = 32, DiscriminatorWidthDivisor = 16, LearningRate = 1e-3,
    };

    private static MeloTTS<double> CreateModel() => new(
        new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 8)
        { RandomSeed = 5 },
        TinyOptions());

    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static double[] Snapshot(IEnumerable<ILayer<double>> layers) => layers.SelectMany(l => l.GetParameters().ToArray()).ToArray();

    [Fact(Timeout = 120000)]
    public async Task BertFeatures_MatchMeloTTS()
    {
        await Task.Yield();
        var reference = Load("melo_tts_bert_reference.json");
        var c = reference.GetProperty("config");
        var shape = new BertConfiguration
        {
            VocabSize = 30522, Hidden = c.GetProperty("hidden").GetInt32(), Layers = c.GetProperty("layers").GetInt32(),
            Heads = c.GetProperty("heads").GetInt32(), Intermediate = c.GetProperty("intermediate").GetInt32(),
        };
        var features = new MeloBertFeatures<double>(new CpuEngine(), new List<LayerBase<double>>(), MeloLanguage.English, shape);
        features.Network.LoadTorchWeights((name, dims) =>
        {
            int count = dims.Aggregate(1, (a, b) => a * b);
            return Enumerable.Range(0, count).Select(i => 0.05 * Math.Sin(0.37 * i + 0.11 * name.Length)).ToArray();
        });
        foreach (var s in reference.GetProperty("sentences").EnumerateArray())
        {
            var text = MeloTextPipeline.Prepare(MeloLanguage.English, s.GetProperty("text").GetString()!);
            var actual = features.Features(text.Normalized, text.PhonesPerPiece);
            var expectedShape = s.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            Assert.Equal(new[] { 1, expectedShape[0], expectedShape[1] }, actual.Shape.ToArray());
            var expected = s.GetProperty("features").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-9, $"feature {i}: {expected[i]} vs {actual[i]}");
        }
    }

    [Fact(Timeout = 300000)]
    public async Task MultiPeriodDiscriminator_MatchesMeloTTS()
    {
        await Task.Yield();
        var reference = Load("melo_mpd_reference.json");
        var mpd = new AiDotNet.TextToSpeech.Vocoders.HiFiGanDiscriminators<double>(new CpuEngine(), new[] { 2, 3, 5, 7, 11 }, 1, true, 1,
            spectralFirstScale: false, vitsScale: true);
        var names = new HashSet<string>(reference.GetProperty("names").EnumerateArray().Select(v => v.GetString()!));
        var loaded = new HashSet<string>();
        mpd.LoadTorch(name =>
        {
            if (!names.Contains(name)) return null;
            loaded.Add(name);
            // The fixture's sizes come from MeloTTS's own modules; rebuild each tensor from its name and size.
            int count = SizeOf(name, mpd);
            return Enumerable.Range(0, count).Select(i => 0.02 * Math.Sin(0.37 * i + 0.11 * name.Length)).ToArray();
        });
        Assert.Equal(names.OrderBy(n => n), loaded.OrderBy(n => n));
        Tensor<double> Signal(string key) => new(new[] { 300 }, new Vector<double>(reference.GetProperty(key).EnumerateArray().Select(v => v.GetDouble()).ToArray()));
        var real = mpd.Forward(Signal("real"));
        var fake = mpd.Forward(Signal("fake"));
        // MeloTTS lists the scale discriminator first; this module lists the periods first.
        int[] order = { 5, 0, 1, 2, 3, 4 };
        var realScores = reference.GetProperty("real_scores").EnumerateArray().ToList();
        var fakeScores = reference.GetProperty("fake_scores").EnumerateArray().ToList();
        double dLoss = 0, gLoss = 0, fmLoss = 0;
        for (int k = 0; k < 6; k++)
        {
            var r = real[order[k]];
            var f = fake[order[k]];
            var expectedReal = realScores[k].EnumerateArray().Select(v => v.GetDouble()).ToArray();
            var expectedFake = fakeScores[k].EnumerateArray().Select(v => v.GetDouble()).ToArray();
            Assert.Equal(expectedReal.Length, r.Score.Length);
            // A period discriminator's score is [period, 1, T] here; torch.flatten reads its [1, 1, T, period] time-major.
            int period = k == 0 ? 1 : r.Score.Shape[0], frames = r.Score.Length / period;
            for (int t = 0; t < frames; t++)
                for (int j = 0; j < period; j++)
                {
                    int ours = j * frames + t, theirs = t * period + j;
                    Assert.True(Math.Abs(expectedReal[theirs] - r.Score[ours]) <= 1e-9, $"real score {k}/{theirs}: {expectedReal[theirs]} vs {r.Score[ours]}");
                    Assert.True(Math.Abs(expectedFake[theirs] - f.Score[ours]) <= 1e-9, $"fake score {k}/{theirs}");
                }
            dLoss += r.Score.ToArray().Average(v => (1 - v) * (1 - v)) + f.Score.ToArray().Average(v => v * v);
            gLoss += f.Score.ToArray().Average(v => (1 - v) * (1 - v));
            for (int l = 0; l < r.Features.Count; l++)
                fmLoss += 2 * r.Features[l].ToArray().Zip(f.Features[l].ToArray(), (a, b) => Math.Abs(a - b)).Average();
        }
        Assert.True(Math.Abs(reference.GetProperty("discriminator_loss").GetDouble() - dLoss) <= 1e-9, $"discriminator loss {dLoss}");
        Assert.True(Math.Abs(reference.GetProperty("generator_loss").GetDouble() - gLoss) <= 1e-9, $"generator loss {gLoss}");
        Assert.True(Math.Abs(reference.GetProperty("feature_loss").GetDouble() - fmLoss) <= 1e-9, $"feature loss {fmLoss}");
    }

    // The flat size of a weight-norm parameter of the full-size VITS discriminator, from its layer's shape.
    private static int SizeOf(string name, AiDotNet.TextToSpeech.Vocoders.HiFiGanDiscriminators<double> mpd)
    {
        var parts = name.Split('.');
        int k = int.Parse(parts[1]);
        bool post = parts[2] == "conv_post";
        int i = post ? -1 : int.Parse(parts[3]);
        // Scale discriminator (k = 0): MelGAN's layout; periods (k ≥ 1): HiFi-GAN's.
        (int In, int Out, int Kernel, int Groups)[] scale =
            { (1, 16, 15, 1), (16, 64, 41, 4), (64, 256, 41, 16), (256, 1024, 41, 64), (1024, 1024, 41, 256), (1024, 1024, 5, 1) };
        (int In, int Out, int Kernel, int Groups)[] period = { (1, 32, 5, 1), (32, 128, 5, 1), (128, 512, 5, 1), (512, 1024, 5, 1), (1024, 1024, 5, 1) };
        var layer = post ? (1024, 1, 3, 1) : (k == 0 ? scale[i] : period[i]);
        string leaf = parts[parts.Length - 1];
        return leaf switch
        {
            "bias" or "weight_g" => layer.Item2,
            _ => layer.Item2 * (layer.Item1 / layer.Item4) * layer.Item3,
        };
    }

    [Fact(Timeout = 300000)]
    public async Task Synthesize_JoinsSentencesWithSilence()
    {
        await Task.Yield();
        var model = CreateModel();
        var one = model.Synthesize("Hello world.");
        var two = model.Synthesize("Hello world. Hello world.");
        int silence = (int)(16000 * 0.05);
        Assert.True(one.Length > silence, "a sentence yields speech before its silence");
        Assert.All(one.ToArray(), v => Assert.True(!double.IsNaN(v) && !double.IsInfinity(v)));
        // split_sentence keeps a short text as one piece; the tail is the 50 ms of silence api.py appends.
        Assert.All(one.ToArray().Skip(one.Length - silence), v => Assert.Equal(0.0, v));
        Assert.True(two.Length > one.Length);
    }

    [Fact(Timeout = 600000)]
    public async Task TrainingStep_UpdatesGeneratorAndDiscriminators_NotBert()
    {
        await Task.Yield();
        var model = CreateModel();
        int samples = 16000 / 2;
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++) audio[i] = 0.3 * Math.Sin(2 * Math.PI * 220.0 * i / 16000) + 0.1 * Math.Sin(2 * Math.PI * 530.0 * i / 16000);
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 1 }), Audio = audio, Text = "Hello world.", SpeakerId = 1,
        };
        // Layers initialize on their first forward, so the snapshots follow one.
        double before = model.EvaluateTrainingObjective(sample);
        var generator = Snapshot(model.GeneratorLayers);
        var discriminator = Snapshot(model.DiscriminatorLayers);
        var duration = Snapshot(model.DurationDiscriminatorLayers);
        var bert = Snapshot(model.BertLayers);
        model.Train(sample);
        double after = model.EvaluateTrainingObjective(sample);
        Assert.True(!double.IsNaN(before) && !double.IsNaN(after));
        Assert.NotEqual(generator, Snapshot(model.GeneratorLayers));
        Assert.NotEqual(discriminator, Snapshot(model.DiscriminatorLayers));
        Assert.NotEqual(duration, Snapshot(model.DurationDiscriminatorLayers));
        Assert.Equal(bert, Snapshot(model.BertLayers));
    }

    [Fact(Timeout = 120000)]
    public async Task TensorPath_ReadsSymbolIds()
    {
        await Task.Yield();
        var model = CreateModel();
        var a = model.Predict(new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 10, 20, 30, 40 })));
        var b = model.Predict(new Tensor<double>(new[] { 4 }, new Vector<double>(new double[] { 50, 60, 70, 80 })));
        Assert.True(a.Length > 0 && b.Length > 0);
        Assert.False(a.Length == b.Length && a.ToArray().SequenceEqual(b.ToArray()), "different symbols give different audio");
    }
}
