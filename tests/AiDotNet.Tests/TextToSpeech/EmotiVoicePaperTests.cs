using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Weights;
using AiDotNet.Audio.Pitch;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.StyleEmotion;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// EmotiVoice has no paper: its released code (netease-youdao/EmotiVoice) is the reference. A tiny JETSGenerator built by
/// that code (tools/reference-data/emotivoice_reference.py) fixes the teacher-forced training pass, inference, the data
/// features and SimBERT's pooled output.
/// </summary>
public class EmotiVoicePaperTests
{
    private static JObject Reference()
    {
        const string fileName = "emotivoice_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static double[] Values(JToken token) => token.Select(v => (double)v).ToArray();

    private static int[] Ints(JToken token) => token.Select(v => (int)v).ToArray();

    private static void Close(double[] expected, double[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-8 * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: reference {expected[i]:R}, port {actual[i]:R}.");
    }

    private static void Close(double[] expected, Tensor<double> actual, string what) => Close(expected, actual.ToArray(), what);

    private static Tensor<double> Row(double[] values) => new(new[] { 1, values.Length }, new Vector<double>(values));

    private static (EmotiVoice<double> Model, JObject Fixture) Model()
    {
        var fixture = Reference();
        var model = new EmotiVoice<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), new EmotiVoiceOptions
        {
            SampleRate = 16000, HopSize = 16, FftSize = 64, WindowLength = 64, MelChannels = 10, HiddenDim = 16, VocabSize = 20,
            SpeakerCount = 3, Heads = 2, EncoderLayers = 2, DecoderLayers = 2, VarianceChannels = 16, UpsampleRates = new[] { 4, 4 },
            UpsampleKernels = new[] { 8, 8 }, UpsampleInitialChannels = 16, SegmentFrames = 4,
            StyleBert = new BertConfiguration { VocabSize = 30, Hidden = 8, Layers = 2, Heads = 2, Intermediate = 16, MaxPositions = 32 },
        });
        model.SetTrainingMode(false);
        var (read, unused) = Weights(fixture, "safetensors_base64");
        model.LoadReferenceWeights(read, name => Tensors(fixture, "safetensors_base64").ContainsKey(name) ? read(name, Array.Empty<int>()) : null);
        Assert.Empty(unused());
        string bert = Path.Combine(Path.GetTempPath(), $"emotivoice_bert_{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(bert, Convert.FromBase64String((string)fixture["bert_safetensors_base64"]!));
        model.LoadStyleEncoderWeights(bert);
        File.Delete(bert);
        return (model, fixture);
    }

    private static readonly Dictionary<string, Dictionary<string, Tensor<float>>> Cache = new();

    private static Dictionary<string, Tensor<float>> Tensors(JObject fixture, string key)
    {
        string blob = (string)fixture[key]!;
        lock (Cache)
        {
            if (Cache.TryGetValue(blob, out var cached)) return cached;
            string path = Path.Combine(Path.GetTempPath(), $"emotivoice_{Guid.NewGuid():N}.safetensors");
            File.WriteAllBytes(path, Convert.FromBase64String(blob));
            var weights = new WeightLoader().LoadWeights(path);
            File.Delete(path);
            return Cache[blob] = weights;
        }
    }

    /// <summary>A reader over the fixture's weights that records which tensors were read (shape checked when given).</summary>
    private static (TorchReader Read, Func<IEnumerable<string>> Unused) Weights(JObject fixture, string key)
    {
        var weights = Tensors(fixture, key);
        var used = new HashSet<string>();
        TorchReader read = (name, shape) =>
        {
            Assert.True(weights.ContainsKey(name), $"The reference has no tensor '{name}'.");
            if (shape.Length > 0) Assert.Equal(shape.Aggregate(1, (a, b) => a * b), weights[name].Length);
            used.Add(name);
            return weights[name].ToVector().Select(v => (double)v).ToArray();
        };
        return (read, () => weights.Keys.Where(k => !used.Contains(k)).ToList());
    }

    [Fact(Timeout = 300000)]
    public async Task TrainingPass_MatchesJetsGenerator()
    {
        await Task.Yield();
        var (model, f) = Model();
        int frames = (int)f["frames"]!, mels = (int)f["config"]!["n_mels"]!, hop = (int)f["hop"]!, segment = (int)f["config"]!["segment_size"]!;
        var mel = Values(f["mel"]!);
        var target = new Tensor<double>(new[] { frames, mels });
        for (int t = 0; t < frames; t++)
            for (int m = 0; m < mels; m++) target[t, m] = mel[m * frames + t];
        int start = (int)f["segment_start"]!;
        var real = Values(f["real_wave"]!).Skip(start * hop).Take(segment * hop).ToArray();
        var step = new EmotiVoice<double>.Step
        {
            Phonemes = Ints(f["phonemes"]!), Speaker = (int)f["speaker"]!, Style = Row(Values(f["style"]!)), Content = Row(Values(f["content"]!)),
            Mel = target, Pitch = Values(f["pitch_frames"]!), Energy = Values(f["energy_frames"]!), Start = start,
            Wave = new Tensor<double>(new[] { real.Length }, new Vector<double>(real)),
        };
        var pass = model.Generate(step, training: false, withLosses: true, adversarialTerms: false);
        Close(Values(f["conditioned"]!), pass.Conditioned, "conditioned");
        Close(Values(f["log_p_attn"]!), pass.LogP, "log_p_attn");
        Assert.Equal(Ints(f["durations"]!), pass.Durations);
        Close(Values(f["bin_loss"]!), pass.Binarization!, "binarization loss");
        Close(Values(f["forward_sum"]!), pass.ForwardSum!, "forward-sum loss");
        Close(Values(f["pitch_targets"]!), pass.PitchTargets, "pitch targets");
        Close(Values(f["energy_targets"]!), pass.EnergyTargets, "energy targets");
        Close(Values(f["pitch_predictions"]!), pass.PitchOut, "pitch predictions");
        Close(Values(f["energy_predictions"]!), pass.EnergyOut, "energy predictions");
        Close(Values(f["log_duration_predictions"]!), pass.DurationOut, "log-duration predictions");
        var losses = Values(f["prosody_losses"]!);
        Close(new[] { losses[0] }, pass.Duration!, "duration loss");
        Close(new[] { losses[1] }, pass.Pitch!, "pitch loss");
        Close(new[] { losses[2] }, pass.Energy!, "energy loss");
        Close(Values(f["latent"]!), pass.Latent, "latent");
        Close(Values(f["wave"]!), pass.Wave, "segment waveform");
        Close(Values(f["training_mel"]!), model.TrainingMel(pass.Wave), "training mel");
        Close(Values(f["mel_loss"]!), pass.Mel!, "mel loss");
    }

    [Fact(Timeout = 300000)]
    public async Task Inference_MatchesJetsGenerator()
    {
        await Task.Yield();
        var (model, f) = Model();
        var wave = model.SynthesizeWave(Ints(f["phonemes"]!), (int)f["speaker"]!, Row(Values(f["style"]!)), Row(Values(f["content"]!)), new Random(0));
        Close(Values(f["inference_wave"]!), wave, "inference waveform");
    }

    [Fact(Timeout = 300000)]
    public async Task StyleEncoder_MatchesBertPooler()
    {
        await Task.Yield();
        var (model, f) = Model();
        Close(Values(f["pooled"]!), model.StyleEmbedding(Ints(f["style_ids"]!)), "pooled");
    }

    [Fact(Timeout = 300000)]
    public async Task StyleTokens_MatchSimBertTokenizer()
    {
        await Task.Yield();
        var f = Reference();
        var prompts = f["prompts"]!.Select(v => (string)v!).ToArray();
        var expected = f["prompt_tokens"]!.Select(Ints).ToArray();
        var model = new EmotiVoice<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), new EmotiVoiceOptions { HiddenDim = 16, Heads = 2, EncoderLayers = 1, DecoderLayers = 1, VarianceChannels = 16,
            UpsampleRates = new[] { 16, 16 }, UpsampleKernels = new[] { 32, 32 }, UpsampleInitialChannels = 16, DiscriminatorWidthDivisor = 16,
            StyleBertHidden = 8, StyleBertLayers = 1, StyleBertHeads = 2, StyleBertIntermediate = 16 });
        for (int i = 0; i < prompts.Length; i++)
            Assert.Equal(expected[i], model.StyleTokens(prompts[i]));
    }

    [Fact(Timeout = 300000)]
    public async Task DataFeatures_MatchTheLoader()
    {
        await Task.Yield();
        var (model, f) = Model();
        int sr = (int)f["sr"]!, fft = (int)f["fft"]!, hop = (int)f["hop"]!, mels = (int)f["config"]!["n_mels"]!;
        var voice = Values(f["voice"]!);
        var mel = new TacotronSpectrogram(sr, fft, hop, fft, mels, 0, 8000).LogMel(voice);
        var flat = new double[mel.Length];
        for (int t = 0; t < mel.GetLength(0); t++)
            for (int m = 0; m < mels; m++) flat[t * mels + m] = mel[t, m];
        Close(Values(f["target_mel"]!), flat, "target mel");
        var f0 = new WorldPitchDetector<double>(sr).EstimateF0(voice, 1000.0 * hop / sr).F0;
        Close(Values(f["pitch"]!), EmotiVoice<double>.ContinuousPitch(f0), "pitch");
        Close(Values(f["energy"]!), model.FrameEnergy(voice), "energy");
        Close(Values(f["trimmed"]!), model.Trim(voice), "trimmed");
    }
}
