using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Mega-TTS's acoustic model and prosody language model against their composition from the authors' NATSpeech and
/// SyntaSpeech modules (tools/reference-data/megatts_model_reference.py): the mel features, the prosody encoder and its VQ
/// codes and loss, the timbre vector, the decoded mel spectrogram, the duration predictor and NATSpeech's duration loss,
/// the L2 and adversarial terms, and the P-LLM's logits and cross-entropy.
/// </summary>
public class MegaTtsModelReferenceTests
{
    /// <summary>The script's sizes: width 16, two layers or blocks everywhere, 10 mel bins, 12 codes of width 8.</summary>
    internal static MegaTTSOptions TinyOptions() => new()
    {
        HiddenDim = 16, ContentLayers = 2, ContentHeads = 2, ConvBlocks = 2, TimbreKernel = 7, ProsodyBins = 6, Codes = 12,
        CodeDimension = 8, MelChannels = 10, LmLayers = 2, LmHidden = 16, LmHeads = 2, DiscriminatorWindows = new[] { 8, 16 },
        DiscriminatorHidden = 6, TopK = 3, VocoderInitialChannels = 32, VocoderResBlockKernels = new[] { 3 },
        VocoderResBlockDilations = new[] { new[] { 1, 3, 5 } },
    };

    [Fact(Timeout = 300000)]
    public async Task Model_MatchesNatSpeechComposition()
    {
        await Task.Yield();
        const string fileName = "megatts_model_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        double[] Read(string name)
        {
            used.Add(name);
            return Doubles(state[name] ?? throw new InvalidDataException(name));
        }
        var model = new MegaTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), TinyOptions());
        model.Network.LoadTorch(Read);
        model.SetTrainingMode(false);
        // The decoder layers' unused encoder attention (the P-LLM is decoder-only).
        Assert.Empty(state.Properties().Select(p => p.Name).Where(n => !used.Contains(n) && !n.Contains("encoder_attn") && !n.Contains(".op.layer_norm2.")));

        void Close(JToken expected, Tensor<double> actual, string what, double tolerance = 1e-9)
        {
            var e = expected.Type == JTokenType.Array ? Doubles(expected) : new[] { (double)expected };
            Assert.Equal(e.Length, actual.Length);
            for (int i = 0; i < e.Length; i++)
                Assert.True(Math.Abs(e[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual[i]}");
        }
        using var _ = new NoGradScope<double>();
        Close(f["features"]!, model.MelOf(Doubles(f["wave"]!)), "features");

        var phones = NatSpeechFrontEnd.Encode(f["phones"]!.Select(v => (string)v!), model.Vocabulary);
        var durations = f["durations"]!.Select(v => (int)v).ToArray();
        int frames = durations.Sum();
        var mel = new Tensor<double>(new[] { frames, 10 }, new Vector<double>(Doubles(f["mel"]!)));
        var reference = new Tensor<double>(new[] { 9, 10 }, new Vector<double>(Doubles(f["reference"]!)));
        var u = new MegaTTS<double>.Utterance(phones, durations, MegaTts3Aligner<double>.LengthRegulate(durations), mel, reference);
        var network = model.Network;
        var content = model.Content(phones);
        Close(f["content"]!, content, "content");
        var hidden = network.ProsodyHidden(mel, u.Mel2Ph, phones.Length);
        Close(f["hidden"]!, hidden, "hidden");
        var codes = network.Quantizer.Encode(hidden);
        Assert.Equal(f["codes"]!.Select(v => (int)v), Enumerable.Range(0, phones.Length).Select(j => codes[0, j]));
        Close(f["timbre"]!, network.TimbreVector(reference), "timbre");

        var r = model.Reconstruct(u);
        Close(f["decoded"]!, r.Mel, "decoded");
        Close(f["dur_pred"]!, r.Durations, "dur_pred");
        Close(f["vq"]!, r.VqLoss, "vq");
        double durationLoss = (double)f["pdur"]! + (double)f["wdur"]! + (double)f["sdur"]!;
        Close(new JArray(durationLoss), model.DurationLoss(r.Durations, phones, durations), "duration loss");
        Close(f["total"]!, model.AcousticLoss(u, null).Loss, "total");
        double adversarial = (double)f["total"]! + 0.05 * (double)f["adversarial"]!;
        Close(new JArray(adversarial), model.AcousticLoss(u, f["starts"]!.Select(v => (int)v).ToArray()).Loss, "with adversarial");

        var previous = f["previous"]!.Select(v => (int)v).ToArray();
        var targets = f["targets"]!.Select(v => (int)v).ToArray();
        var timbre = network.TimbreVector(reference);
        Close(f["logits"]!, network.ProsodyLogits(previous, content, timbre), "logits");
        Close(f["ce"]!, model.ProsodyLoss(previous, content, timbre, targets), "ce");
    }

    /// <summary>Synthesis with a full prompt (its phonemes and durations) runs the paper's decoding: the prompt's codes
    /// lead the P-LLM, the same seed gives the same audio, and the audio is the vocoder's frames of the predicted
    /// durations.</summary>
    [Fact(Timeout = 300000)]
    public async Task Synthesis_WithPrompt_IsDeterministicAndFramed()
    {
        await Task.Yield();
        var options = TinyOptions();
        var model = new MegaTTS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 1), options);
        var promptTokens = model.TextToTokens("Hello world.");
        int count = promptTokens.Length;
        var durations = Enumerable.Range(0, count).Select(i => 1 + i % 3).ToArray();
        int samples = (durations.Sum() - 1) * options.HopSize;
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++) audio[i] = 0.3 * Math.Sin(2 * Math.PI * 200 * i / options.SampleRate);
        model.Voice = new AiDotNet.TextToSpeech.TtsVoice<double> { ReferenceAudio = audio, ReferenceTokens = promptTokens, ReferenceDurations = durations };
        var first = model.Synthesize("Good morning.");
        var second = model.Synthesize("Good morning.");
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.Equal(0, first.Length % options.HopSize);
        Assert.All(first.ToArray(), v => Assert.True(Math.Abs(v) <= 1));
        var phones = model.TextToTokens("Good morning.").ToArray().Select(v => (int)v).ToArray();
        using var _ = new NoGradScope<double>();
        var mel = model.SynthesizeMel(phones, AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(options.SamplingSeed));
        Assert.Equal(first.Length, mel.Shape[0] * options.HopSize);
        Assert.Equal(mel.ToArray(), model.Predict(model.TextToTokens("Good morning.")).ToArray());
    }
}
