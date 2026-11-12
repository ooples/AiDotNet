using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SpeechT5 against Hugging Face transformers 4.57.1 (tools/reference-data/speecht5_hf_reference.py): a tiny
/// SpeechT5ForTextToSpeech loaded from its Hugging Face names (teacher-forced outputs, generate_speech with the summed stop
/// rule) and a tiny SpeechT5HifiGan with its mean and scale; with <c>SPEECHT5_HF_DIR</c> set (holding
/// <c>microsoft/speecht5_tts</c> in tts/ and <c>microsoft/speecht5_hifigan</c> in hifigan/), the released model's tokens,
/// outputs, generation and audio.
/// </summary>
public class SpeechT5HuggingFaceReferenceTests
{
    private static JObject Fixture(string fileName)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static double[] Doubles(JToken t) => t.Select(v => (double)v).ToArray();
    private static int[] Ints(JToken t) => t.Select(v => (int)v).ToArray();
    private static Tensor<double> Tensor(JToken t, params int[] shape) => new(shape, new Vector<double>(Doubles(t)));

    private static void Close(JToken expected, Tensor<double> actual, string what, double tolerance)
    {
        var e = Doubles(expected);
        Assert.Equal(e.Length, actual.Length);
        for (int i = 0; i < e.Length; i++)
            Assert.True(Math.Abs(e[i] - actual.GetFlat(i)) <= tolerance * Math.Max(1, Math.Abs(e[i])), $"{what}[{i}]: {e[i]} vs {actual.GetFlat(i)}");
    }

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    internal static SpeechT5Options TinyOptions() => new()
    {
        HiddenDim = 16, FeedForwardDim = 32, NumHeads = 2, NumEncoderLayers = 2, NumDecoderLayers = 2, MaxRelativePosition = 4, VocabSize = 15,
        MaxTextLength = 40, MaxMelLength = 200, MelChannels = 10, ReductionFactor = 2, SpeakerDim = 6, PrenetUnits = 8, PrenetLayers = 2,
        PrenetDropout = 0, PostnetChannels = 8, PostnetLayers = 5, PostnetKernel = 5, PostnetDropout = 0, DropoutRate = 0, AttentionDropout = 0,
        ActivationDropout = 0, PositionalDropout = 0, FeatureLayers = new[] { (8, 10, 5), (8, 3, 2), (8, 3, 2) }, ConvPositionKernel = 8,
        ConvPositionGroups = 2, FinalDim = 6, HubertClasses = new[] { 16 }, LatentVars = 5, MaskId = 13, SpeakerEncoderChannels = new[] { 8, 8, 8, 8, 12 },
        HopSize = 4, VocoderInitialChannels = 16, VocoderUpsampleRates = new[] { 2, 2 }, VocoderUpsampleKernels = new[] { 4, 4 },
        VocoderResBlockKernels = new[] { 3, 5 }, VocoderResBlockDilations = new[] { new[] { 1, 3 }, new[] { 1, 3 } },
        StopCriterion = SpeechT5StopCriterion.SummedProbability, MaxLengthRatio = 3.0,
    };

    [Fact(Timeout = 300000)]
    public async Task TinyModel_MatchesTransformers()
    {
        await Task.Yield();
        var f = Fixture("speecht5_hf_reference.json");
        var state = (JObject)f["state"]!;
        var used = new HashSet<string>();
        var model = new SpeechT5<double>(Architecture(), TinyOptions());
        model.LoadState((name, _) =>
        {
            string key = SpeechT5Core<double>.HuggingFaceName(name) ?? throw new InvalidDataException(name);
            used.Add(key);
            return Doubles(state[key] ?? throw new InvalidDataException(key));
        }, pretraining: false);
        Assert.All(state.Properties().Select(p => p.Name), n => Assert.Contains(n, used));
        var vocoder = (JObject)f["vocoder_state"]!;
        model.LoadVocoder(name => vocoder[name] is { } t ? Doubles(t) : null);
        model.SetTrainingMode(false);

        var core = model.Core;
        var tokens = Ints(f["tokens"]!);
        var speaker = Tensor(f["speaker"]!, 6);
        var mel = Tensor(f["mel"]!, 12, 10);
        using (new NoGradScope<double>())
        {
            var outputs = core.DecodeSpeech(core.EncodeText(tokens), core.DecoderInputs(mel), speaker);
            Close(f["before"]!, outputs.Before, "before", 1e-9);
            Close(f["after"]!, outputs.After, "after", 1e-9);
            Close(f["logits"]!, outputs.Logits, "logits", 1e-9);
        }
        var generated = model.GenerateSpectrogram(tokens, new TtsVoice<double> { SpeakerEmbedding = speaker });
        Assert.Equal((int)f["generated_frames"]!, generated.Shape[0]);
        Close(f["generated"]!, generated, "generated", 1e-9);
        Close(f["wave"]!, model.Vocode(mel), "wave", 1e-9);
    }

    [Fact]
    public void TokenizeText_MatchesTransformers()
    {
        // The release's spm_char.model is embedded, so the tokenizer needs no download.
        var f = Fixture("speecht5_hf_released_reference.json");
        var model = new SpeechT5<double>(Architecture(), TinyOptions());
        var texts = f["texts"]!.Select(t => (string)t!).ToArray();
        var expectedIds = f["token_ids"]!.Select(Ints).ToArray();
        for (int i = 0; i < texts.Length; i++) Assert.Equal(expectedIds[i], model.TokenizeText(texts[i]));
    }

    [Fact(Timeout = 1800000)]
    public async Task ReleasedModel_MatchesTransformers()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("SPEECHT5_HF_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "tts", "pytorch_model.bin"))) return;
        var f = Fixture("speecht5_hf_released_reference.json");
        var model = new SpeechT5<double>(Architecture(), new SpeechT5Options
        {
            PrenetDropout = 0, StopCriterion = SpeechT5StopCriterion.SummedProbability, BertInitialization = false,
        });
        model.LoadReferenceWeights(new SpeechT5Checkpoints
        {
            Model = Path.Combine(dir, "tts", "pytorch_model.bin"), Vocoder = Path.Combine(dir, "hifigan", "pytorch_model.bin"),
        });
        model.SetTrainingMode(false);
        var expectedIds = f["token_ids"]!.Select(Ints).ToArray();

        var core = model.Core;
        var speaker = Tensor(f["speaker"]!, 512);
        var mel = Tensor(f["mel"]!, 20, 80);
        using (new NoGradScope<double>())
        {
            var outputs = core.DecodeSpeech(core.EncodeText(expectedIds[0]), core.DecoderInputs(mel), speaker);
            Close(f["after"]!, outputs.After, "after", 1e-8);
            Close(f["logits"]!, outputs.Logits, "logits", 1e-8);
        }
        var generated = model.GenerateSpectrogram(expectedIds[0], new TtsVoice<double> { SpeakerEmbedding = speaker });
        Assert.Equal((int)f["generated_frames"]!, generated.Shape[0]);
        Close(f["generated"]!, generated, "generated", 1e-8);
        Close(f["wave"]!, model.Vocode(generated), "wave", 1e-8);
    }
}
