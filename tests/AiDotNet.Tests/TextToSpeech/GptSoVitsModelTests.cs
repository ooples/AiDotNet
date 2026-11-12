using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// GPT-SoVITS end to end at test scale for every version: a voice from a recording and its English transcript, synthesis,
/// and one step of each training stage the version has (s1, the GAN, flow matching with LoRA). The release's real
/// weights load and speak when <c>GPTSOVITS_DIR</c> names its <c>pretrained_models</c> directory.
/// </summary>
public class GptSoVitsModelTests
{
    private static GPTSoVITSOptions Tiny(GptSoVitsVersion version, GptSoVitsStage stage) => new()
    {
        Version = version, CurrentStage = stage,
        HiddenDim = 16, NumHeads = 2, T2SLayers = 1, SemanticCodebookSize = 32,
        InterChannels = 8, SynthesizerHidden = 16, SynthesizerFilter = 32, SynthesizerHeads = 2, SynthesizerLayers = 2,
        StyleChannels = 16, MrteChannels = 16, StyleHidden = 16, UpsampleInitialChannels = 32,
        ResblockKernels = new[] { 3 }, ResblockDilations = new[] { new[] { 1, 3, 5 } },
        HubertHidden = 16, HubertLayers = 1, HubertHeads = 2, HubertIntermediate = 32, HubertConvDims = new[] { 8, 8, 8, 8, 8, 8, 8 },
        HubertPositionKernel = 8, HubertPositionGroups = 2, BertHidden = 16, BertLayers = 3, BertHeads = 2, BertIntermediate = 32,
        SpeakerChannels = 8, SpeakerBlocks = new[] { 1, 1, 1, 1 },
        DitDim = 16, DitDepth = 1, DitHeads = 2, DitHeadDim = 8, DitConvLayers = 1, BridgeChannels = 16, SampleSteps = 2,
        VocoderInitialChannels = version == GptSoVitsVersion.V3 ? 64 : 32, VocoderResblockKernels = new[] { 3 },
        VocoderResblockDilations = new[] { new[] { 1 } }, LoraRank = 2,
        TextLanguage = GptSoVitsTextLanguage.English, PromptLanguage = GptSoVitsTextLanguage.English, MaxSeconds = 1, T2SAccumulation = 1,
    };

    private static GPTSoVITS<double> Model(GPTSoVITSOptions options) =>
        new(new NeuralNetworkArchitecture<double>(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);

    private static Tensor<double> Tone(int samples, int rate, double frequency, int seed)
    {
        var random = new Random(seed);
        var x = new double[samples];
        for (int i = 0; i < samples; i++) x[i] = 0.3 * Math.Sin(2 * Math.PI * frequency * i / rate) + 0.02 * (random.NextDouble() - 0.5);
        return new Tensor<double>(new[] { samples }, new Vector<double>(x));
    }

    private static void AssertSpeech(Tensor<double> audio)
    {
        Assert.True(audio.Length > 0);
        for (int i = 0; i < audio.Length; i++) Assert.False(double.IsNaN(audio.GetFlat(i)) || double.IsInfinity(audio.GetFlat(i)), $"sample {i}");
    }

    [Theory(Timeout = 900000)]
    [InlineData(GptSoVitsVersion.V1)]
    [InlineData(GptSoVitsVersion.V2)]
    [InlineData(GptSoVitsVersion.V2Pro)]
    [InlineData(GptSoVitsVersion.V2ProPlus)]
    [InlineData(GptSoVitsVersion.V3)]
    [InlineData(GptSoVitsVersion.V4)]
    [InlineData(GptSoVitsVersion.V5Dev)]
    [InlineData(GptSoVitsVersion.V5Turbo)]
    public async Task EveryVersion_ClonesAVoiceAndTrainsEachStage(GptSoVitsVersion version)
    {
        await Task.Yield();
        bool flow = version >= GptSoVitsVersion.V3;
        var model = Model(Tiny(version, GptSoVitsStage.TextToSemantic));
        int rate = model.SampleRate;
        model.Voice = model.CreateVoice(Tone(32000, 32000, 180, 1), 32000, "Hello there, how are you today?");
        AssertSpeech(model.Synthesize("Good morning to you."));

        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 1 }), Text = "Hello world, this is a test.", Audio = Tone(rate * 2 / 5, rate, 220, 2),
        };
        Assert.False(double.IsNaN(model.Train(sample)));
        var staged = Model(Tiny(version, flow ? GptSoVitsStage.FlowMatching : GptSoVitsStage.SoVits));
        Assert.False(double.IsNaN(staged.Train(sample)));
        double before = staged.EvaluateTrainingObjective(sample);
        Assert.False(double.IsNaN(before));
    }

    /// <summary>The released v2ProPlus weights (<c>GPTSOVITS_DIR</c>: the release's <c>pretrained_models</c>): every tensor
    /// loads and a reference-free English sentence comes out as finite 32 kHz audio.</summary>
    [Fact(Timeout = 3600000)]
    public async Task ReleasedV2ProPlus_LoadsAndSpeaks()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("GPTSOVITS_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "v2Pro", "s2Gv2ProPlus.pth"))) return;
        var model = Model(new GPTSoVITSOptions
        {
            TextLanguage = GptSoVitsTextLanguage.English, PromptLanguage = GptSoVitsTextLanguage.English, MaxSeconds = 4,
        });
        model.LoadReferenceWeights(new GptSoVitsCheckpoints
        {
            TextToSemantic = Path.Combine(dir, "s1v3.ckpt"),
            Synthesizer = Path.Combine(dir, "v2Pro", "s2Gv2ProPlus.pth"),
            Hubert = Path.Combine(dir, "chinese-hubert-base", "pytorch_model.bin"),
            Bert = Path.Combine(dir, "chinese-roberta-wwm-ext-large", "pytorch_model.bin"),
            SpeakerVerification = Path.Combine(dir, "sv", "pretrained_eres2netv2w24s4ep4.ckpt"),
        });
        model.Voice = model.CreateVoice(Tone(32000 * 4, 32000, 150, 3), 32000);
        var audio = model.Synthesize("Hello, this is a test.");
        Assert.Equal(32000, model.SampleRate);
        AssertSpeech(audio);
    }
}
