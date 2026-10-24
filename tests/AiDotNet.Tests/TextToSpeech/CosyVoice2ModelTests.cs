using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.Tokenization.Algorithms;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The CosyVoice 2 model class: each training stage updates its own network only, a zero-shot voice keeps two mel frames
/// per prompt token as <c>frontend_zero_shot</c> does at 24 kHz, the language model reads the prompt only in zero-shot, and
/// synthesis gives 24 kHz audio (the components are checked against the official code in
/// <see cref="CosyVoice2LmReferenceTests"/>, <see cref="CosyVoice2FlowReferenceTests"/>, <see cref="HiftReferenceTests"/>,
/// <see cref="S3TokenizerV2ReferenceTests"/> and <see cref="QwenTokenizerReferenceTests"/>).
/// </summary>
public class CosyVoice2ModelTests
{
    private static CosyVoice2Options SmallOptions() => new()
    {
        LLMDim = 16, NumLLMLayers = 1, NumHeads = 2, KeyValueHeads = 1, LmFeedForwardDim = 32, FlowDim = 16, FlowHeads = 2,
        FlowFeedForwardDim = 32, FlowEncoderLayers = 1, StaticChunkSize = 2, DecoderChannels = new[] { 16 }, DecoderHeads = 2,
        DecoderHeadDim = 8, DecoderBlocks = 1, DecoderMidBlocks = 1, FlowTimesteps = 2, HiftBaseChannels = 16, HiftHarmonics = 2,
        // S3Tokenizer v2's rotary table fixes its heads at 64 wide and its FSQ at 6,561 codes.
        SpeechTokenizerState = 64, SpeechTokenizerHeads = 1, SpeechTokenizerLayers = 1, SpeakerEncoderChannels = 8,
        SpeakerEncoderGrowthRate = 4, SpeakerEmbeddingDim = 8, MelChannels = 8, DropoutRate = 0.0,
    };

    private static NeuralNetworkArchitecture<double> Architecture() => new(
        AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1);

    private static CosyVoice2<double> Small() => new(Architecture(), SmallOptions());

    private static Tensor<double> Tone(int samples, int rate)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.4 * Math.Sin(2 * Math.PI * 180.0 * i / rate) + 0.2 * Math.Sin(2 * Math.PI * 470.0 * i / rate);
        return audio;
    }

    private static double[] Snapshot(IEnumerable<LayerBase<double>> layers) =>
        AiDotNet.Training.TapeTrainingStep<double>.CollectParameters(layers.Cast<ILayer<double>>().ToList(), -1)
            .SelectMany(p => p.ToArray()).ToArray();

    private static int[] Ints(Tensor<double>? t) => t is null ? Array.Empty<int>() : t.ToArray().Select(v => (int)Math.Round(v)).ToArray();

    private static TtsTrainingSample<double> Sample(CosyVoice2<double> model)
    {
        var text = model.Tokenize("hello there");
        var tokens = new Tensor<double>(new[] { text.Length }, new Vector<double>(text.Select(i => (double)i).ToArray()));
        // 20 tokens and 43 frames: the flow keeps min(43 / 2, 20) = 20 tokens and their 40 frames.
        var speech = new Tensor<double>(new[] { 20 }, new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)((977 * i + 3) % 6561)).ToArray()));
        var mel = new Tensor<double>(new[] { 43, 8 });
        for (int i = 0; i < mel.Length; i++) mel[i] = Math.Sin(0.13 * i) - 4;
        var xvector = new Tensor<double>(new[] { 8 }, new Vector<double>(Enumerable.Range(0, 8).Select(i => Math.Cos(i + 1.0)).ToArray()));
        return new TtsTrainingSample<double> { Tokens = tokens, CodecTokens = speech, Mel = mel, SpeakerReference = xvector };
    }

    [Fact(Timeout = 300000)]
    public async Task EachStage_TrainsItsOwnNetworkOnly()
    {
        await Task.Yield();
        using var model = Small();
        var sample = Sample(model);
        foreach (var stage in new[] { CosyVoiceTrainingStage.LanguageModel, CosyVoiceTrainingStage.Flow })
        {
            model.CurrentStage = stage;
            var lm = Snapshot(model.LanguageModelLayers);
            var flow = Snapshot(model.FlowLayers);
            var others = model.FrozenLayers;
            var rest = Snapshot(others);
            double loss = Convert.ToDouble(model.Train(sample));
            Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss) && loss > 0, $"{stage}: loss {loss}.");
            Assert.Equal(stage == CosyVoiceTrainingStage.LanguageModel, !lm.SequenceEqual(Snapshot(model.LanguageModelLayers)));
            Assert.Equal(stage == CosyVoiceTrainingStage.Flow, !flow.SequenceEqual(Snapshot(model.FlowLayers)));
            Assert.Equal(rest, Snapshot(others));
        }
    }

    [Fact(Timeout = 300000)]
    public async Task ZeroShotVoice_KeepsTwoFramesPerToken_AndDrivesSynthesis()
    {
        await Task.Yield();
        using var model = Small();
        // 1.1 s: S3 v2 gives 28 tokens at 25 Hz, the 24 kHz mel 55 frames (hop 480, reflect-padded by 720 each side), so
        // frontend_zero_shot keeps min(55 / 2, 28) = 27 tokens and their 54 frames.
        var recording = Tone(17600, 16000);
        int raw = model.SpeechTokens(recording, 16000).Length;
        var voice = model.CreateVoice(recording, 16000, "a short prompt");
        Assert.Equal(28, raw);
        Assert.Equal(27, voice.ReferenceCodecTokens?.Length);
        Assert.Equal(new[] { 54, 8 }, voice.Reference?.Shape.ToArray());
        Assert.Equal(model.SpeechTokens(recording, 16000).Take(27), Ints(voice.ReferenceCodecTokens));
        Assert.Equal(8, voice.SpeakerEmbedding?.Length);
        Assert.Equal(QwenTokenizer.CosyVoice2.Encode("a short prompt"), Ints(voice.ReferenceTokens));

        model.Voice = voice;
        var audio = model.Synthesize("hi");
        Assert.True(audio.Length > 0 && audio.Length % 480 == 0, $"{audio.Length} samples.");
        Assert.All(audio.ToArray(), v => Assert.True(!double.IsNaN(v) && !double.IsInfinity(v) && Math.Abs(v) <= 0.99));
    }

    [Fact(Timeout = 300000)]
    public async Task Voices_RouteThePromptAsTheFrontEndModesDo()
    {
        await Task.Yield();
        using var model = Small();
        var recording = Tone(16000, 16000);
        var zeroShot = model.CreateVoice(recording, 16000, "a short prompt");
        var crossLingual = model.CreateVoice(recording, 16000);
        var lm = model.LanguageModel ?? throw new InvalidOperationException("native model");
        var text = model.Tokenize("hi there");

        // frontend_zero_shot: Qwen2LM continues the prompt's text and speech tokens (it reads no x-vector).
        model.Voice = zeroShot;
        Assert.Equal(lm.Generate(text, Ints(zeroShot.ReferenceTokens), Ints(zeroShot.ReferenceCodecTokens), new Random(5)),
            model.GenerateSpeechTokens(text, new Random(5)));
        // frontend_cross_lingual: no prompt in the LM.
        model.Voice = crossLingual;
        Assert.Equal(lm.Generate(text, Array.Empty<int>(), Array.Empty<int>(), new Random(5)),
            model.GenerateSpeechTokens(text, new Random(5)));
    }

    [Fact(Timeout = 120000)]
    public async Task Tokenize_IsQwensTokenizer_WithCosyVoice2Tokens()
    {
        await Task.Yield();
        using var model = Small();
        Assert.Equal(151936, model.GetOptions() is CosyVoice2Options o ? o.VocabSize : 0);
        Assert.Equal(QwenTokenizer.CosyVoice2.Encode("<|endofprompt|>Hello [laughter]"), model.Tokenize("<|endofprompt|>Hello [laughter]"));
        Assert.Throws<ArgumentException>(() => new CosyVoice2<double>(Architecture(), new CosyVoice2Options { VocabSize = 1000 }));
    }

    [Fact(Timeout = 600000)]
    public async Task VocoderStage_TrainsHiftAndItsDiscriminatorsOnly()
    {
        await Task.Yield();
        var options = SmallOptions();
        options.MelChannels = 80;
        options.IncludeDiscriminators = true;
        using var model = new CosyVoice2<double>(Architecture(), options);
        model.CurrentStage = CosyVoiceTrainingStage.Vocoder;
        var hift = Snapshot(model.VocoderLayers);
        var discriminators = Snapshot(model.DiscriminatorLayers);
        var lm = Snapshot(model.LanguageModelLayers);
        var flow = Snapshot(model.FlowLayers);
        var sample = new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 1 }), Audio = Tone(30000, 24000), CodecTokens = new Tensor<double>(new[] { 1 }),
        };
        // truncate_length 24,480 = 51 frames of 480.
        var (real, mel, pitch) = model.VocoderTargets(sample, new Random(0));
        Assert.Equal(24480, real.Length);
        Assert.Equal(mel.Shape[2], pitch.Length);

        double loss = Convert.ToDouble(model.Train(sample));
        Assert.True(!double.IsNaN(loss) && !double.IsInfinity(loss) && loss > 0, $"generator loss {loss}.");
        Assert.NotEqual(hift, Snapshot(model.VocoderLayers));
        Assert.NotEqual(discriminators, Snapshot(model.DiscriminatorLayers));
        Assert.Equal(lm, Snapshot(model.LanguageModelLayers));
        Assert.Equal(flow, Snapshot(model.FlowLayers));
    }
}
