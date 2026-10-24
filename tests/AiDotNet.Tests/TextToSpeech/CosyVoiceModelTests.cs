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
/// The CosyVoice model class: its two training stages each update their own network and nothing else, and a zero-shot
/// voice made from a recording drives synthesis (the components themselves are checked against the official code in
/// <see cref="CosyVoiceReferenceTests"/>, <see cref="S3TokenizerReferenceTests"/>, <see cref="CamPlusPlusReferenceTests"/>
/// and <see cref="WhisperTiktokenReferenceTests"/>).
/// </summary>
public class CosyVoiceModelTests
{
    private static CosyVoice<double> Small() => new(new NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
        new CosyVoiceOptions
        {
            TextEncoderDim = 16, LLMDim = 16, NumEncoderLayers = 1, NumLLMLayers = 1, NumHeads = 2, LmFeedForwardDim = 32,
            CodebookSize = 16, FlowDim = 16, FlowHeads = 2, FlowFeedForwardDim = 32, FlowEncoderLayers = 1,
            DecoderChannels = new[] { 16 }, DecoderHeads = 2, DecoderHeadDim = 8, DecoderBlocks = 1, DecoderMidBlocks = 1,
            FlowTimesteps = 2, HiftBaseChannels = 16, HiftHarmonics = 2, SpeechTokenizerState = 16, SpeechTokenizerHeads = 2,
            SpeechTokenizerLayers = 1, SpeakerEncoderChannels = 8, SpeakerEncoderGrowthRate = 4, SpeakerEmbeddingDim = 8,
            MelChannels = 8, DropoutRate = 0.0, WarmupSteps = 1,
        });

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

    private static TtsTrainingSample<double> Sample(CosyVoice<double> model)
    {
        var text = model.Tokenize("hello there");
        var tokens = new Tensor<double>(new[] { text.Length }, new Vector<double>(text.Select(i => (double)i).ToArray()));
        var speech = new Tensor<double>(new[] { 20 }, new Vector<double>(Enumerable.Range(0, 20).Select(i => (double)((7 * i + 3) % 16)).ToArray()));
        var mel = new Tensor<double>(new[] { 34, 8 });
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
            Assert.True(double.IsFinite(loss) && loss > 0, $"{stage}: loss {loss}.");
            Assert.Equal(stage == CosyVoiceTrainingStage.LanguageModel, !lm.SequenceEqual(Snapshot(model.LanguageModelLayers)));
            Assert.Equal(stage == CosyVoiceTrainingStage.Flow, !flow.SequenceEqual(Snapshot(model.FlowLayers)));
            Assert.Equal(rest, Snapshot(others));
        }
    }

    [Fact(Timeout = 300000)]
    public async Task ZeroShotVoice_FromARecording_DrivesSynthesis()
    {
        await Task.Yield();
        using var model = Small();
        var recording = Tone(16000, 16000);
        var voice = model.CreateVoice(recording, 16000, "a short prompt");
        // One second: 50 S3 tokens, 86 mel frames at 22.05 kHz with a 256 hop (not centred), an 8-wide x-vector.
        Assert.Equal(50, voice.ReferenceCodecTokens?.Length);
        Assert.Equal(new[] { 86, 8 }, voice.Reference?.Shape.ToArray());
        Assert.Equal(8, voice.SpeakerEmbedding?.Length);
        Assert.Equal(WhisperTiktoken.Multilingual(100).Encode("a short prompt"), voice.ReferenceTokens?.ToArray().Select(v => (int)v).ToArray());

        model.Voice = voice;
        var audio = model.Synthesize("hi");
        Assert.True(audio.Length > 0 && audio.Length % 256 == 0, $"{audio.Length} samples.");
        Assert.All(audio.ToArray(), v => Assert.True(double.IsFinite(v) && Math.Abs(v) <= 0.99));
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
        int[] Ints(Tensor<double>? t) => t is null ? Array.Empty<int>() : t.ToArray().Select(v => (int)Math.Round(v)).ToArray();

        // frontend_zero_shot: the LM continues the prompt's text and speech tokens, with the x-vector.
        model.Voice = zeroShot;
        Assert.Equal(lm.Generate(text, Ints(zeroShot.ReferenceTokens), Ints(zeroShot.ReferenceCodecTokens), zeroShot.SpeakerEmbedding, new Random(5)),
            model.GenerateSpeechTokens(text, new Random(5)));
        // frontend_cross_lingual: no prompt in the LM, only the x-vector.
        model.Voice = crossLingual;
        Assert.Equal(lm.Generate(text, Array.Empty<int>(), Array.Empty<int>(), crossLingual.SpeakerEmbedding, new Random(5)),
            model.GenerateSpeechTokens(text, new Random(5)));
    }

    private static CosyVoice<double> SmallWithDiscriminators() => new(new NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional, AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1),
        new CosyVoiceOptions
        {
            TextEncoderDim = 16, LLMDim = 16, NumEncoderLayers = 1, NumLLMLayers = 1, NumHeads = 2, LmFeedForwardDim = 32,
            CodebookSize = 16, FlowDim = 16, FlowHeads = 2, FlowFeedForwardDim = 32, FlowEncoderLayers = 1,
            DecoderChannels = new[] { 16 }, DecoderHeads = 2, DecoderHeadDim = 8, DecoderBlocks = 1, DecoderMidBlocks = 1,
            FlowTimesteps = 2, HiftBaseChannels = 16, HiftHarmonics = 2, SpeechTokenizerState = 16, SpeechTokenizerHeads = 2,
            SpeechTokenizerLayers = 1, SpeakerEncoderChannels = 8, SpeakerEncoderGrowthRate = 4, SpeakerEmbeddingDim = 8,
            DropoutRate = 0.0, WarmupSteps = 1, IncludeDiscriminators = true,
        });

    [Fact(Timeout = 300000)]
    public async Task VocoderTargets_AreCosyVoicesComputeF0()
    {
        await Task.Yield();
        using var model = SmallWithDiscriminators();
        var f = WorldHarvestFixture();
        var signal = f["signals"]!["22050"]!.Select(v => (double)v).ToArray();
        var audio = new Tensor<double>(new[] { signal.Length }, new Vector<double>(signal));
        // Shorter than the segment, so truncate zero-pads it: no crop is drawn.
        var (real, mel, pitch) = model.VocoderTargets(new TtsTrainingSample<double> { Tokens = new Tensor<double>(new[] { 1 }), Audio = audio }, new Random(0));
        Assert.Equal(24576, real.Length);
        Assert.Equal(new[] { 1, 80, 96 }, mel.Shape.ToArray());
        var expected = f["compute_f0"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expected.Length, pitch.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - pitch[i]) <= 1e-8 * Math.Max(1, expected[i]), $"frame {i}: CosyVoice {expected[i]}, port {pitch[i]}.");
    }

    [Fact(Timeout = 600000)]
    public async Task VocoderStage_TrainsHiftAndItsDiscriminatorsOnly()
    {
        await Task.Yield();
        using var model = SmallWithDiscriminators();
        model.CurrentStage = CosyVoiceTrainingStage.Vocoder;
        var hift = Snapshot(model.VocoderLayers);
        var discriminators = Snapshot(model.DiscriminatorLayers);
        var lm = Snapshot(model.LanguageModelLayers);
        var flow = Snapshot(model.FlowLayers);
        var audio = Tone(30000, 22050);
        double loss = Convert.ToDouble(model.Train(new TtsTrainingSample<double>
        {
            Tokens = new Tensor<double>(new[] { 1 }), Audio = audio, CodecTokens = new Tensor<double>(new[] { 1 }),
        }));
        Assert.True(double.IsFinite(loss) && loss > 0, $"generator loss {loss}.");
        Assert.NotEqual(hift, Snapshot(model.VocoderLayers));
        Assert.NotEqual(discriminators, Snapshot(model.DiscriminatorLayers));
        Assert.Equal(lm, Snapshot(model.LanguageModelLayers));
        Assert.Equal(flow, Snapshot(model.FlowLayers));
    }

    private static Newtonsoft.Json.Linq.JObject WorldHarvestFixture()
    {
        const string fileName = "world_harvest_reference.json";
        string output = System.IO.Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!System.IO.File.Exists(output))
        {
            var dir = new System.IO.DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !System.IO.File.Exists(output))
            {
                output = System.IO.Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return Newtonsoft.Json.Linq.JObject.Parse(System.IO.File.ReadAllText(output));
    }

    [Fact(Timeout = 120000)]
    public async Task Tokenize_IsWhispersHundredLanguageEncoding_WithSpecialsAllowed()
    {
        await Task.Yield();
        using var model = Small();
        Assert.Equal(51866, model.GetOptions() is CosyVoiceOptions o ? o.VocabSize : 0);
        Assert.Equal(WhisperTiktoken.Multilingual(100).Encode("<|en|>Hello"), model.Tokenize("<|en|>Hello"));
    }
}
