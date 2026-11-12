using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.Interfaces;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Regression tests for Voicebox's duration/context, OT-CFM, midpoint, and adapted-vocoder pipeline.</summary>
public class VoiceboxPaperTests
{
    private static VoiceboxOptions SmallOptions() => new()
    {
        PhoneVocabularySize = 320,
        UseNativeEnglishFallback = true,
        PhoneEmbeddingDim = 8,
        AudioModelDim = 16,
        AudioFeedForwardDim = 32,
        AudioLayers = 2,
        AudioHeads = 4,
        AudioAttentionDropout = 0,
        AudioActivationDropout = 0,
        AudioConvPositionWidth = 3,
        AudioConvPositionGroups = 4,
        AudioConvPositionDepth = 1,
        DurationModelDim = 16,
        DurationFeedForwardDim = 32,
        DurationLayers = 2,
        DurationHeads = 4,
        DurationAttentionDropout = 0,
        DurationActivationDropout = 0,
        DurationConvPositionWidth = 3,
        DurationConvPositionGroups = 4,
        DurationConvPositionDepth = 1,
        OdeSteps = 1,
        GuidanceStrength = 0.2,
        MaximumAudioFrames = 64,
        MaximumPhoneDuration = 4,
        MaxTextLength = 16,
        VocoderInitialChannels = 16,
        VocoderResBlockKernels = [3],
        VocoderResBlockDilations = [[1]],
        SamplingSeed = 7,
    };

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Generative, inputSize: 16, outputSize: 80);

    private static Voicebox<double> Create() => new(Architecture(), SmallOptions());

    [Fact]
    public void ConfiguredLearningRateChangesNativeTrainingUpdates()
    {
        var slowOptions = SmallOptions();
        slowOptions.LearningRate = 1e-4;
        var fastOptions = new VoiceboxOptions(slowOptions) { LearningRate = 1e-3 };
        var slowArchitecture = Architecture();
        var fastArchitecture = Architecture();
        slowArchitecture.RandomSeed = fastArchitecture.RandomSeed = 37;
        using var slow = new Voicebox<double>(slowArchitecture, slowOptions);
        using var fast = new Voicebox<double>(fastArchitecture, fastOptions);
        fast.SetParameters(slow.GetParameters());
        Assert.Equal(slow.GetParameters().ToArray(), fast.GetParameters().ToArray());
        var sample = new TtsTrainingSample<double>
        {
            Tokens = Ids(2, 3, 4), Mel = Mel(3), Durations = [1, 1, 1],
        };
        for (int step = 0; step < 2; step++)
        {
            slow.Train(sample);
            fast.Train(sample);
        }
        Assert.True(slow.GetParameters().Zip(fast.GetParameters(),
            (a, b) => Math.Abs(a - b)).Any(delta => delta > 1e-12));
    }

    [Fact]
    public void SynthesisReportsInputThatExceedsItsFrameBudget()
    {
        var options = SmallOptions();
        options.MaximumAudioFrames = 1;
        using var model = new Voicebox<double>(Architecture(), options);
        var error = Assert.Throws<ArgumentException>(() => model.Predict(Ids(1, 2, 3, 4)));
        Assert.Contains("frame limit", error.Message);
    }

    private static Tensor<double> Ids(params int[] ids)
    {
        var tensor = new Tensor<double>(new[] { ids.Length });
        for (int i = 0; i < ids.Length; i++) tensor[i] = ids[i];
        return tensor;
    }

    private static Tensor<double> Mel(int frames)
    {
        var mel = new Tensor<double>(new[] { frames, 80 });
        for (int f = 0; f < frames; f++)
            for (int m = 0; m < 80; m++) mel[f, m] = -5.8 + 0.02 * Math.Sin(0.3 * f + 0.07 * m);
        return mel;
    }

    private static Tensor<double> Audio(int samples = 320)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++) audio[i] = 0.2 * Math.Sin(0.13 * i);
        return audio;
    }

    [Fact]
    public void ReferenceDefaults_MatchTablesD13D14AndAppendixA()
    {
        var options = new VoiceboxOptions();

        Assert.Equal(16000, options.SampleRate);
        Assert.Equal(1024, options.FftSize);
        Assert.Equal(640, options.WindowLength);
        Assert.Equal(160, options.HopSize);
        Assert.Equal(80, options.MelChannels);
        Assert.Equal(-5.8843, options.MelMean);
        Assert.Equal(2.2615, options.MelStandardDeviation);
        Assert.Equal(24, options.AudioLayers);
        Assert.Equal(1024, options.AudioModelDim);
        Assert.Equal(16, options.AudioHeads);
        Assert.Equal(4096, options.AudioFeedForwardDim);
        Assert.Equal(31, options.AudioConvPositionWidth);
        Assert.Equal(16, options.AudioConvPositionGroups);
        Assert.Equal(2, options.AudioConvPositionDepth);
        Assert.Equal(8, options.DurationLayers);
        Assert.Equal(512, options.DurationModelDim);
        Assert.Equal(8, options.DurationHeads);
        Assert.Equal(2048, options.DurationFeedForwardDim);
        Assert.Equal(15, options.DurationConvPositionWidth);
        Assert.Equal(1e-5, options.SigmaMinimum);
        Assert.Equal(32, 2 * options.OdeSteps);
        Assert.Equal(0.7, options.GuidanceStrength);
        Assert.Equal(new[] { 5, 4, 4, 2 }, options.VocoderUpsampleRates);
        Assert.Equal(new[] { 11, 8, 8, 4 }, options.VocoderUpsampleKernels);
    }

    [Fact]
    public void NativeModel_ExposesSeparateDurationAudioFlowAndVocoderGraph()
    {
        var model = Create();
        var metadata = model.GetModelMetadata();

        Assert.True(model.HasPaperPipeline);
        Assert.Equal(2, model.AudioLayerCount);
        Assert.Equal(2, model.DurationLayerCount);
        Assert.Equal(2, model.NumberOfFunctionEvaluations);
        Assert.Equal(true, metadata.AdditionalInfo["SymmetricBidirectionalAlibi"]);
        Assert.Equal(true, metadata.AdditionalInfo["UnetSkipConnections"]);
        Assert.Equal(false, metadata.AdditionalInfo["OfficialWeightsAvailable"]);
        Assert.False(model is ICodecTts<double>);
        Assert.Equal(0, model.CodecTokenCodebooks);
    }

    [Fact]
    public void PhoneFrontEnd_InsertsGhostSilenceAndWordPositionSuffixes()
    {
        var model = Create();

        var phones = model.Phonemize("hello world").ToArray();

        Assert.Equal("SIL", phones[0]);
        Assert.Equal("SIL", phones[^1]);
        Assert.Equal(3, phones.Count(phone => phone == "SIL"));
        Assert.Contains(phones, phone => phone.EndsWith("_B", StringComparison.Ordinal));
        Assert.Contains(phones, phone => phone.EndsWith("_E", StringComparison.Ordinal));
        Assert.All(model.TextToTokens("hello world").ToArray(),
            id => Assert.InRange(id, 0, SmallOptions().PhoneVocabularySize - 1));
    }

    [Fact]
    public void TextOnlySynthesis_RunsRegressionDurationMidpointFlowAndAdaptedHifiGan()
    {
        var model = Create();

        var waveform = model.Synthesize("hi").ToArray();

        Assert.NotEmpty(waveform);
        Assert.Equal(0, waveform.Length % 160);
        Assert.All(waveform, value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Fact]
    public void Infill_PreservesUnmaskedContextFramesExactly()
    {
        var model = Create();
        var normalized = new Tensor<double>(new[] { 3, 80 });
        for (int i = 0; i < normalized.Length; i++) normalized[i] = 0.01 * i;

        var output = model.Infill(normalized, new[] { 2, 2, 2 }, new[] { false, true, false });

        for (int m = 0; m < 80; m++)
        {
            Assert.Equal(normalized[0, m], output[0, m]);
            Assert.Equal(normalized[2, m], output[2, m]);
            Assert.True(TtsPaperNumerics.IsFinite(output[1, m]));
        }
    }

    [Fact]
    public void MaskedOtCfmAndDurationObjective_IsFinite()
    {
        var model = Create();
        var sample = new TtsTrainingSample<double>
        {
            Tokens = Ids(2, 3, 4),
            Mel = Mel(3),
            Durations = [1, 1, 1],
        };

        double objective = model.EvaluateTrainingObjective(sample);

        Assert.True(TtsPaperNumerics.IsFinite(objective), $"Objective was {objective}.");
        Assert.True(objective >= 0);
    }

    [Fact]
    public void PaperObjectiveTraining_UpdatesNativeParameters()
    {
        var model = Create();
        var sample = new TtsTrainingSample<double>
        {
            Tokens = Ids(2, 3, 4),
            Mel = Mel(3),
            Durations = [1, 1, 1],
        };
        var before = model.GetParameters().ToArray();

        double loss = model.Train(sample);
        var after = model.GetParameters().ToArray();

        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Any(delta => delta > 0));
    }

    [Fact]
    public void ZeroShotVoice_RequiresAndUsesForcedAlignment()
    {
        var model = Create();
        int phones = model.TextToTokens("hi").Length;
        var durations = new int[phones];
        durations[^1] = 3;

        model.Voice = model.CreateVoice(Audio(), 16000, "hi", durations);
        var waveform = model.Synthesize("ok").ToArray();

        Assert.NotEmpty(waveform);
        Assert.All(waveform, value => Assert.True(TtsPaperNumerics.IsFinite(value)));
        Assert.Throws<ArgumentException>(() => model.CreateVoice(Audio(), 16000, "hi", [1]));
    }

    [Fact]
    public void ConvertedCheckpoint_IsStrictAndRestoresCompletePipeline()
    {
        var source = Create();
        var expected = source.GetParameters().ToArray();
        var restored = Create();

        restored.LoadConvertedWeights(name => name == "voicebox.parameters" ? expected : null);

        Assert.Equal(expected, restored.GetParameters().ToArray());
        Assert.Throws<InvalidDataException>(() =>
            restored.LoadConvertedWeights(_ => new double[expected.Length - 1]));
    }

    [Fact]
    public void CopyConstructor_PreservesVoiceboxSettingsWithoutAliasingArrays()
    {
        var original = SmallOptions();
        var copy = new VoiceboxOptions(original);
        copy.VocoderUpsampleRates[0] = 1;
        copy.VocoderResBlockDilations[0][0] = 9;

        Assert.Equal(5, original.VocoderUpsampleRates[0]);
        Assert.Equal(1, original.VocoderResBlockDilations[0][0]);
        Assert.Equal(original.AudioModelDim, copy.AudioModelDim);
        Assert.Equal(original.DurationModelDim, copy.DurationModelDim);
        Assert.Equal(original.SamplingSeed, copy.SamplingSeed);
    }
}
