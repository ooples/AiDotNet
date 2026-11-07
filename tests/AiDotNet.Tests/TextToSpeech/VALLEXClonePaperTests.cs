using AiDotNet.Audio.Generation;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TextToSpeech;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>Regression tests for the voice-cloning facade over the native VALL-E X graph.</summary>
public class VALLEXClonePaperTests
{
    [Theory]
    [InlineData("WarmupSteps")]
    [InlineData("TrainingSteps")]
    public void NativeLearningRateScheduleAdvancesForEachTrainingBatch(string option)
    {
        var controlOptions = Tiny();
        controlOptions.WarmupSteps = 0;
        controlOptions.TrainingSteps = 8;
        var changedOptions = new VALLEXCloneOptions(controlOptions);
        if (option == "WarmupSteps") changedOptions.WarmupSteps = 4;
        else changedOptions.TrainingSteps = 2;
        using var control = Create(controlOptions);
        using var changed = Create(changedOptions);
        var sample = new TtsTrainingSample<double>
        {
            Tokens = control.TextToTokens("你好"), CodecTokens = Codes(6, 0), PromptCodecTokens = Codes(3, 17),
            LanguageId = (int)VallEXLanguage.Mandarin,
        };
        Assert.Equal(control.GetParameters().ToArray(), changed.GetParameters().ToArray());
        for (int step = 0; step < 2; step++) { control.Train(sample); changed.Train(sample); }
        Assert.False(control.GetParameters().SequenceEqual(changed.GetParameters()));
    }

    [Fact]
    public void InitializationSeedIsReproducibleAndSeparateFromSamplingSeed()
    {
        var firstOptions = Tiny(); firstOptions.Seed = 11;
        var repeatOptions = new VALLEXCloneOptions(firstOptions) { SamplingSeed = 1 };
        var changedOptions = new VALLEXCloneOptions(firstOptions) { Seed = 12 };
        using var first = Create(firstOptions);
        using var repeat = Create(repeatOptions);
        using var changed = Create(changedOptions);
        Assert.Equal(first.GetParameters().ToArray(), repeat.GetParameters().ToArray());
        Assert.False(first.GetParameters().SequenceEqual(changed.GetParameters()));
    }

    [Theory]
    [InlineData(17, 2)]
    [InlineData(16, 3)]
    [InlineData(16, 0)]
    [InlineData(0, 2)]
    public void InvalidTransformerWidthsAreRejectedBeforeInference(int width, int heads)
    {
        var options = Tiny();
        options.HiddenDim = width;
        options.NumHeads = heads;
        Assert.Throws<ArgumentException>(() => Create(options));
    }

    private static VALLEXCloneOptions Tiny(string referenceText = "a prompt") => VallEXCloneFixture.Options(referenceText);

    private static NeuralNetworkArchitecture<double> Architecture() =>
        new(InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 7 };

    private static VALLEXClone<double> Create(VALLEXCloneOptions? options = null) =>
        new(Architecture(), options ?? Tiny());

    private static Tensor<double> Tone(int samples = 640)
    {
        var audio = new Tensor<double>(new[] { samples });
        for (int i = 0; i < samples; i++)
            audio[i] = 0.35 * Math.Sin(2 * Math.PI * 180 * i / 24000.0)
                + 0.1 * Math.Sin(2 * Math.PI * 510 * i / 24000.0);
        return audio;
    }

    private static Tensor<double> Codes(int frames, int offset)
    {
        var codes = new Tensor<double>(new[] { frames, 3 });
        for (int frame = 0; frame < frames; frame++)
            for (int q = 0; q < 3; q++) codes[frame, q] = (frame * 5 + q * 11 + offset) % 64;
        return codes;
    }

    [Fact]
    public void Defaults_AreThePaperVallEXGraphAndThreeSecondEnrollment()
    {
        var options = new VALLEXCloneOptions();

        Assert.Equal(24000, options.SampleRate);
        Assert.Equal(320, options.HopSize);
        Assert.Equal(1024, options.HiddenDim);
        Assert.Equal(16, options.NumHeads);
        Assert.Equal(12, options.NumEncoderLayers);
        Assert.Equal(12, options.NumDecoderLayers);
        Assert.Equal(4096, options.FeedForwardDim);
        Assert.Equal(8, options.NumCodebooks);
        Assert.Equal(1024, options.CodebookSize);
        Assert.Equal(8000, options.WarmupSteps);
        Assert.Equal(3.0, options.MinReferenceDurationSec);
    }

    [Fact]
    public void Facade_InheritsTheCanonicalBilingualArNarAndCodecGraph()
    {
        var options = Tiny();
        using var clone = Create(options);
        using var canonical = new VALLEX<double>(Architecture(), new VALLEXOptions(options));

        Assert.IsAssignableFrom<VALLEX<double>>(clone);
        Assert.Equal(canonical.GetParameters().Length, clone.GetParameters().Length);
        Assert.Equal(canonical.AutoRegressiveLayers.Count, clone.AutoRegressiveLayers.Count);
        Assert.Equal(canonical.NonAutoRegressiveLayers.Count, clone.NonAutoRegressiveLayers.Count);
        Assert.Equal(3, clone.SpeakerEmbeddingDim);
        Assert.Equal("VoiceCloning", clone.GetModelMetadata().AdditionalInfo["TaskFacade"]);
    }

    [Fact]
    public void TranscriptlessInterface_RejectsMissingEquationThreeCondition()
    {
        using var model = Create(Tiny(referenceText: string.Empty));

        var error = Assert.Throws<InvalidOperationException>(() => model.SynthesizeWithVoice("你好", Tone()));

        Assert.Contains("reference transcript", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ExtractSpeakerEmbedding_ReturnsActualVariableLengthCodecPrompt()
    {
        using var model = Create();
        var audio = Tone();

        var representation = model.ExtractSpeakerEmbedding(audio);
        var direct = model.EncodeToTokens(audio);

        Assert.Equal(2, representation.Rank);
        Assert.Equal(3, representation.Shape[1]);
        Assert.Equal(direct.ToArray(), representation.ToArray());
        Assert.All(representation.ToArray(), value => Assert.InRange(value, 0, 63));
    }

    [Fact]
    public void CrossLingualClone_RunsCanonicalPromptedArAndNarInference()
    {
        using var model = Create();

        var waveform = model.SynthesizeWithVoice("你好，世界。", "a prompt", Tone());

        Assert.NotEmpty(waveform.ToArray());
        Assert.Equal(0, waveform.Length % 320);
        Assert.All(waveform.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData(VallETrainingStage.AutoRegressive)]
    [InlineData(VallETrainingStage.NonAutoRegressive)]
    public void FacadeTraining_UsesTheCanonicalStageSpecificObjective(VallETrainingStage stage)
    {
        using var model = Create();
        model.CurrentStage = stage;
        var sample = new TtsTrainingSample<double>
        {
            Tokens = model.TextToTokens("你好"),
            CodecTokens = Codes(6, 0),
            PromptCodecTokens = Codes(3, 17),
            LanguageId = (int)VallEXLanguage.Mandarin,
        };
        var before = model.GetParameters().ToArray();

        double objective = model.EvaluateTrainingObjective(sample);
        double loss = model.Train(sample);
        var after = model.GetParameters().ToArray();

        Assert.True(TtsPaperNumerics.IsFinite(objective));
        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.True(before.Zip(after, (a, b) => Math.Abs(a - b)).Any(delta => delta > 0));
    }

    [Fact]
    public void ReferenceCheckpointLoader_IsInheritedFromTheCanonicalImplementation()
    {
        var method = typeof(VALLEXClone<double>).GetMethod(nameof(VALLEX<double>.LoadReferenceWeights));

        Assert.NotNull(method);
        Assert.Equal(typeof(VallEModelBase<double>), method!.DeclaringType);
        Assert.ThrowsAny<IOException>(() => Create().LoadReferenceWeights("missing-vallex-checkpoint.safetensors"));
    }

    [Fact]
    public void CopyConstructor_DeepCopiesCodecAndFacadeSettings()
    {
        var original = Tiny();
        var copy = new VALLEXCloneOptions(original);
        copy.Codec.Ratios[0] = 4;
        copy.ReferenceText = "changed";

        Assert.Equal(8, original.Codec.Ratios[0]);
        Assert.Equal("a prompt", original.ReferenceText);
        Assert.Equal(original.HiddenDim, copy.HiddenDim);
        Assert.Equal(original.MinReferenceDurationSec, copy.MinReferenceDurationSec);
    }
}
