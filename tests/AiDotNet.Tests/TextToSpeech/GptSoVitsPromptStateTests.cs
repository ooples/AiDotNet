using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.TextToSpeech.CodecBased;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class GptSoVitsPromptStateTests
{
    [Fact]
    public void OptionProbesMeasureArchitectureEvenWhenTheirTaskObserverIsConstant()
    {
        var probe = new ModelFamilyTests.Base.OptionSensitivityProbe<GPTSoVITS<double>>(
            () => new GPTSoVITS<double>(new NeuralNetworkArchitecture<double>(InputType.OneDimensional,
                NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), GptSoVitsFixture.Options()),
            _ => 0, train: null,
            parameterCount: ModelFamilyTests.Base.OptionSensitivityFixtureSetup.ParameterCountBeforeFitting);
        var report = ModelFamilyTests.Base.OptionSensitivityHarness.Measure(GetType(), "HiddenDim", probe);
        Assert.Equal(ModelFamilyTests.Base.OptionSensitivityOutcome.Sensitive, report.Outcome);
        Assert.Contains("parameter count", report.Detail);
    }

    [Theory]
    [InlineData("DitDim")]
    [InlineData("DitDepth")]
    [InlineData("DitHeads")]
    [InlineData("DitHeadDim")]
    [InlineData("DitConvLayers")]
    [InlineData("BridgeChannels")]
    public void FlowArchitectureChangesActualNativeMelGeneration(string option)
    {
        var controlOptions = GptSoVitsFixture.Options();
        var changedOptions = (GPTSoVITSOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(controlOptions);
        var property = typeof(GPTSoVITSOptions).GetProperty(option)!;
        property.SetValue(changedOptions, 2 * (int)property.GetValue(changedOptions)!);
        var architecture = new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 1337 };
        using var control = new GPTSoVITS<float>(architecture, controlOptions);
        using var changed = new GPTSoVITS<float>(architecture, changedOptions);
        var input = new Tensor<float>(new[] { 3 });
        Assert.NotEqual(GptSoVitsFixture.Observe(control, input, option), GptSoVitsFixture.Observe(changed, input, option));
    }

    [Theory]
    [InlineData(16)]
    [InlineData(32)]
    public void BoundedDiscriminatorsRunTheCompleteNativeGanTrainingStep(int divisor)
    {
        var options = GptSoVitsFixture.Options();
        options.Version = GptSoVitsVersion.V2;
        options.CurrentStage = GptSoVitsStage.SoVits;
        options.DiscriminatorWidthDivisor = divisor;
        options.SegmentSize = 2560;
        using var model = new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 1337 }, options);
        var audio = new Tensor<float>(new[] { 12800 });
        for (int i = 0; i < audio.Length; i++) audio[i] = (float)(0.3 * Math.Sin(i * 0.13) + 0.1 * Math.Cos(i * 0.037));
        var before = model.GetParameters().ToArray();
        var loss = model.Train(new AiDotNet.TextToSpeech.TtsTrainingSample<float>
        {
            Tokens = new Tensor<float>(new[] { 3 }, new Vector<float>(new[] { 1f, 2f, 3f })), Audio = audio,
        });
        Assert.True(TtsPaperNumerics.IsFinite(loss));
        Assert.False(before.SequenceEqual(model.GetParameters()));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void InvalidDiscriminatorDivisorsAreRejectedBeforeAllocation(int divisor)
    {
        var options = GptSoVitsFixture.Options();
        options.DiscriminatorWidthDivisor = divisor;
        Assert.Throws<ArgumentException>(() => new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options));
    }

    [Theory]
    [InlineData("SynthesizerDropout")]
    [InlineData("TextLanguage")]
    public void NativeTaskContextExercisesTrainingDropoutAndFrontendDependencies(string option)
    {
        var controlOptions = GptSoVitsFixture.Options();
        GptSoVitsFixture.PrepareOptions(controlOptions, option);
        var changedOptions = (GPTSoVITSOptions)AiDotNet.Models.CloneEngine.CopyConfiguration(controlOptions);
        if (option == "SynthesizerDropout") changedOptions.SynthesizerDropout = 0.2;
        else changedOptions.TextLanguage = AiDotNet.TextToSpeech.FrontEnd.GptSoVitsTextLanguage.Chinese;
        var architecture = new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 1337 };
        using var control = new GPTSoVITS<float>(architecture, controlOptions);
        using var changed = new GPTSoVITS<float>(architecture, changedOptions);
        var input = new Tensor<float>(new[] { 3 }, new Vector<float>(new[] { 1f, 2f, 3f }));
        Assert.NotEqual(GptSoVitsFixture.Observe(control, input, option), GptSoVitsFixture.Observe(changed, input, option));
    }

    [Theory]
    [InlineData(320)]
    [InlineData(640)]
    public void PairedGanUpsamplingControlsActualDecodedWaveformLength(int hop)
    {
        var options = GptSoVitsFixture.Options();
        options.Version = GptSoVitsVersion.V2;
        options.HopSize = hop;
        options.SegmentSize = 4 * hop;
        options.UpsampleRates = hop == 320 ? new[] { 8, 5, 4, 2 } : new[] { 10, 8, 2, 2, 2 };
        options.UpsampleKernels = options.UpsampleRates.Select(rate => 2 * rate + rate % 2).ToArray();
        using var model = new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 1337 }, options);
        var reference = new Tensor<float>(new[] { 1, model.SynthesizerConfiguration().SpecChannels, 8 });
        for (int i = 0; i < reference.Length; i++) reference[i] = (float)(0.2 * Math.Sin(i * 0.13));
        var audio = model.SoVits.Decode(new[] { 1, 2, 3 }, new[] { 1, 2, 3 }, new[] { reference }, null,
            0, 1, shape => new Tensor<float>(shape));
        Assert.Equal(6 * hop, audio.Length);
        Assert.All(audio.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    public void OddVocoderResidualKernelsRunNativeInference(int kernel)
    {
        var options = GptSoVitsFixture.Options();
        options.VocoderResblockKernels = new[] { kernel };
        using var model = new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 1337 }, options);
        GptSoVitsFixture.Prepare(model);
        model.SetTrainingMode(false);
        var audio = model.Predict(new Tensor<float>(new[] { 3 }, new Vector<float>(new[] { 1f, 2f, 3f })));
        Assert.NotEmpty(audio.ToArray());
        Assert.All(audio.ToArray(), value => Assert.True(TtsPaperNumerics.IsFinite(value)));
    }

    [Theory]
    [InlineData(GptSoVitsVersion.V3)]
    [InlineData(GptSoVitsVersion.V5Dev)]
    public void ClonePreservesTranscriptAndOriginalReferenceSampleRate(GptSoVitsVersion version)
    {
        var options = GptSoVitsFixture.Options();
        options.Version = version;
        using var model = new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1) { RandomSeed = 1337 }, options);
        var recording = new Tensor<float>(new[] { 6400 });
        for (int i = 0; i < recording.Length; i++) recording[i] = (float)(0.3 * Math.Sin(i * 0.13) + 0.1 * Math.Cos(i * 0.037));
        model.Voice = model.CreateVoice(recording, 16000, "hello world");
        var input = new Tensor<float>(new[] { 3 }, new Vector<float>(new[] { 1f, 2f, 3f }));
        model.SetTrainingMode(false);
        var expected = model.Predict(input);
        using var clone = (INeuralNetworkModel<float>)model.Clone();
        clone.SetTrainingMode(false);
        var actual = clone.Predict(input);
        Assert.NotEmpty(expected.ToArray());
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5 * (1 + Math.Abs(expected[i])));
        using var restored = new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(InputType.OneDimensional,
            NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options);
        restored.Deserialize(model.Serialize());
        restored.SetTrainingMode(false);
        var restoredOutput = restored.Predict(input);
        Assert.Equal(expected.Length, restoredOutput.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - restoredOutput[i]) <= 1e-5 * (1 + Math.Abs(expected[i])));
    }

    [Theory]
    [InlineData(GptSoVitsVersion.V3)]
    [InlineData(GptSoVitsVersion.V2)]
    public void EvenResidualKernelsAreRejectedBeforeInference(GptSoVitsVersion version)
    {
        var options = GptSoVitsFixture.Options();
        options.Version = version;
        options.ResblockKernels = options.VocoderResblockKernels = new[] { 4 };
        Assert.Throws<ArgumentException>(() => new GPTSoVITS<float>(new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 1, outputSize: 1), options));
    }
}
