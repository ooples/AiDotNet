using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.TextToSpeech.MultiModal;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class SpeechT5OptionContextTests
{
    [Theory]
    [InlineData("PretrainEpsilon")]
    [InlineData("MaskId")]
    [InlineData("PretrainUpdates")]
    [InlineData("FeaturePenaltyWeight")]
    [InlineData("PretrainWeightDecay")]
    [InlineData("BosId")]
    [InlineData("EosId")]
    [InlineData("PretrainLearningRate")]
    [InlineData("DiversityWeight")]
    [InlineData("TextRandomRatio")]
    [InlineData("PretrainMaxGradientNorm")]
    [InlineData("CodebookProbability")]
    [InlineData("MaskProbability")]
    [InlineData("PretrainWarmupFraction")]
    [InlineData("TextMaskRatio")]
    [InlineData("MaxSpeechSamples")]
    [InlineData("PoissonLambda")]
    [InlineData("LabelRate")]
    [InlineData("MaskLength")]
    [InlineData("StopCriterion")]
    [InlineData("MaxLengthRatio")]
    [InlineData("MaxTextLength")]
    [InlineData("MaxMelLength")]
    public void NativeTaskContextExercisesTheConfiguredControl(string option)
    {
        var probe = new OptionSensitivityProbe<SpeechT5<double>>(
            () => SpeechT5Fixture.Create<double>(SpeechT5Fixture.Options()),
            model => SpeechT5Fixture.Observe(model, option)
                ?? OptionSensitivityHarness.Fingerprint(model.GetParameters()),
            model =>
            {
                for (int i = 0; i < OptionSensitivityFixtureSetup.TrainingSteps(model, option, 1); i++)
                    Assert.True(double.IsFinite(model.Train(SpeechT5Fixture.Sample<double>(option))));
            },
            prepareOptions: settings => SpeechT5Fixture.PrepareOptions((SpeechT5Options)settings, option),
            controlGroup: option);
        var report = OptionSensitivityHarness.Measure(typeof(SpeechT5OptionContextTests), option, probe);
        Assert.True(report.Outcome == OptionSensitivityOutcome.Sensitive, report.Detail);
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    public void PairedVocoderRatesAndHopProduceTheExpectedNativeWaveform(int hop)
    {
        var options = SpeechT5Fixture.Options();
        options.HopSize = hop;
        options.VocoderUpsampleRates = new[] { 4, 4, 4, hop / 64 };
        options.VocoderUpsampleKernels = options.VocoderUpsampleRates.Select(r => r * 2).ToArray();
        using var model = SpeechT5Fixture.Create<double>(options);
        model.SetTrainingMode(false);
        var mel = new Tensor<double>(new[] { 3, options.MelChannels });
        for (int i = 0; i < mel.Length; i++) mel[i] = .1 * Math.Sin(i * .17);
        var wave = model.Vocode(mel);
        Assert.Equal(3 * hop, wave.Length);
        Assert.All(wave, value => Assert.True(double.IsFinite(value)));
        Assert.Contains(wave, value => value != 0);
    }

    [Fact]
    public void UnpairedVocoderHopIsRejectedBeforeInference()
    {
        var options = SpeechT5Fixture.Options();
        options.HopSize = 512;
        Assert.Throws<ArgumentException>(() => SpeechT5Fixture.Create<double>(options));
    }
}
