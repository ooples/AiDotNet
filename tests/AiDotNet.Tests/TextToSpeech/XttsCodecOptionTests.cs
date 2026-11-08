using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class XttsCodecOptionTests
{
    [Theory]
    [InlineData(false, "Temperature", 1.5)]
    [InlineData(false, "TopP", .5)]
    [InlineData(false, "RepetitionPenalty", 1)]
    [InlineData(true, "Temperature", 1.5)]
    [InlineData(true, "TopP", .5)]
    [InlineData(true, "RepetitionPenalty", 1)]
    public void NativeSamplingExercisesTemperatureNucleusAndRepetition(bool clone, string option, double value)
    {
        void Prepare(XTTSv2Options options) => XttsOptionFixture.PrepareOptions(options, option);
        using XTTSv2<double> control = clone ? XTTSv2Fixture.CreateClone<double>(Prepare) : XTTSv2Fixture.Create<double>(Prepare);
        using XTTSv2<double> perturbed = clone ? XTTSv2Fixture.CreateClone<double>(Prepare) : XTTSv2Fixture.Create<double>(Prepare);
        typeof(XTTSv2Options).GetProperty(option)!.SetValue(perturbed.GetOptions(), value);
        control.SetTrainingMode(true);
        perturbed.SetTrainingMode(true);
        Assert.NotEqual(XttsOptionFixture.ObserveSampler(control), XttsOptionFixture.ObserveSampler(perturbed));
        Assert.True(control.IsTrainingMode);
        Assert.True(perturbed.IsTrainingMode);
    }

    [Theory]
    [InlineData(false, "DvaeFftSize", 255)]
    [InlineData(false, "DvaeWindowSize", 257)]
    [InlineData(false, "DvaeHopSize", 65)]
    [InlineData(true, "DvaeFftSize", 255)]
    [InlineData(true, "DvaeWindowSize", 257)]
    [InlineData(true, "DvaeHopSize", 65)]
    public void NativeWaveformEncodingEnforcesSpectralOptions(bool clone, string option, int value)
    {
        using XTTSv2<double> control = clone ? XTTSv2Fixture.CreateClone<double>() : XTTSv2Fixture.Create<double>();
        using XTTSv2<double> perturbed = clone ? XTTSv2Fixture.CreateClone<double>() : XTTSv2Fixture.Create<double>();
        typeof(XTTSv2Options).GetProperty(option)!.SetValue(perturbed.GetOptions(), value);
        control.SetTrainingMode(true);
        perturbed.SetTrainingMode(true);
        long baseline = XttsOptionFixture.ObserveCodec(control);
        if (option == "DvaeHopSize")
            Assert.NotEqual(baseline, XttsOptionFixture.ObserveCodec(perturbed));
        else
            Assert.Throws<ArgumentOutOfRangeException>(() => XttsOptionFixture.ObserveCodec(perturbed));
        Assert.True(control.IsTrainingMode);
        Assert.True(perturbed.IsTrainingMode);
    }
}
