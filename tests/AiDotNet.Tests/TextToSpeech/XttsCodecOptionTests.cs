using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public class XttsCodecOptionTests
{
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
        control.SetTrainingMode(false);
        perturbed.SetTrainingMode(false);
        long baseline = XttsOptionFixture.ObserveCodec(control);
        if (option == "DvaeHopSize")
            Assert.NotEqual(baseline, XttsOptionFixture.ObserveCodec(perturbed));
        else
            Assert.Throws<ArgumentOutOfRangeException>(() => XttsOptionFixture.ObserveCodec(perturbed));
    }
}
