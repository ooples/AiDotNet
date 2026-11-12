using AiDotNet.TextToSpeech.Latest;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class KaniSamplingOptionValidationTests
{
    [Theory]
    [InlineData("Temperature", -1.0)]
    [InlineData("Temperature", double.NaN)]
    [InlineData("Temperature", double.PositiveInfinity)]
    [InlineData("TopP", 0.0)]
    [InlineData("TopP", 1.1)]
    [InlineData("TopP", double.NaN)]
    [InlineData("TopK", -1.0)]
    [InlineData("RepetitionPenalty", 0.0)]
    [InlineData("RepetitionPenalty", double.PositiveInfinity)]
    [InlineData("MaxNewTokens", 0.0)]
    [InlineData("StreamingChunkFrames", 0.0)]
    [InlineData("StreamingLookbackFrames", -1.0)]
    public void InvalidSamplingSettingNamesTheRejectedOption(string option, double value)
    {
        var options = new KaniTTSOptions();
        var property = typeof(KaniTTSOptions).GetProperty(option);
        Assert.NotNull(property);
        property.SetValue(options, Convert.ChangeType(value, property.PropertyType, System.Globalization.CultureInfo.InvariantCulture));
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => options.ValidateSampling());
        Assert.Equal(option, error.ParamName);
    }

    [Fact]
    public void SupportedGreedyAndUnrestrictedCutoffBoundariesRemainValid()
        => new KaniTTSOptions { Temperature = 0, TopK = 0, TopP = 1, RepetitionPenalty = 3,
            MaxNewTokens = 1, StreamingChunkFrames = 1, StreamingLookbackFrames = 0 }.ValidateSampling();

    [Fact]
    public void MutatedSamplingOptionsAreValidatedAtTheGenerationBoundary()
    {
        using var model = AiDotNet.Tests.TextToSpeech.KaniTTSFixture.Create<double>();
        var options = Assert.IsType<KaniTTSOptions>(model.GetOptions());
        options.TopP = 0;
        var error = Assert.Throws<ArgumentOutOfRangeException>(() => model.Synthesize("Hi."));
        Assert.Equal(nameof(KaniTTSOptions.TopP), error.ParamName);
    }
}
