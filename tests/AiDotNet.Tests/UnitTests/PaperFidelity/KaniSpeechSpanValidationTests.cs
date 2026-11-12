using AiDotNet.TextToSpeech.Latest;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class KaniSpeechSpanValidationTests
{
    [Theory]
    [InlineData("missing")]
    [InlineData("reversed")]
    [InlineData("incomplete")]
    [InlineData("empty")]
    [InlineData("non-audio")]
    public void InvalidGeneratedSpansHaveASpecificDecodingFailure(string kind)
    {
        var layout = new KaniLayout(10, 2, 2, 3);
        var tokens = kind switch
        {
            "missing" => new[] { 20, 23, 12 },
            "reversed" => new[] { 12, 11 },
            "incomplete" => new[] { 11, 20, 12 },
            "empty" => new[] { 11, 12 },
            "non-audio" => new[] { 11, 19, 23, 12 },
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };
        Assert.Throws<KaniSpeechSpanException>(() => layout.SpanCodes(tokens));
    }

    [Fact]
    public void CompleteSpeechFramesStillDecode()
    {
        var codes = new KaniLayout(10, 2, 2, 3).SpanCodes(new[] { 11, 20, 23, 12 });
        Assert.Equal(2, codes.GetLength(0));
        Assert.Equal(1, codes.GetLength(1));
        Assert.Equal(0, codes[0, 0]);
        Assert.Equal(0, codes[1, 0]);
    }
}
