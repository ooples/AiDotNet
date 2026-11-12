using System.Text.Json;
using AiDotNet.TextToSpeech.VoiceCloning;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

public sealed class XttsTextReferenceTests
{
    [Fact]
    public void Hindi_MatchesTheReleasedBasicCleanerWithoutInventingNumberExpansions()
    {
        Assert.Contains("hi", XTTSv2Options.ReleasedLanguages);
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "xtts_hindi_reference.json")));
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
            Assert.Equal(item.GetProperty("expected").GetString(), XTTSv2<double>.NormalizeText(
                item.GetProperty("text").GetString()!, "hi"));
    }

    [Fact]
    public void NumberBoundaries_MatchAuthorCleanerAcrossThirteenLanguages()
    {
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "xtts_number_boundaries.json")));
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            string language = item.GetProperty("language").GetString()!;
            string text = item.GetProperty("text").GetString()!;
            string expected = item.GetProperty("expected").GetString()!;
            string actual = XTTSv2<double>.NormalizeText(text, language);
            Assert.True(expected == actual, $"{language}: {text}\nAuthor: {expected}\nNative: {actual}");
        }
    }

    [Fact]
    public void TurkishCardinalsAndOrdinals_MatchEveryTripletAndTargetedGroupBoundaries()
    {
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "xtts_turkish_reference.json")));
        var reader = AiDotNet.TextToSpeech.FrontEnd.Num2WordsTurkish.Default;
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
        {
            var value = System.Numerics.BigInteger.Parse(item.GetProperty("value").GetString()!, System.Globalization.CultureInfo.InvariantCulture);
            Assert.Equal(item.GetProperty("cardinal").GetString(), reader.ToCardinal(value));
            Assert.Equal(item.GetProperty("ordinal").GetString(), reader.ToOrdinal(value));
        }
    }

    [Fact]
    public void KoreanAcademicRomanization_MatchesEveryConsonantBoundaryAndEveryHangulSyllable()
    {
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "hangul_academic_reference.json")));
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
            Assert.Equal(item.GetProperty("expected").GetString(), AiDotNet.TextToSpeech.FrontEnd.HangulAcademicRomanizer.Romanize(
                item.GetProperty("text").GetString()!));
    }

    [Fact]
    public void LatinLanguages_MatchReleasedNum2WordsCurrencyOrdinalsAndWhitespace()
    {
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "xtts_latin_reference.json")));
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
            Assert.Equal(item.GetProperty("expected").GetString(), XTTSv2<double>.NormalizeText(
                item.GetProperty("text").GetString()!, item.GetProperty("language").GetString()!));
    }

    [Fact]
    public void Chinese_MatchesCoquiNumberNormalizationAndFullToneThreeSyllables()
    {
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "xtts_chinese_reference.json")));
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
            Assert.Equal(item.GetProperty("expected").GetString(), XTTSv2<double>.NormalizeText(item.GetProperty("text").GetString()!, "zh-cn"));
    }

    [Fact]
    public void Japanese_MatchesUpstreamCutletOverUnidicLite()
    {
        using var reference = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory,
            "TextToSpeech", "ReferenceData", "xtts_cutlet_reference.json")));
        foreach (var item in reference.RootElement.GetProperty("cases").EnumerateArray())
            Assert.Equal(item.GetProperty("expected").GetString(), XTTSv2<double>.NormalizeText(item.GetProperty("text").GetString()!, "ja"));
    }

    [Fact]
    public void English_NormalizesAbbreviationsAndNumbersBeforeBpe()
    {
        Assert.Equal("doctor smith has twelve apples.", XTTSv2<double>.NormalizeText("Dr. Smith has 12 apples.", "en"));
        Assert.Equal("mister jones is twenty-first.", XTTSv2<double>.NormalizeText("Mr. Jones is 21st.", "en"));
    }
}
