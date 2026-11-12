using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.CodecBased;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Fish Speech's text front end against the releases' own code (tools/reference-data/fishspeech_text_reference.py):
/// clean_text and split_text of 1.4.3 and 1.5.1, and 1.4's Chinese number normalizer (ChnNormedText).
/// </summary>
public class FishSpeechTextTests
{
    private static JObject Fixture()
    {
        const string fileName = "fishspeech_text_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    [Theory]
    [InlineData("v14")]
    [InlineData("v15")]
    public void CleanAndSplit_MatchTheRelease(string release)
    {
        var f = Fixture();
        var version = release == "v15" ? FishSpeechVersion.V15 : FishSpeechVersion.V14;
        var texts = f["texts"]!.Select(t => (string)t!).ToArray();
        var expected = f[release]!;
        for (int i = 0; i < texts.Length; i++)
            Assert.True((string)expected["clean"]![i]! == FishSpeechText.Clean(texts[i], version), $"clean_text of text {i}");
        foreach (var length in f["lengths"]!.Select(n => (int)n))
            for (int i = 0; i < texts.Length; i++)
            {
                var want = expected["split"]![length.ToString()]![i]!.Select(s => (string)s!).ToArray();
                var got = FishSpeechText.Split(texts[i], length, version).ToArray();
                Assert.True(want.SequenceEqual(got), $"split_text of text {i} at {length}: [{string.Join(" | ", want)}] vs [{string.Join(" | ", got)}]");
            }
    }

    [Fact]
    public void ChineseNormalizer_MatchesTheRelease()
    {
        var f = Fixture();
        var texts = f["texts"]!.Select(t => (string)t!).ToArray();
        for (int i = 0; i < texts.Length; i++)
            Assert.Equal((string)f["v14"]!["normalized"]![i]!, ChineseNswNormalizer.Normalize(texts[i], fishSpeech: true));
    }
}
