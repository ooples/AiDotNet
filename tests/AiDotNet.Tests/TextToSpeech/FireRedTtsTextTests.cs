using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// FireRedTTS's text front end against the release's own code (tools/reference-data/fireredtts_text_reference.py):
/// preprocess_text, rettt, lingua's English/Chinese detection, TextNormalizer.tn, the tokenizer's ids, text_split and
/// sentencex's segmentation.
/// </summary>
public class FireRedTtsTextTests
{
    private static JObject Fixture()
    {
        const string fileName = "fireredtts_text_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    private static string Show(string s) => string.Concat(s.Select(c => c < 128 ? c.ToString() : $"\\u{(int)c:X4}"));

    [Fact]
    public void Normalizer_MatchesTheRelease()
    {
        foreach (var c in Fixture()["cases"]!)
        {
            string text = (string)c["text"]!;
            string label = Show(text);
            string preprocessed = FireRedTtsTextNormalizer.Preprocess(text);
            Assert.True((string)c["preprocessed"]! == preprocessed, $"preprocess_text of '{label}': '{Show((string)c["preprocessed"]!)}' vs '{Show(preprocessed)}'");
            string abbreviated = FireRedTtsTextNormalizer.Abbreviations(preprocessed);
            Assert.True((string)c["rettt"]! == abbreviated, $"rettt of '{label}'");
            Assert.True((string?)c["detected"] == FireRedTtsLanguageDetector.Detect(abbreviated), $"detection of '{label}'");
            var (normalized, language) = FireRedTtsTextNormalizer.Normalize(PythonText.Strip(text));
            Assert.True((string)c["normalized"]! == normalized, $"tn of '{label}': '{Show((string)c["normalized"]!)}' vs '{Show(normalized)}'");
            Assert.True((string)c["language"]! == language, $"language of '{label}'");
        }
    }

    [Fact]
    public void Tokenizer_MatchesTheRelease()
    {
        var tokenizer = FireRedTtsTokenizer.Get(instructionTokens: false);
        Assert.Equal(57184, FireRedTtsTokenizer.RankCount);
        Assert.Equal(57184, tokenizer.Special("[STOP]"));
        Assert.Equal(57187, tokenizer.Special("[START]"));
        foreach (var c in Fixture()["cases"]!)
        {
            string text = (string)c["text"]!;
            var want = c["ids"]!.Select(v => (int)v).ToArray();
            var got = tokenizer.Encode(text);
            Assert.True(want.SequenceEqual(got), $"ids of '{Show(text)}': [{string.Join(",", want)}] vs [{string.Join(",", got)}]");
        }
    }

    [Fact]
    public void TextSplit_MatchesTheRelease()
    {
        foreach (var c in Fixture()["cases"]!)
        {
            string text = (string)c["text"]!;
            var want = c["split"]!.Select(v => (string)v!).ToArray();
            var got = FireRedTtsTextSplitter.Split(text, dropLastPiece: true);
            Assert.True(want.SequenceEqual(got), $"text_split of '{Show(text)}': [{string.Join(" | ", want.Select(Show))}] vs [{string.Join(" | ", got.Select(Show))}]");
        }
    }

    [Fact]
    public void TextSplit_KeepsTheLastPieceByDefault()
    {
        const string text = "The quick brown fox jumps over the lazy dog. It was a sunny day in the park. "
            + "Children were playing football happily. Then it started raining heavily over the whole town.";
        var release = FireRedTtsTextSplitter.Split(text, dropLastPiece: true);
        var fixedSplit = FireRedTtsTextSplitter.Split(text);
        Assert.Equal(release.Count + 1, fixedSplit.Count);
        Assert.Equal("Then it started raining heavily over the whole town.", fixedSplit[fixedSplit.Count - 1]);
        Assert.Equal(text.Replace(" ", ""), string.Concat(fixedSplit).Replace(" ", ""));
    }

    [Fact]
    public void LanguageDetection_MatchesLingua()
    {
        foreach (var p in Fixture()["probes"]!)
        {
            string text = (string)p["text"]!;
            Assert.True((string?)p["detected"] == FireRedTtsLanguageDetector.Detect(text), $"lingua on '{Show(text)}'");
        }
    }

    [Fact]
    public void Segmenter_MatchesSentencex()
    {
        var f = Fixture();
        foreach (var s in f["segments"]!.Concat(f["zh_segments"]!))
        {
            string text = (string)s["text"]!;
            var want = s["sentences"]!.Select(v => (string)v!).ToArray();
            var got = SentencexSegmenter.Segment(text);
            Assert.True(want.SequenceEqual(got), $"segment of '{Show(text)}': [{string.Join(" | ", want.Select(Show))}] vs [{string.Join(" | ", got.Select(Show))}]");
        }
    }
}
