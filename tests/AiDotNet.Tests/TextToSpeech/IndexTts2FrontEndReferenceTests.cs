using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.FrontEnd;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// IndexTTS2's text front end against the release's <c>front.py</c> on Linux
/// (tools/reference-data/indextts2_frontend_reference.py): normalization, BPE pieces and ids, and segments.
/// </summary>
/// <remarks>The BPE model is IndexTeam/IndexTTS-2's <c>bpe.model</c>; set <c>INDEXTTS2_DIR</c> to the release directory.
/// Without it only the normalization is checked.</remarks>
public class IndexTts2FrontEndReferenceTests
{
    private static JObject Fixture()
    {
        const string fileName = "indextts2_frontend_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    [Fact]
    public void Normalization_MatchesRelease()
    {
        var normalizer = new IndexTtsTextNormalizer();
        foreach (var c in Fixture()["cases"]!)
        {
            if (c["normalized"]!.Type == JTokenType.Null) continue;
            Assert.Equal((string)c["normalized"]!, normalizer.Normalize((string)c["text"]!));
        }
    }

    [Fact]
    public void Tokenization_AndSegments_MatchRelease()
    {
        string? dir = Environment.GetEnvironmentVariable("INDEXTTS2_DIR");
        string model = dir is null ? "" : Path.Combine(dir, "bpe.model");
        if (!File.Exists(model)) return;
        var tokenizer = new IndexTtsTokenizer(new SentencePieceModel(File.ReadAllBytes(model)), new IndexTtsTextNormalizer());
        foreach (var c in Fixture()["cases"]!)
        {
            string text = (string)c["text"]!;
            var pieces = tokenizer.Tokenize(text);
            Assert.Equal(c["pieces"]!.Select(v => (string)v!).ToList(), pieces);
            Assert.Equal(c["ids"]!.Select(v => (int)v).ToArray(), tokenizer.ToIds(pieces));
            foreach (var (key, max) in new[] { ("segments", 120), ("segments12", 12) })
            {
                var expected = c[key]!.Select(s => s.Select(v => (string)v!).ToList()).ToList();
                var actual = IndexTtsTokenizer.SplitSegments(pieces, max);
                Assert.Equal(expected.Count, actual.Count);
                for (int i = 0; i < expected.Count; i++) Assert.Equal(expected[i], actual[i]);
            }
        }
    }

    [Fact]
    public void SegmentSplit_IsPureAndKeepsTheReleaseQuirks()
    {
        // A quote after a sentence mark joins that segment and also starts the next (the release's no-op "i += 1").
        var tokens = new[] { "▁A", "▁B", "▁C", ".", "'", "▁D", "▁E", "▁F", "." };
        var segments = IndexTtsTokenizer.SplitSegments(tokens, 4);
        Assert.Equal(new[] { "▁A", "▁B", "▁C", ".", "'" }, segments[0]);
        Assert.Equal("'", segments[1][0]);
    }
}
