using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.Latest;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MegaTTS 3's text chunking and segment joining against the release's own code
/// (tools/reference-data/megatts3_text_reference.py).
/// </summary>
public class MegaTts3TextReferenceTests
{
    private static JObject Fixture()
    {
        const string fileName = "megatts3_text_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    [Fact]
    public void Chunking_MatchesTheRelease()
    {
        var f = Fixture();
        foreach (var e in f["english"]!)
            Assert.Equal(e["chunks"]!.Select(c => (string)c!), MegaTts3Text.ChunkEnglish((string)e["text"]!, (int)e["max"]!));
        foreach (var c in f["chinese"]!)
            Assert.Equal(c["chunks"]!.Select(x => (string)x!), MegaTts3Text.ChunkChinese((string)c["text"]!, (int)c["limit"]!, (int)c["look"]!));
    }

    [Fact]
    public void Joining_MatchesTheRelease()
    {
        var f = Fixture();
        var segments = f["segments"]!.Select(s => s.Select(v => (double)v).ToArray()).ToArray();
        var joined = MegaTts3Text.Combine(segments, 0.02, 16000);
        var expected = f["joined"]!.Select(v => (double)v).ToArray();
        Assert.Equal(expected.Length, joined.Length);
        for (int i = 0; i < expected.Length; i++) Assert.True(Math.Abs(expected[i] - joined[i]) <= 1e-12, $"joined[{i}]: {expected[i]} vs {joined[i]}");
    }
}
