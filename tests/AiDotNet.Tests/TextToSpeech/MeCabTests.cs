using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The MeCab port with unidic-lite reproduces mecab-python3's Tagger("-Owakati") (tools/reference-data/
/// mecab_unidic_reference.py) on Japanese and Chinese text, full- and half-width characters, spaces, emoji, long
/// katakana runs and empty input.
/// </summary>
public class MeCabTests
{
    private static JObject Reference()
    {
        const string fileName = "mecab_unidic_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    [Fact(Timeout = 300000)]
    public async Task Wakati_MatchesMeCab()
    {
        await Task.Yield();
        var mismatches = Reference()["corpus"]!
            .Select(c => (Text: (string)c["text"]!, Expected: (string)c["wakati"]!))
            .Select(c => (c.Text, c.Expected, Actual: MeCab.UnidicLite.Wakati(c.Text)))
            .Where(c => c.Expected != c.Actual)
            .Select(c => $"{c.Text}\n  mecab {c.Expected.TrimEnd('\n')}\n  port  {c.Actual.TrimEnd('\n')}")
            .ToList();
        Assert.True(mismatches.Count == 0, $"{mismatches.Count} mismatches:\n" + string.Join("\n", mismatches));
    }
}
