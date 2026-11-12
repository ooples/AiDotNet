using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MeloTTS's Korean front end against MeloTTS's own korean.py over g2pkk 0.1.2 and python-mecab-ko
/// (tools/reference-data/melo_korean_reference.py): normalized text, bert-kor-base word pieces, phones, tones and phones
/// per piece of sentences; g2pkk on idioms, liaison, tensification, English, numbers and bound nouns; process_num.
/// </summary>
public class MeloKoreanFrontEndTests
{
    private static JsonElement Load()
    {
        const string name = "melo_korean_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static string Join(JsonElement array) => string.Join(" ", array.EnumerateArray().Select(v => v.ToString()));

    [Fact(Timeout = 300000)]
    public async Task Sentences_MatchMeloTTS()
    {
        await Task.Yield();
        var frontEnd = MeloKoreanFrontEnd.Default;
        foreach (var s in Load().GetProperty("sentences").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            string normalized = frontEnd.Normalize(text);
            Assert.Equal(s.GetProperty("normalized").GetString(), normalized);
            Assert.Equal(Join(s.GetProperty("pieces")), string.Join(" ", frontEnd.WordPieces(normalized)));
            var (phones, tones, perPiece) = frontEnd.G2P(normalized);
            Assert.True(Join(s.GetProperty("phones")) == string.Join(" ", phones),
                $"{text}: phones\n expected {Join(s.GetProperty("phones"))}\n actual   {string.Join(" ", phones)}");
            Assert.Equal(Join(s.GetProperty("tones")), string.Join(" ", tones));
            Assert.Equal(Join(s.GetProperty("word2ph")), string.Join(" ", perPiece));
        }
    }

    [Fact(Timeout = 300000)]
    public async Task G2pKkAndNumbers_MatchTheReference()
    {
        await Task.Yield();
        var reference = Load();
        var mismatches = new List<string>();
        foreach (var g in reference.GetProperty("g2pkk").EnumerateArray())
        {
            string text = g.GetProperty("text").GetString()!, expected = g.GetProperty("output").GetString()!;
            string actual = G2pKk.Default.Convert(text);
            if (expected != actual) mismatches.Add($"g2pkk {text}: {expected} vs {actual}");
        }
        foreach (var n in reference.GetProperty("numbers").EnumerateArray())
        {
            string num = n.GetProperty("num").GetString()!, expected = n.GetProperty("output").GetString()!;
            string actual = G2pKk.ProcessNumber(num, n.GetProperty("sino").GetBoolean());
            if (expected != actual) mismatches.Add($"process_num {num}: {expected} vs {actual}");
        }
        Assert.True(mismatches.Count == 0, string.Join(Environment.NewLine, mismatches));
    }
}
