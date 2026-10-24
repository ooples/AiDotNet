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
/// MeloTTS's Japanese front end against MeloTTS's own japanese.py (tools/reference-data/melo_japanese_reference.py):
/// normalized text, bert-base-japanese-v3 word pieces, phones, tones and phones per piece of sentences; pykakasi's
/// katakana for kanji, kana and long-vowel marks plus 400 kanwa entries; the number conversion; kata2phoneme.
/// </summary>
public class MeloJapaneseFrontEndTests
{
    private static JsonElement Load()
    {
        const string name = "melo_japanese_reference.json";
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
        var frontEnd = MeloJapaneseFrontEnd.Default;
        foreach (var s in Load().GetProperty("sentences").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            string normalized = frontEnd.Normalize(text);
            Assert.Equal(s.GetProperty("normalized").GetString(), normalized);
            Assert.Equal(Join(s.GetProperty("pieces")), string.Join(" ", frontEnd.WordPieces(normalized)));
            if (s.TryGetProperty("fails", out _))
            {
                Assert.Throws<ArgumentException>(() => frontEnd.G2P(normalized));
                continue;
            }
            var (phones, tones, perPiece) = frontEnd.G2P(normalized);
            Assert.True(Join(s.GetProperty("phones")) == string.Join(" ", phones),
                $"{text}: phones\n expected {Join(s.GetProperty("phones"))}\n actual   {string.Join(" ", phones)}");
            Assert.Equal(Join(s.GetProperty("tones")), string.Join(" ", tones));
            Assert.Equal(Join(s.GetProperty("word2ph")), string.Join(" ", perPiece));
        }
    }

    [Fact(Timeout = 300000)]
    public async Task Pykakasi_NumbersAndKana_MatchTheReference()
    {
        await Task.Yield();
        var reference = Load();
        var mismatches = new List<string>();
        foreach (var k in reference.GetProperty("kakasi").EnumerateArray())
        {
            string text = k.GetProperty("text").GetString()!, expected = k.GetProperty("kana").GetString()!;
            string actual = PykakasiKatakana.Default.Convert(text);
            if (expected != actual) mismatches.Add($"kakasi {text}: {expected} vs {actual}");
        }
        foreach (var n in reference.GetProperty("numbers").EnumerateArray())
        {
            string text = n.GetProperty("text").GetString()!, expected = n.GetProperty("words").GetString()!;
            string actual = MeloJapaneseFrontEnd.ConvertNumbers(text);
            if (expected != actual) mismatches.Add($"number {text}: {expected} vs {actual}");
        }
        foreach (var k in reference.GetProperty("katakana").EnumerateArray())
        {
            string text = k.GetProperty("text").GetString()!, expected = Join(k.GetProperty("phonemes"));
            string actual = string.Join(" ", MeloJapaneseFrontEnd.Default.KatakanaToPhonemes(text));
            if (expected != actual) mismatches.Add($"kata2phoneme {text}: {expected} vs {actual}");
        }
        Assert.True(mismatches.Count == 0, string.Join(Environment.NewLine, mismatches));
    }
}
