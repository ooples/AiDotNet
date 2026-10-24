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
/// MeloTTS's Spanish and French front ends against MeloTTS's own spanish.py and french.py over gruut 2.2.3
/// (tools/reference-data/melo_romance_reference.py): normalized text, word pieces, phones, tones and phones per piece of
/// sentences; and the phonemizer on single words (numbers, years, currency, compact dates, liaison, punctuation), a
/// spread of eSpeak lexicon words and unknown words the CRF guesses.
/// </summary>
public class MeloRomanceFrontEndTests
{
    private static JsonElement Load()
    {
        const string name = "melo_romance_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static string Join(JsonElement array) => string.Join(" ", array.EnumerateArray().Select(v => v.ToString()));

    private static MeloRomanceFrontEnd FrontEnd(string lang) =>
        lang == "es" ? MeloSpanishFrontEnd.Default : MeloFrenchFrontEnd.Default;

    [Theory(Timeout = 300000)]
    [InlineData("es")]
    [InlineData("fr")]
    public async Task Sentences_MatchMeloTTS(string lang)
    {
        await Task.Yield();
        var frontEnd = FrontEnd(lang);
        foreach (var s in Load().GetProperty(lang).GetProperty("sentences").EnumerateArray())
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

    [Theory(Timeout = 600000)]
    [InlineData("es", "words")]
    [InlineData("es", "dictionary")]
    [InlineData("es", "unknown")]
    [InlineData("fr", "words")]
    [InlineData("fr", "dictionary")]
    [InlineData("fr", "unknown")]
    public async Task Words_MatchGruut(string lang, string set)
    {
        await Task.Yield();
        var frontEnd = FrontEnd(lang);
        var mismatches = new List<string>();
        foreach (var w in Load().GetProperty(lang).GetProperty(set).EnumerateArray())
        {
            string word = w.GetProperty("word").GetString()!;
            if (w.TryGetProperty("fails", out var error))
            {
                bool threw = false;
                try { frontEnd.WordToIpa(word); }
                catch (Exception) { threw = true; }
                if (!threw) mismatches.Add($"{word}: MeloTTS raises {error.GetString()}, the port does not");
                continue;
            }
            string expected = w.GetProperty("ipa").GetString()!;
            string actual = frontEnd.WordToIpa(word);
            if (expected != actual) mismatches.Add($"{word}: {expected} vs {actual}");
        }
        Assert.True(mismatches.Count == 0, string.Join(Environment.NewLine, mismatches));
    }
}
