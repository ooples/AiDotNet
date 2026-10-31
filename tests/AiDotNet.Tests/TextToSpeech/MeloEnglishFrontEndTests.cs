using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MeloTTS's English front end against MeloTTS's own english.py (tools/reference-data/melo_english_reference.py): the
/// normalized text, the BERT word pieces, the phones, the tones and the phones per piece of sentences with numbers, times,
/// currency, abbreviations, homographs, unknown words and punctuation.
/// </summary>
public class MeloEnglishFrontEndTests
{
    private static JsonElement Load()
    {
        const string name = "melo_english_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    [Fact(Timeout = 120000)]
    public async Task NormalizationAndG2P_MatchMeloTTS()
    {
        await Task.Yield();
        var frontEnd = MeloEnglishFrontEnd.Default;
        foreach (var s in Load().GetProperty("sentences").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            string normalized = frontEnd.Normalize(text);
            Assert.Equal(s.GetProperty("normalized").GetString(), normalized);
            Assert.Equal(s.GetProperty("pieces").EnumerateArray().Select(v => v.GetString()), frontEnd.WordPieces(normalized));
            var (phones, tones, perPiece) = frontEnd.G2P(normalized);
            Assert.True(s.GetProperty("phones").EnumerateArray().Select(v => v.GetString()!).SequenceEqual(phones),
                $"{text}: phones\n expected {string.Join(" ", s.GetProperty("phones").EnumerateArray().Select(v => v.GetString()))}\n actual   {string.Join(" ", phones)}");
            Assert.Equal(s.GetProperty("tones").EnumerateArray().Select(v => v.GetInt32()), tones);
            Assert.Equal(s.GetProperty("word2ph").EnumerateArray().Select(v => v.GetInt32()), perPiece);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task G2pEn_HomographsAndUnknownWords_MatchG2pEn()
    {
        await Task.Yield();
        int mismatches = 0;
        var messages = new System.Collections.Generic.List<string>();
        foreach (var w in Load().GetProperty("g2p_en").EnumerateArray())
        {
            string word = w.GetProperty("word").GetString()!;
            var expected = w.GetProperty("phones").EnumerateArray().Select(v => v.GetString()!).ToList();
            var actual = G2pEn.Default.Word(word);
            if (!expected.SequenceEqual(actual))
            {
                mismatches++;
                messages.Add($"{word}: {string.Join(" ", expected)} vs {string.Join(" ", actual)}");
            }
        }
        Assert.True(mismatches == 0, string.Join(Environment.NewLine, messages));
    }
}
