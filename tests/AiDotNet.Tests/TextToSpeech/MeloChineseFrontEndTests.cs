using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MeloTTS's Chinese front end against MeloTTS's own chinese.py (tools/reference-data/melo_chinese_reference.py): jieba's
/// part-of-speech cut, the normalized text, the phones, the tones and the phones per character of sentences with numbers,
/// 一/不 sandhi, third-tone sandhi, neutral tones, 儿, reduplication, Latin text and punctuation.
/// </summary>
public class MeloChineseFrontEndTests
{
    private static JsonElement Load()
    {
        const string name = "melo_chinese_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static string Strings(JsonElement array) => string.Join(" ", array.EnumerateArray().Select(v => v.GetString()));

    [Fact(Timeout = 120000)]
    public async Task NormalizationAndG2P_MatchMeloTTS()
    {
        await Task.Yield();
        var frontEnd = MeloChineseFrontEnd.Default;
        foreach (var s in Load().GetProperty("sentences").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            string normalized = frontEnd.Normalize(text);
            Assert.Equal(s.GetProperty("normalized").GetString(), normalized);
            if (s.TryGetProperty("fails", out _))
            {
                Assert.Throws<ArgumentException>(() => frontEnd.G2P(normalized));
                continue;
            }
            var (phones, tones, perCharacter) = frontEnd.G2P(normalized);
            Assert.True(Strings(s.GetProperty("phones")) == string.Join(" ", phones),
                $"{text}: phones\n expected {Strings(s.GetProperty("phones"))}\n actual   {string.Join(" ", phones)}");
            Assert.True(s.GetProperty("tones").EnumerateArray().Select(v => v.GetInt32()).SequenceEqual(tones),
                $"{text}: tones\n expected {string.Join(" ", s.GetProperty("tones").EnumerateArray())}\n actual   {string.Join(" ", tones)}");
            Assert.Equal(s.GetProperty("word2ph").EnumerateArray().Select(v => v.GetInt32()), perCharacter);
        }
    }

    [Fact(Timeout = 120000)]
    public async Task DictionaryWords_MatchPypinyinAndTheToneSandhi()
    {
        await Task.Yield();
        var mismatches = new System.Collections.Generic.List<string>();
        foreach (var w in Load().GetProperty("words").EnumerateArray())
        {
            string word = w.GetProperty("word").GetString()!, pos = w.GetProperty("pos").GetString()!;
            if (w.TryGetProperty("fails", out _))
            {
                Assert.ThrowsAny<SystemException>(() => MeloChineseFrontEnd.Default.Readings(word, pos));
                continue;
            }
            var (initials, finals) = MeloChineseFrontEnd.Default.Readings(word, pos);
            string expected = Strings(w.GetProperty("initials")) + " | " + Strings(w.GetProperty("finals"));
            string actual = string.Join(" ", initials) + " | " + string.Join(" ", finals);
            if (expected != actual) mismatches.Add($"{word}/{pos}: {expected} vs {actual}");
        }
        Assert.True(mismatches.Count == 0, string.Join(Environment.NewLine, mismatches));
    }

    [Fact(Timeout = 120000)]
    public async Task JiebaPosCut_MatchesJiebaPosseg()
    {
        await Task.Yield();
        foreach (var s in Load().GetProperty("sentences").EnumerateArray())
        {
            if (!s.TryGetProperty("cuts", out var cuts)) continue;
            var sentences = cuts.EnumerateArray().ToList();
            var pieces = sentences.Select(c => string.Concat(c.EnumerateArray().Select(p => p[0].GetString()))).ToList();
            for (int i = 0; i < sentences.Count; i++)
            {
                string expected = string.Join(" ", sentences[i].EnumerateArray().Select(p => $"{p[0].GetString()}/{p[1].GetString()}"));
                string actual = string.Join(" ", Jieba.Default.PosCut(pieces[i]).Select(p => $"{p.Word}/{p.Tag}"));
                Assert.Equal(expected, actual);
            }
        }
    }
}
