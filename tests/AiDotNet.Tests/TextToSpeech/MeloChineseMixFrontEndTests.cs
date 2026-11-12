using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MeloTTS's mixed Chinese-English front end against MeloTTS's own chinese_mix.py
/// (tools/reference-data/melo_chinese_mix_reference.py): the normalized text, multilingual BERT's word pieces of the
/// English runs, the phones, the tones and the phones per character or piece of sentences mixing the two languages.
/// </summary>
public class MeloChineseMixFrontEndTests
{
    private static JsonElement Load()
    {
        const string name = "melo_chinese_mix_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    private static string Join(JsonElement array) => string.Join(" ", array.EnumerateArray().Select(v => v.ToString()));

    [Fact(Timeout = 120000)]
    public async Task NormalizationAndG2P_MatchMeloTTS()
    {
        await Task.Yield();
        var frontEnd = MeloChineseMixFrontEnd.Default;
        foreach (var s in Load().GetProperty("sentences").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            string normalized = frontEnd.Normalize(text);
            Assert.Equal(s.GetProperty("normalized").GetString(), normalized);
            var runs = Regex.Matches(normalized, @"[a-zA-Z\s]+").Cast<Match>().Select(m => m.Value).ToList();
            var expectedPieces = s.GetProperty("pieces").EnumerateArray().Select(Join).Where(p => p.Length > 0);
            Assert.Equal(expectedPieces, runs.Select(r => string.Join(" ", frontEnd.WordPieces(r))).Where(p => p.Length > 0));
            var (phones, tones, perUnit) = frontEnd.G2P(normalized);
            Assert.True(Join(s.GetProperty("phones")) == string.Join(" ", phones),
                $"{text}: phones\n expected {Join(s.GetProperty("phones"))}\n actual   {string.Join(" ", phones)}");
            Assert.True(Join(s.GetProperty("tones")) == string.Join(" ", tones),
                $"{text}: tones\n expected {Join(s.GetProperty("tones"))}\n actual   {string.Join(" ", tones)}");
            Assert.Equal(Join(s.GetProperty("word2ph")), string.Join(" ", perUnit));
        }
    }
}
