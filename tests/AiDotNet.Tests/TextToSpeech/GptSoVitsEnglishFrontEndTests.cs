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
/// GPT-SoVITS's English front end against GPT-SoVITS's own english.py (tools/reference-data/
/// gptsovits_english_reference.py): text_normalize, expend.normalize, TweetTokenizer with NLTK's tagger, wordsegment and
/// the phones of sentences and of a spread of words.
/// </summary>
public class GptSoVitsEnglishFrontEndTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "gptsovits_english_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static List<string> Strings(JsonElement array) => array.EnumerateArray().Select(v => v.GetString()!).ToList();

    [Fact(Timeout = 120000)]
    public async Task Normalize_MatchesExpend()
    {
        await Task.Yield();
        var failures = new List<string>();
        foreach (var c in Reference.Value.GetProperty("numbers").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            if (c.TryGetProperty("fails", out _))
            {
                Assert.ThrowsAny<FormatException>(() => GptSoVitsEnglishFrontEnd.Expand(text));
                continue;
            }
            string actual = GptSoVitsEnglishFrontEnd.Expand(text);
            if (actual != c.GetProperty("normalized").GetString()) failures.Add($"'{text}': '{c.GetProperty("normalized").GetString()}' vs '{actual}'");
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact(Timeout = 120000)]
    public async Task SentencesAndWords_MatchEnglishG2P()
    {
        await Task.Yield();
        var frontEnd = GptSoVitsEnglishFrontEnd.Default;
        var failures = new List<string>();
        foreach (var s in Reference.Value.GetProperty("sentences").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            string normalized = frontEnd.TextNormalize(text);
            if (normalized != s.GetProperty("normalized").GetString())
            {
                failures.Add($"{text}: normalized '{s.GetProperty("normalized").GetString()}' vs '{normalized}'");
                continue;
            }
            var tokens = GptSoVitsEnglishFrontEnd.TweetTokenize(normalized);
            if (!Strings(s.GetProperty("tokens")).SequenceEqual(tokens))
                failures.Add($"{text}: tokens {string.Join("|", Strings(s.GetProperty("tokens")))} vs {string.Join("|", tokens)}");
            string expected = string.Join(" ", Strings(s.GetProperty("phones")));
            string actual = string.Join(" ", frontEnd.G2P(normalized));
            if (expected != actual) failures.Add($"{text}:\n expected {expected}\n actual   {actual}");
        }
        foreach (var w in Reference.Value.GetProperty("words").EnumerateArray())
        {
            string word = w.GetProperty("word").GetString()!;
            string expected = string.Join(" ", Strings(w.GetProperty("phones")));
            string actual = string.Join(" ", frontEnd.G2P(word));
            if (expected != actual) failures.Add($"{word}: {expected} vs {actual}");
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact(Timeout = 120000)]
    public async Task WordSegment_MatchesWordsegment()
    {
        await Task.Yield();
        foreach (var c in Reference.Value.GetProperty("segments").EnumerateArray())
            Assert.Equal(Strings(c.GetProperty("words")), WordSegment.Default.Segment(c.GetProperty("text").GetString()!));
    }
}
