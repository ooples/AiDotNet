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
/// GPT-SoVITS's Korean and Cantonese front ends against GPT-SoVITS's own korean.py and cantonese.py
/// (tools/reference-data/gptsovits_korean_cantonese_reference.py): g2pk2 with mecab-ko, jamo and the phone mapping;
/// zh_normalization, ToJyutping and the Yue phones.
/// </summary>
public class GptSoVitsKoreanCantoneseFrontEndTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "gptsovits_korean_cantonese_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static string Joined(JsonElement array) => string.Join(" ", array.EnumerateArray().Select(v => v.GetString()));

    [Fact(Timeout = 120000)]
    public async Task Korean_MatchesKoreanPy()
    {
        await Task.Yield();
        var failures = new List<string>();
        foreach (var s in Reference.Value.GetProperty("korean").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            string actual = string.Join(" ", GptSoVitsKoreanFrontEnd.G2P(text));
            if (actual != Joined(s.GetProperty("phones"))) failures.Add($"{text}:\n expected {Joined(s.GetProperty("phones"))}\n actual   {actual}");
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact(Timeout = 120000)]
    public async Task Cantonese_MatchesCantonesePy()
    {
        await Task.Yield();
        var failures = new List<string>();
        foreach (var s in Reference.Value.GetProperty("cantonese").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            string normalized = GptSoVitsCantoneseFrontEnd.TextNormalize(text);
            if (normalized != s.GetProperty("normalized").GetString())
            {
                failures.Add($"{text}: normalized {s.GetProperty("normalized").GetString()} vs {normalized}");
                continue;
            }
            var (phones, word2ph) = GptSoVitsCantoneseFrontEnd.G2P(normalized);
            if (string.Join(" ", phones) != Joined(s.GetProperty("phones")))
                failures.Add($"{text}:\n expected {Joined(s.GetProperty("phones"))}\n actual   {string.Join(" ", phones)}");
            else Assert.Equal(s.GetProperty("word2ph").EnumerateArray().Select(v => v.GetInt32()), word2ph);
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }
}
