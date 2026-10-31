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
/// GPT-SoVITS's LangSegmenter against GPT-SoVITS's own (tools/reference-data/gptsovits_langsegmenter_reference.py):
/// BudouX's chunks, and where fastText's language model is present (<c>FASTTEXT_LID</c>: lid.176.bin) fast_langdetect's
/// detections and getTexts with each default language.
/// </summary>
public class GptSoVitsLangSegmenterTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "gptsovits_langsegmenter_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static List<string> Strings(JsonElement array) => array.EnumerateArray().Select(v => v.GetString()!).ToList();

    [Fact(Timeout = 120000)]
    public async Task Budoux_MatchesBudoux()
    {
        await Task.Yield();
        foreach (var c in Reference.Value.GetProperty("budoux").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            Assert.Equal(Strings(c.GetProperty("ja")), Budoux.Japanese.Parse(text));
            Assert.Equal(Strings(c.GetProperty("zh")), Budoux.SimplifiedChinese.Parse(text));
        }
    }

    [Fact(Timeout = 600000)]
    public async Task DetectionAndSegments_MatchGptSoVits_WhenTheModelIsPresent()
    {
        await Task.Yield();
        string? model = Environment.GetEnvironmentVariable("FASTTEXT_LID");
        if (model is null || !File.Exists(model)) return;
        var detector = new FastLangDetect(model);
        var failures = new List<string>();
        foreach (var d in Reference.Value.GetProperty("detections").EnumerateArray())
        {
            string text = d.GetProperty("text").GetString()!;
            foreach (var (key, k, threshold, input) in new[]
            {
                ("top", 1, 0f, text.Replace("\n", " ")),
                ("five", 5, 0.01f, AiDotNet.TextToSpeech.FrontEnd.PythonText.Strip(text.Replace("\n", ""))),
            })
            {
                var expected = d.GetProperty(key).EnumerateArray().Select(r => (r[0].GetString()!, r[1].GetDouble())).ToList();
                var actual = detector.Detect(input, k, threshold);
                bool same = expected.Count == actual.Count && expected.Zip(actual, (e, a) => e.Item1 == a.Lang && Math.Abs(e.Item2 - a.Score) < 1e-5).All(x => x);
                if (!same)
                    failures.Add($"{key} '{text}': {string.Join(",", expected.Select(e => $"{e.Item1}:{e.Item2:F6}"))} vs {string.Join(",", actual.Select(a => $"{a.Lang}:{a.Score:F6}"))}");
            }
        }
        var segmenter = new GptSoVitsLangSegmenter(model);
        foreach (var s in Reference.Value.GetProperty("segments").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            foreach (var (key, lang) in new[] { ("default_none", ""), ("default_zh", "zh"), ("default_ja", "ja"), ("default_ko", "ko") })
            {
                string expected = string.Join(" | ", s.GetProperty(key).EnumerateArray().Select(p => $"{p[0].GetString()}:{p[1].GetString()}"));
                string actual = string.Join(" | ", segmenter.GetTexts(text, lang).Select(p => $"{p.Lang}:{p.Text}"));
                if (expected != actual) failures.Add($"{key} '{text}':\n expected {expected}\n actual   {actual}");
            }
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }
}
