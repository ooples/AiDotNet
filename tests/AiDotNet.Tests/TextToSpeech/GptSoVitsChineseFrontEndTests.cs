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
/// GPT-SoVITS's Chinese front end against GPT-SoVITS's own text package (tools/reference-data/
/// gptsovits_chinese_reference.py): zh_normalization, text_normalize, v1's pypinyin G2P and v2's G2PW G2P (fed G2PW's
/// recorded readings; the model itself is checked by the gated test), jieba_fast's cuts, OpenCC s2tw and the tone sandhi
/// over jieba's dictionary words.
/// </summary>
public class GptSoVitsChineseFrontEndTests
{
    private static readonly Lazy<JsonElement> Reference = new(() =>
    {
        const string name = "gptsovits_chinese_reference.json";
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    });

    private static List<string> Strings(JsonElement array) => array.EnumerateArray().Select(v => v.GetString()!).ToList();

    private static List<List<string?>> Readings(JsonElement g2pw)
        => g2pw.EnumerateArray().Select(s => s.EnumerateArray().Select(v => v.ValueKind == JsonValueKind.Null ? null : v.GetString()).ToList()).ToList();

    [Fact(Timeout = 120000)]
    public async Task Normalizer_MatchesZhNormalization()
    {
        await Task.Yield();
        foreach (var c in Reference.Value.GetProperty("numbers").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            Assert.True(Strings(c.GetProperty("sentences")).SequenceEqual(GptSoVitsChineseNormalizer.Normalize(text)),
                $"{text}: expected {string.Join("|", Strings(c.GetProperty("sentences")))} actual {string.Join("|", GptSoVitsChineseNormalizer.Normalize(text))}");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task TextNormalizeAndG2P_MatchV1AndV2()
    {
        await Task.Yield();
        var frontEnd = GptSoVitsChineseFrontEnd.Default;
        var failures = new List<string>();
        foreach (var s in Reference.Value.GetProperty("sentences").EnumerateArray())
        {
            string text = s.GetProperty("text").GetString()!;
            string normalized = frontEnd.TextNormalize(text);
            if (normalized != s.GetProperty("v2_normalized").GetString())
            {
                failures.Add($"{text}: normalized {s.GetProperty("v2_normalized").GetString()} vs {normalized}");
                continue;
            }
            Assert.Equal(s.GetProperty("v1_normalized").GetString(), normalized);
            var recorded = Readings(s.GetProperty("g2pw"));
            foreach (var (version, v1) in new[] { ("v1", true), ("v2", false) })
            {
                var (phones, word2ph) = frontEnd.G2P(normalized, v1, batch =>
                {
                    Assert.Equal(Strings(s.GetProperty("segments")).Where(x => x.Length > 0), batch);
                    return recorded.Select(r => r.ToList()).ToList();
                });
                string expected = string.Join(" ", Strings(s.GetProperty($"{version}_phones")));
                if (expected != string.Join(" ", phones))
                    failures.Add($"{text} ({version}):\n expected {expected}\n actual   {string.Join(" ", phones)}");
                else
                    Assert.Equal(s.GetProperty($"{version}_word2ph").EnumerateArray().Select(v => v.GetInt32()), word2ph);
            }
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact(Timeout = 120000)]
    public async Task JiebaFast_MatchesItsCutsAndPosCuts()
    {
        await Task.Yield();
        var failures = new List<string>();
        foreach (var c in Reference.Value.GetProperty("jieba").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            string cut = string.Join("/", Jieba.Default.Cut(text, jiebaFast: true));
            string search = string.Join("/", Jieba.Default.CutForSearch(text, jiebaFast: true));
            if (cut != string.Join("/", Strings(c.GetProperty("cut")))) failures.Add($"cut {string.Join("/", Strings(c.GetProperty("cut")))} vs {cut}");
            if (search != string.Join("/", Strings(c.GetProperty("search")))) failures.Add($"search {string.Join("/", Strings(c.GetProperty("search")))} vs {search}");
        }
        foreach (var s in Reference.Value.GetProperty("sentences").EnumerateArray())
        {
            var segments = Strings(s.GetProperty("segments"));
            var cuts = s.GetProperty("cuts").EnumerateArray().ToList();
            for (int i = 0; i < segments.Count; i++)
            {
                string expected = string.Join(" ", cuts[i].EnumerateArray().Select(p => $"{p[0].GetString()}/{p[1].GetString()}"));
                string actual = string.Join(" ", Jieba.Default.PosCut(segments[i]).Select(p => $"{p.Word}/{p.Tag}"));
                if (expected != actual) failures.Add($"pos {expected} vs {actual}");
            }
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Fact(Timeout = 120000)]
    public async Task OpenCcS2Tw_MatchesOpenCC()
    {
        await Task.Yield();
        foreach (var c in Reference.Value.GetProperty("opencc").EnumerateArray())
            Assert.Equal(c.GetProperty("s2tw").GetString(), OpenCc.S2Tw.Convert(c.GetProperty("text").GetString()!));
    }

    [Fact(Timeout = 120000)]
    public async Task DictionaryWords_MatchPypinyinTheToneSandhiAndCutForSearch()
    {
        await Task.Yield();
        var frontEnd = GptSoVitsChineseFrontEnd.Default;
        var failures = new List<string>();
        foreach (var w in Reference.Value.GetProperty("words").EnumerateArray())
        {
            string word = w.GetProperty("word").GetString()!, pos = w.GetProperty("pos").GetString()!;
            string search = string.Join("/", Jieba.Default.CutForSearch(word, jiebaFast: true));
            if (search != string.Join("/", Strings(w.GetProperty("search")))) failures.Add($"{word} search {string.Join("/", Strings(w.GetProperty("search")))} vs {search}");
            var readings = GptSoVitsPinyin.Default.Read(word);
            if (!Strings(w.GetProperty("initials")).SequenceEqual(readings.Select(r => r.Initial)))
                failures.Add($"{word} initials {string.Join(" ", Strings(w.GetProperty("initials")))} vs {string.Join(" ", readings.Select(r => r.Initial))}");
            if (w.TryGetProperty("fails", out _))
            {
                Assert.ThrowsAny<SystemException>(() => frontEnd.ModifiedTone(word, pos, readings.Select(r => r.Final).ToList()));
                continue;
            }
            var finals = frontEnd.ModifiedTone(word, pos, readings.Select(r => r.Final).ToList());
            if (!Strings(w.GetProperty("finals")).SequenceEqual(finals))
                failures.Add($"{word}/{pos} finals {string.Join(" ", Strings(w.GetProperty("finals")))} vs {string.Join(" ", finals)}");
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    /// <summary>
    /// The released G2PW model (<c>G2PW_DIR</c>: GPT-SoVITS's G2PWModel_1.1 with g2pW.onnx) where present: its
    /// probabilities for the first queries against onnxruntime's, and the converter's readings of every sentence
    /// against GPT-SoVITS's.
    /// </summary>
    [Fact(Timeout = 3600000)]
    public async Task ReleasedG2pw_MatchesGptSoVits_WhenPresent()
    {
        await Task.Yield();
        string? dir = Environment.GetEnvironmentVariable("G2PW_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "g2pW.onnx"))) return;
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var model = new G2pwModel<double>(engine, new List<AiDotNet.NeuralNetworks.Layers.LayerBase<double>>(), new G2pwConfiguration());
        var weights = new AiDotNet.ComputerVision.Weights.WeightLoader().LoadWeights(Path.Combine(dir, "g2pW.onnx"));
        model.Load((name, shape) =>
        {
            var tensor = weights[name];
            Assert.Equal(shape.Aggregate(1, (a, b) => a * b), tensor.Length);
            var values = new double[tensor.Length];
            for (int i = 0; i < values.Length; i++) values[i] = tensor.GetFlat(i);
            weights.Remove(name);
            return values;
        });
        foreach (var sentence in Reference.Value.GetProperty("probabilities").EnumerateArray())
            foreach (var q in sentence.GetProperty("queries").EnumerateArray())
            {
                var labels = q.GetProperty("labels").EnumerateArray().Select(v => v.GetInt32()).ToList();
                var mask = new double[1305];
                foreach (int l in labels) mask[l] = 1;
                var probabilities = model.Probabilities(q.GetProperty("input_ids").EnumerateArray().Select(v => v.GetInt32()).ToArray(),
                    new[] { q.GetProperty("position").GetInt32() }, new[] { q.GetProperty("char_id").GetInt32() }, new[] { mask });
                var expected = q.GetProperty("probabilities").EnumerateArray().Select(v => v.GetDouble()).ToList();
                for (int j = 0; j < labels.Count; j++)
                    Assert.True(Math.Abs(expected[j] - probabilities[0, labels[j]]) < 1e-4,
                        $"{q.GetProperty("text").GetString()}: label {labels[j]} expected {expected[j]} actual {probabilities[0, labels[j]]}");
            }
        var converter = new G2pwConverter<double>(model);
        var failures = new List<string>();
        foreach (var s in Reference.Value.GetProperty("sentences").EnumerateArray())
        {
            var batch = Strings(s.GetProperty("segments")).Where(x => x.Length > 0).ToList();
            if (batch.Count == 0) continue;
            var expected = Readings(s.GetProperty("g2pw"));
            var actual = converter.Predict(batch);
            for (int i = 0; i < batch.Count; i++)
                if (!expected[i].SequenceEqual(actual[i]))
                    failures.Add($"{batch[i]}: {string.Join(" ", expected[i])} vs {string.Join(" ", actual[i])}");
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }
}
