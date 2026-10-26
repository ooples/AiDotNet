using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.TextToSpeech.FlowDiffusion;
using AiDotNet.TextToSpeech.FrontEnd;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// MaskGCT's Mandarin front end against the release's (tools/reference-data/maskgct_mandarin_reference.py, in the release's
/// environment): Chinese and mixed texts through <c>g2p_</c>, phone for phone and id for id, with the polyphone model's
/// predictions and eSpeak's English output replayed (each call checked); cn2an's transform; and, where the release's
/// <c>poly_bert_model.onnx</c> is present, the native polyphone model against the recorded logits.
/// </summary>
public class MaskGctMandarinTests
{
    private static JsonElement Load(string name)
    {
        string relative = Path.Combine("TextToSpeech", "ReferenceData", name);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
        string path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    [Fact(Timeout = 300000)]
    public async Task G2P_MatchesTheRelease()
    {
        await Task.Yield();
        var reference = Load("maskgct_mandarin_reference.json");
        foreach (var c in reference.GetProperty("cases").EnumerateArray())
        {
            var espeak = new Queue<JsonElement>(c.GetProperty("espeak").EnumerateArray());
            var poly = new Queue<JsonElement>(c.GetProperty("poly").EnumerateArray());
            var mandarin = new MaskGctMandarinFrontEnd((text, index) =>
            {
                Assert.True(poly.Count > 0, $"an unexpected polyphone prediction: {text} @ {index}");
                var call = poly.Dequeue();
                Assert.Equal(call.GetProperty("text").GetString(), text);
                Assert.Equal(call.GetProperty("index").GetInt32(), index);
                string label = call.GetProperty("pinyin")[0].GetString()!;
                return label.Substring(2, label.Length - 3);
            });
            var frontEnd = new MaskGctEspeakFrontEnd((language, text) =>
            {
                var call = espeak.Dequeue();
                Assert.Equal(call.GetProperty("voice").GetString(), language);
                Assert.Equal(call.GetProperty("text").GetString(), text);
                return call.GetProperty("phonemes").GetString() ?? "";
            });
            var (phonemes, tokens) = frontEnd.G2P(c.GetProperty("text").GetString()!, "zh", (segment, _) => mandarin.ChineseToIpa(segment));
            Assert.Empty(espeak);
            Assert.Empty(poly);
            Assert.Equal(c.GetProperty("phonemes").GetString(), phonemes);
            Assert.Equal(c.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray(), tokens);
        }
    }

    [Theory(Timeout = 60000)]
    [InlineData("今天是2024年10月18日，温度25℃，增长3.5%，1/3的人，-12.05元，第123期，电话13800138000。",
        "今天是二零二四年十月十八日，温度二十五摄氏度，增长百分之三点五，三分之一的人，负十二点零五元，第一百二十三期，电话一百三十八亿零一十三万八千。")]
    [InlineData("全角１２３和٣", "全角一百二十三和٣")]
    public async Task Cn2AnTransform_MatchesCn2An(string text, string expected)
    {
        await Task.Yield();
        Assert.Equal(expected, Cn2An.Transform(text));
    }

    [Fact(Timeout = 300000)]
    public async Task PolyphoneModel_MatchesTheReleasedOnnx_WhenPresent()
    {
        await Task.Yield();
        string? onnx = Environment.GetEnvironmentVariable("MASKGCT_POLY_ONNX");
        if (onnx is null || !File.Exists(onnx)) return;
        var model = new MaskGctPolyphoneModel<double>(new CpuEngine());
        model.Load(onnx);
        foreach (var call in Load("maskgct_mandarin_reference.json").GetProperty("cases").EnumerateArray().SelectMany(c => c.GetProperty("poly").EnumerateArray()))
        {
            string text = call.GetProperty("text").GetString()!;
            int index = call.GetProperty("index").GetInt32();
            Assert.Equal(call.GetProperty("ids").EnumerateArray().Select(v => v.GetInt32()).ToArray(), MaskGctMandarinFrontEnd.BertIds(text));
            var logits = model.Logits(text);
            var expected = call.GetProperty("logits").EnumerateArray().Select(v => v.GetDouble()).ToArray();
            for (int k = 0; k < expected.Length; k++)
                Assert.True(Math.Abs(expected[k] - logits[index, k]) <= 1e-3, $"logit {k}: {expected[k]} vs {logits[index, k]}");
            string label = call.GetProperty("pinyin")[0].GetString()!;
            Assert.Equal(label.Substring(2, label.Length - 3), model.Predict(text, index));
        }
    }
}
