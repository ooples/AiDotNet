using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.TextToSpeech.Latest;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// The byte-level BPE tokenizer against Hugging Face tokenizers (tools/reference-data/outetts_tokenizer_reference.py):
/// OuteTTS-1.0-0.6B's own tokenizer, built from the embedded Qwen2 data and the release's added tokens, and small
/// tokenizers of Llama 3's form (the release's 1B model) and GPT-2's, read from their tokenizer.json.
/// </summary>
public class OuteTTSTokenizerTests
{
    private static JObject Reference()
    {
        const string fileName = "outetts_tokenizer_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(output) && dir is not null; dir = dir.Parent)
            output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(output));
    }

    private static void Check(ByteLevelBpeTokenizer tokenizer, string[] texts, JToken ids, string what)
    {
        var expected = ids.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        for (int i = 0; i < texts.Length; i++)
        {
            var actual = tokenizer.Encode(texts[i]);
            Assert.True(expected[i].SequenceEqual(actual), $"{what}, text {i}: reference [{string.Join(",", expected[i])}], port [{string.Join(",", actual)}]");
            Assert.Equal(texts[i], tokenizer.Decode(actual));
        }
    }

    [Fact(Timeout = 300000)]
    public async Task Qwen3Tokenizer_MatchesTheRelease()
    {
        await Task.Yield();
        var f = Reference();
        Check(OuteTTSTokenizers.Qwen3, f["texts"]!.Select(t => (string)t!).ToArray(), f["qwen3_ids"]!, "OuteTTS-1.0-0.6B");
    }

    // HF tokenizers against OuteAI/Llama-OuteTTS-1.0-1B tokenizer.json, revision 6cb62a5,
    // SHA256 ec54a55e3c6fc7318ea02cbd1a6eb1fb180bff1b58acbf068504a25f1b407b7b;
    // encode(add_special_tokens=False), decode(skip_special_tokens=False).
    [Theory]
    [InlineData("Hello, world!", new[] { 9906, 11, 1917, 0 })]
    [InlineData("cafe\u0301 / caf\u00e9 / \u4f60\u597d / \ud83d\ude42", new[] { 936, 1897, 54939, 611, 53050, 611, 118195, 53901, 611, 28584 })]
    [InlineData("1234567890", new[] { 4513, 10961, 16474, 15 })]
    [InlineData("<|im_start|>user\n<|text_start|>Hello<|text_end|><|audio_start|>", new[] { 133309, 882, 198, 133311, 9906, 133312, 133317 })]
    [InlineData("<|c1_0|><|c1_1024|><|c2_0|><|c2_1024|><|t_0.00|><|t_10.00|>", new[] { 128256, 129280, 129281, 130305, 133325, 134325 })]
    [InlineData("<|energy_0|><|energy_1000|><|spectral_centroid_1000|><|pitch_1000|>", new[] { 130306, 131306, 132307, 133308 })]
    public void LlamaReleaseTokenizer_MatchesPublishedTokenIds(string text, int[] expected)
    {
        var tokenizer = OuteTTSTokenizers.Llama;
        Assert.Equal(expected, tokenizer.Encode(text));
        Assert.Equal(text, tokenizer.Decode(expected));
    }

    [Fact(Timeout = 120000)]
    public async Task Llama3FormTokenizer_MatchesTokenizers()
    {
        await Task.Yield();
        var f = Reference();
        var form = f["llama3_form"]!;
        Check(ByteLevelBpeTokenizer.FromJson((string)form["tokenizer"]!), f["texts"]!.Select(t => (string)t!).ToArray(), form["ids"]!, "Llama 3 form");
    }

    [Fact(Timeout = 120000)]
    public async Task Gpt2FormTokenizer_MatchesTokenizers()
    {
        await Task.Yield();
        var f = Reference();
        var form = f["gpt2_form"]!;
        Check(ByteLevelBpeTokenizer.FromJson((string)form["tokenizer"]!), f["texts"]!.Select(t => (string)t!).ToArray(), form["ids"]!, "GPT-2 form");
    }
}
