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
