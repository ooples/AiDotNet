using System;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.MultiModal;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// SpeechGPT's tokenizer against transformers 4.29's slow LlamaTokenizer over the release's files
/// (<c>tools/reference-data/speechgpt_tokenizer_reference.py</c>): the release's prompts, units and edge cases of white
/// space beside the added tokens. Gated on SPEECHGPT_DIR (a directory with <c>tokenizer.model</c> and
/// <c>added_tokens.json</c> of fnlp/SpeechGPT-7B-cm).
/// </summary>
public class SpeechGptTokenizerReferenceTests
{
    [Fact]
    public void Tokenizer_MatchesTheRelease()
    {
        string? dir = Environment.GetEnvironmentVariable("SPEECHGPT_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "tokenizer.model"))) return;
        const string fileName = "speechgpt_tokenizer_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && d is not null; d = d.Parent)
            path = Path.Combine(d.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        var f = JObject.Parse(File.ReadAllText(path));
        var tokenizer = SpeechGptTokenizer.FromDirectory(dir);
        var texts = f["texts"]!.Select(t => (string)t!).ToArray();
        var ids = f["ids"]!.Select(r => r.Select(v => (int)v).ToArray()).ToArray();
        for (int i = 0; i < texts.Length; i++) Assert.Equal(ids[i], tokenizer.Encode(texts[i]));
    }
}
