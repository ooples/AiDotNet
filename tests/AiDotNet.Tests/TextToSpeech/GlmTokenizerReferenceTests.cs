using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.TextToSpeech.MultiModal;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// GLM-4's tokenizer against GLM-4-Voice-9B's remote code (tools/reference-data/glm_tokenizer_reference.py): cl100k's
/// pre-token splits as tiktoken's regex makes them (always), and, with <c>GLM4VOICE_LM_DIR</c> holding the release's
/// <c>tokenizer.model</c> and <c>tokenizer_config.json</c>, the ids and decoding of texts, added tokens and the prompt.
/// </summary>
public class GlmTokenizerReferenceTests
{
    private static JObject Fixture()
    {
        const string fileName = "glm_tokenizer_reference.json";
        string path = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); !File.Exists(path) && dir is not null; dir = dir.Parent)
            path = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
        return JObject.Parse(File.ReadAllText(path));
    }

    [Fact]
    public void Cl100kPreTokens_MatchTiktoken()
    {
        var f = Fixture();
        // Any byte-level ranks; the split depends on the pattern alone.
        var ranks = Enumerable.Range(0, 256).ToDictionary(b => new[] { (byte)b }, b => b);
        var encoding = new TiktokenEncoding(ranks, Array.Empty<string>(), TiktokenPattern.Cl100k);
        var plain = f["plain"]!.Select(t => (string)t!).ToArray();
        var splits = f["splits"]!.Select(s => s.Select(t => (string)t!).ToArray()).ToArray();
        for (int i = 0; i < plain.Length; i++) Assert.Equal(splits[i], encoding.PreTokens(plain[i]));
    }

    [Fact]
    public void Ids_MatchTheReleaseTokenizer()
    {
        string? dir = Environment.GetEnvironmentVariable("GLM4VOICE_LM_DIR");
        if (dir is null || !File.Exists(Path.Combine(dir, "tokenizer.model"))) return;
        var f = Fixture();
        var tokenizer = GlmTokenizer.FromDirectory(dir);
        var texts = f["texts"]!.Select(t => (string)t!).ToArray();
        var ids = f["ids"]!.Select(a => a.Select(v => (int)v).ToArray()).ToArray();
        var decoded = f["decoded"]!.Select(t => (string)t!).ToArray();
        for (int i = 0; i < texts.Length; i++)
        {
            Assert.Equal(ids[i], tokenizer.Encode(texts[i]));
            Assert.Equal(decoded[i], tokenizer.Decode(ids[i].Skip(2)));
        }
        Assert.Equal(152353, tokenizer.TokenId("<|audio_0|>"));
    }
}
