using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// Whisper's multilingual tiktoken encoding, CosyVoice's text tokenizer, against openai-whisper's
/// <c>get_encoding("multilingual")</c> (<c>tools/reference-data/whisper_tiktoken_reference.py</c>) for 99 and 100 languages.
/// </summary>
public class WhisperTiktokenReferenceTests
{
    private static JObject Reference()
    {
        const string fileName = "whisper_tiktoken_reference.json";
        string output = Path.Combine(AppContext.BaseDirectory, "TextToSpeech", "ReferenceData", fileName);
        if (!File.Exists(output))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(output))
            {
                output = Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", "TextToSpeech", "ReferenceData", fileName);
                dir = dir.Parent;
            }
        }
        return JObject.Parse(File.ReadAllText(output));
    }

    [Theory]
    [InlineData(99)]
    [InlineData(100)]
    public void VocabularyAndSpecialTokens_MatchWhisper(int languages)
    {
        var f = Reference()[languages.ToString()]!;
        var encoding = WhisperTiktoken.Multilingual(languages);
        Assert.Equal((int)f["n_vocab"]!, encoding.VocabularySize);
        var expected = ((JObject)f["specials"]!).Properties().ToDictionary(p => p.Name, p => (int)p.Value);
        Assert.Equal(expected.Count, encoding.SpecialTokens.Count);
        foreach (var pair in expected) Assert.Equal(pair.Value, encoding.SpecialTokenId(pair.Key));
    }

    [Theory]
    [InlineData(99)]
    [InlineData(100)]
    public void Encode_MatchesTiktoken_AndDecodesBack(int languages)
    {
        var reference = Reference();
        var texts = reference["texts"]!.Select(t => (string)t!).ToArray();
        var tokens = reference[languages.ToString()]!["tokens"]!;
        var encoding = WhisperTiktoken.Multilingual(languages);
        for (int i = 0; i < texts.Length; i++)
        {
            var expected = tokens[i]!.Select(t => (int)t).ToArray();
            var actual = encoding.Encode(texts[i]);
            Assert.True(expected.SequenceEqual(actual),
                $"text {i} '{texts[i]}': tiktoken [{string.Join(", ", expected)}], port [{string.Join(", ", actual)}].");
            Assert.Equal(texts[i], encoding.Decode(actual));
        }
    }

    [Fact]
    public void ShippedVocabulary_IsWhispers()
    {
        var resource = typeof(WhisperTiktoken).Assembly.GetManifestResourceStream("AiDotNet.Tokenization.whisper-multilingual.tiktoken.gz");
        Assert.NotNull(resource);
        using var gzip = new System.IO.Compression.GZipStream(resource, System.IO.Compression.CompressionMode.Decompress);
        using var sha = SHA256.Create();
        var hash = string.Concat(sha.ComputeHash(gzip).Select(b => b.ToString("x2")));
        Assert.Equal((string)Reference()["vocabulary_sha256"]!, hash);
    }

    [Fact]
    public void Encode_WithoutSpecials_TreatsTheirTextAsText()
    {
        var encoding = WhisperTiktoken.Multilingual(100);
        var plain = encoding.Encode("<|endoftext|>", allowSpecial: false);
        Assert.DoesNotContain(encoding.SpecialTokenId("<|endoftext|>"), plain);
        Assert.Equal("<|endoftext|>", encoding.Decode(plain));
    }
}
