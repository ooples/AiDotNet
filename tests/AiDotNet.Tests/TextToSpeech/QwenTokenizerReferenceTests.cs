using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using AiDotNet.Tokenization.Algorithms;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>
/// CosyVoice 2's text tokenizer against transformers' Qwen2 tokenizer with CosyVoice's added tokens
/// (<c>tools/reference-data/qwen2_tokenizer_reference.py</c>).
/// </summary>
public class QwenTokenizerReferenceTests
{
    private static readonly Lazy<JObject> Data = new(() =>
    {
        const string fileName = "qwen2_tokenizer_reference.json";
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
    });

    [Fact]
    public void Encode_MatchesTransformers_AndDecodesBack()
    {
        var tokenizer = QwenTokenizer.CosyVoice2;
        var texts = Data.Value["texts"]!.Select(t => (string)t!).ToArray();
        for (int i = 0; i < texts.Length; i++)
        {
            var expected = Data.Value["ids"]![i]!.Select(v => (int)v).ToArray();
            var actual = tokenizer.Encode(texts[i]);
            Assert.True(expected.SequenceEqual(actual),
                $"text {i} '{texts[i]}': transformers [{string.Join(", ", expected)}], port [{string.Join(", ", actual)}].");
            Assert.Equal(texts[i].Normalize(System.Text.NormalizationForm.FormC), tokenizer.Decode(actual, skipAdded: false));
        }
    }

    [Fact]
    public void Vocabulary_AndAddedTokens_MatchCosyVoice2Tokenizer()
    {
        var tokenizer = QwenTokenizer.CosyVoice2;
        Assert.Equal((int)Data.Value["vocabulary"]!, tokenizer.VocabularySize);
        foreach (var pair in ((JObject)Data.Value["added"]!).Properties())
            Assert.Equal((int)pair.Value, tokenizer.AddedTokenId(pair.Name));
    }

    [Theory]
    [InlineData("AiDotNet.Tokenization.qwen2-vocab.json.gz", "vocab.json_sha256")]
    [InlineData("AiDotNet.Tokenization.qwen2-merges.txt.gz", "merges.txt_sha256")]
    public void ShippedFiles_AreQwens(string resource, string key)
    {
        var stream = typeof(QwenTokenizer).Assembly.GetManifestResourceStream(resource);
        Assert.NotNull(stream);
        using var gzip = new System.IO.Compression.GZipStream(stream, System.IO.Compression.CompressionMode.Decompress);
        using var sha = SHA256.Create();
        Assert.Equal((string)Data.Value[key]!, string.Concat(sha.ComputeHash(gzip).Select(b => b.ToString("x2"))));
    }
}
