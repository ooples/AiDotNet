using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.VisionLanguage.Grounding;
using Xunit;

namespace AiDotNet.Tests.VisionLanguage;

/// <summary>
/// SentencePiece BPE with byte fallback against the sentencepiece library, and GLaMM's tokenizer against a transcription of
/// the slow LLaMA tokenizer GLaMM pins (tools/reference-data/glamm_tokenizer_reference.py), on a BPE model trained with
/// LLaMA's settings: added tokens with stripped neighbours, per-part encoding, LLaVA's image placeholder and decoding.
/// </summary>
public class GlammTokenizerReferenceTests
{
    private static JsonElement Fixture()
    {
        const string name = "glamm_tokenizer_reference.json";
        string relative = Path.Combine("VisionLanguage", "ReferenceData", name);
        string path = Path.Combine(AppContext.BaseDirectory, relative);
        if (!File.Exists(path))
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "tests", "AiDotNet.Tests", relative))) dir = dir.Parent;
            path = Path.Combine(dir?.FullName ?? throw new FileNotFoundException(name), "tests", "AiDotNet.Tests", relative);
        }
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    /// <summary>The fixture's BPE model, for other tests that need a GLaMM tokenizer.</summary>
    internal static byte[] TestModel() => Convert.FromBase64String(Fixture().GetProperty("model").GetString()!);

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    [Fact]
    public void Bpe_MatchesSentencePiece()
    {
        var k = Fixture();
        var model = new SentencePieceModel(Convert.FromBase64String(k.GetProperty("model").GetString()!));
        Assert.True(model.IsBpe);
        Assert.Equal(k.GetProperty("pieces").GetInt32(), model.Count);
        foreach (var c in k.GetProperty("cases").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            var pieces = model.EncodePieces(text);
            Assert.Equal(c.GetProperty("pieces").EnumerateArray().Select(p => p.GetString()).ToArray(), pieces.Select(p => p.Piece).ToArray());
            Assert.Equal(Ints(c.GetProperty("spm_ids")), pieces.Select(p => p.Id).ToArray());
        }
    }

    [Fact]
    public void GlammTokenizer_MatchesThePinnedSlowTokenizer()
    {
        var k = Fixture();
        var tokenizer = new GlammTokenizer(new SentencePieceModel(Convert.FromBase64String(k.GetProperty("model").GetString()!)));
        foreach (var c in k.GetProperty("cases").EnumerateArray())
        {
            string text = c.GetProperty("text").GetString()!;
            var ids = tokenizer.Encode(text);
            Assert.Equal(Ints(c.GetProperty("ids")), ids.ToArray());
            Assert.Equal(Ints(c.GetProperty("image_ids")), tokenizer.EncodeWithImage(text).ToArray());
            Assert.Equal(c.GetProperty("decoded").GetString(), tokenizer.Decode(ids));
        }
    }
}
