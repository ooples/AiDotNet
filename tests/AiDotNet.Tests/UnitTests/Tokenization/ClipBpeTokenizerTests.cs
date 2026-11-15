using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// CLIP's BPE tokenizer against ids derived by hand from open_clip's <c>SimpleTokenizer</c> over a two-merge vocabulary.
/// </summary>
/// <remarks>
/// The vocabulary is <c>bytes_to_unicode()</c>'s 256 characters in its order ('!'..'~' are ids 0..93, so 'a' is 64,
/// 'c' 66, 'i' 72, 's' 82, 't' 83, the apostrophe 6, '&amp;' 5; 0xA1..0xAC are 94..105; 0xAE..0xFF are 106..187), the same
/// with <c>&lt;/w&gt;</c> (+256), the merges "c a" (512) and "ca t&lt;/w&gt;" (513), then
/// <c>&lt;start_of_text&gt;</c> (514) and <c>&lt;end_of_text&gt;</c> (515).
/// </remarks>
public class ClipBpeTokenizerTests
{
    private const string Merges = "#version: 0.2\nc a\nca t</w>\n";

    private static ClipBpeTokenizer Tiny() => ClipBpeTokenizer.FromStream(new MemoryStream(Encoding.UTF8.GetBytes(Merges)));

    [Fact]
    public void SpecialTokens_FollowTheMerges()
    {
        var tokenizer = Tiny();
        Assert.Equal(514, tokenizer.StartOfTextId);
        Assert.Equal(515, tokenizer.EndOfTextId);
    }

    [Theory]
    [InlineData("cat", new[] { 513 })]                  // c a t</w> -> ca t</w> -> cat</w>
    [InlineData("a cat!", new[] { 320, 513, 256 })]     // a</w>, cat</w>, !</w>
    [InlineData("tac", new[] { 83, 64, 322 })]          // no merge applies: t, a, c</w>
    [InlineData("&amp;amp;", new[] { 261 })]            // html.unescape twice: "&", then &</w>
    [InlineData("  CAT \n  a ", new[] { 513, 320 })]    // whitespace_clean, strip, lower
    [InlineData("it's", new[] { 72, 339, 6, 338 })]     // "it", then the contraction "'s"
    [InlineData("é", new[] { 127, 358 })]          // UTF-8 C3 A9: byte chars 0xC3 (127), 0xA9</w> (102 + 256)
    [InlineData("   ", new int[0])]
    public void EncodeIds_MatchesHandDerivedIds(string text, int[] expected)
    {
        Assert.Equal(expected, Tiny().EncodeIds(text).ToArray());
    }

    [Fact]
    public void EncodeContext_AddsStartAndEndOfTextAndZeroPadsTo77()
    {
        var ids = Tiny().EncodeContext("a cat!");
        Assert.Equal(77, ids.Length);
        Assert.Equal(new[] { 514, 320, 513, 256, 515 }, ids.Take(5).ToArray());
        Assert.All(ids.Skip(5), id => Assert.Equal(0, id));

        Assert.Equal(new[] { 514, 515 }, Tiny().EncodeContext(string.Empty).Take(2).ToArray());
    }

    [Fact]
    public void EncodeContext_TruncatesKeepingEndOfTextLast()
    {
        var text = string.Join(" ", Enumerable.Repeat("a", 100));
        var ids = Tiny().EncodeContext(text);
        Assert.Equal(77, ids.Length);
        Assert.Equal(514, ids[0]);
        Assert.All(ids.Skip(1).Take(75), id => Assert.Equal(320, id));
        Assert.Equal(515, ids[76]);
    }

    [Fact]
    public void Encode_WithOptionsMatchesEncodeContext()
    {
        var tokenizer = Tiny();
        var options = new EncodingOptions { MaxLength = 77, Padding = true, Truncation = true };
        var result = tokenizer.Encode("a cat!", options);
        Assert.Equal(tokenizer.EncodeContext("a cat!"), result.TokenIds.ToArray());
        Assert.Equal(5, result.AttentionMask.Sum());

        var plain = tokenizer.Encode("a cat!", new EncodingOptions { AddSpecialTokens = false });
        Assert.Equal(new[] { 320, 513, 256 }, plain.TokenIds.ToArray());
    }

    [Fact]
    public void Decode_IsSimpleTokenizerDecode()
    {
        var tokenizer = Tiny();
        // Each </w> becomes a space (open_clip's decode leaves the trailing one); padding after EOT is dropped.
        Assert.Equal("a cat! ", tokenizer.Decode(tokenizer.EncodeContext("a cat!").ToList()));
        Assert.Equal("é ", tokenizer.Decode(new List<int> { 127, 358 }));
    }

    [Fact]
    public void FromStream_ReadsTheGzippedMergesFile()
    {
        var compressed = new MemoryStream();
        using (var gzip = new GZipStream(compressed, CompressionMode.Compress, leaveOpen: true))
        {
            var bytes = Encoding.UTF8.GetBytes(Merges);
            gzip.Write(bytes, 0, bytes.Length);
        }
        compressed.Position = 0;
        var tokenizer = ClipBpeTokenizer.FromStream(compressed);
        Assert.Equal(new[] { 320, 513, 256 }, tokenizer.EncodeIds("a cat!").ToArray());
    }

    [Fact]
    public void FullSizeMergesFile_UsesTheFirst48894MergesAndTheReleasedSpecialIds()
    {
        // merges[1:49152-256-2+1]: the header is skipped and merges past 48,894 are ignored, which puts the special
        // tokens at 49406 and 49407 (open_clip tokenize's sot/eot ids).
        var lines = new List<string> { "#version: 0.2" };
        for (int i = 0; i < ClipBpeTokenizer.MaxMerges + 10; i++)
            lines.Add($"{(char)(0x4E00 + (i / 300))} {(char)(0x4E00 + (i % 300))}");
        var tokenizer = ClipBpeTokenizer.FromMerges(lines);
        Assert.Equal(49406, tokenizer.StartOfTextId);
        Assert.Equal(49407, tokenizer.EndOfTextId);
        Assert.Equal(49408, tokenizer.VocabularySize);
    }

    [Fact]
    public void OpenClipTextConditioner_TokenizesRawTextWithTheClipTokenizer()
    {
        // The conditioner hands the raw text to CLIP's tokenizer, which does open_clip's cleaning (html unescape
        // included), and frames the ids with its own sot/eot (49406/49407) and zero padding.
        var options = new AiDotNet.Diffusion.Conditioning.OpenClipTextEncoderOptions { Width = 8, Layers = 2, Heads = 2 };
        var conditioner = new AiDotNet.Diffusion.Conditioning.OpenClipTextConditioner<double>(options, Tiny());
        var ids = conditioner.Tokenize("A &amp;amp; CAT").ToArray().Select(v => (int)v).ToArray();
        Assert.Equal(77, ids.Length);
        Assert.Equal(new[] { 49406, 320, 261, 513, 49407 }, ids.Take(5).ToArray());
        Assert.All(ids.Skip(5), id => Assert.Equal(0, id));
    }

    [Fact]
    public void Clean_UnescapesHtmlTwiceAndCollapsesWhitespace()
    {
        Assert.Equal("a & b <c>", ClipBpeTokenizer.Clean("  A &amp;amp; B\t\t&lt;C&gt; \r\n"));
    }
}
