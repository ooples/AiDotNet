using System;
using System.Linq;
using AiDotNet.TextToSpeech.FrontEnd;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Tests.TextToSpeech;

/// <summary>The Russian stress front end against russian_text_stresser 0.6.0 under spaCy 3.6.1
/// (tools/reference-data/russian_text_stresser_reference.py). The pipeline checks run with <c>RUSSIAN_STRESSER_DIR</c>
/// (russian_dict.db, simple_cases.pkl and the ru_core_news_sm-3.6.0 tokenizer, tok2vec/model, morphologizer/*).</summary>
public class RussianTextStresserReferenceTests
{
    private static string? Directory() =>
        Environment.GetEnvironmentVariable("RUSSIAN_STRESSER_DIR") is { Length: > 0 } path && System.IO.Directory.Exists(path) ? path : null;

    [Fact]
    public void Hashes_MatchSpacyAndThinc()
    {
        // spacy.strings.hash_string and thinc NumpyOps.hash(ids, seed=8).
        Assert.Equal(14313749767032693980UL, SpacyRussianTagger.HashString(""));
        Assert.Equal(11901859001352538922UL, SpacyRussianTagger.HashString("a"));
        Assert.Equal(17751780907874141760UL, SpacyRussianTagger.HashString("abcdefgh"));
        Assert.Equal(13885881106499164936UL, SpacyRussianTagger.HashString("abcdefghi"));
        Assert.Equal(10887629174180191697UL, SpacyRussianTagger.HashString("Xxxx"));
        Assert.Equal(new uint[] { 0, 0, 0, 0 }, SpacyTok2Vec.Hash(0, 8));
        Assert.Equal(new uint[] { 1161550108, 2247783286, 1742325162, 1224191281 }, SpacyTok2Vec.Hash(1, 8));
        Assert.Equal(new uint[] { 1425090886, 1378295307, 4285119977, 4214926608 }, SpacyTok2Vec.Hash(6558124450146518386UL, 8));
        Assert.Equal(new uint[] { 2488739950, 2974852181, 1585626277, 2314794624 }, SpacyTok2Vec.Hash(ulong.MaxValue, 8));
    }

    [Fact]
    public void Pipeline_TagsAndStressesLikeTheRelease()
    {
        var directory = Directory();
        if (directory is null) return;
        var f = WhisperSpeechT2SReferenceTests.Load("russian_text_stresser_reference.json");
        var tagger = SpacyRussianTagger.Load(directory);
        using var stresser = RussianTextStresser.Load(directory);
        foreach (var row in f["sentences"]!)
        {
            foreach (var (textKey, tokenKey) in new[] { ("text", "tokens"), ("chatterbox_text", "chatterbox_tokens") })
            {
                string text = (string)row[textKey]!;
                var tokens = tagger.Tag(text);
                var expected = row[tokenKey]!.ToArray();
                Assert.True(expected.Select(t => (string)t["text"]!).SequenceEqual(tokens.Select(t => t.Text)),
                    $"tokens of '{text}': {string.Join(" | ", tokens.Select(t => t.Text))}");
                Assert.Equal(expected.Select(t => (string)t["ws"]!).ToArray(), tokens.Select(t => t.Space ? " " : "").ToArray());
                var features = row[tokenKey + "_features"]!.Select(r => r.Select(v => ulong.Parse(v.ToString(), System.Globalization.CultureInfo.InvariantCulture)).ToArray()).ToArray();
                for (int i = 0; i < tokens.Count; i++)
                    Assert.True(features[i].SequenceEqual(SpacyRussianTagger.Features(tokens[i])), $"features of '{tokens[i].Text}' in '{text}'");
                for (int i = 0; i < tokens.Count; i++)
                {
                    Assert.True((string)expected[i]["pos"]! == tokens[i].Pos, $"POS of '{tokens[i].Text}' in '{text}': {tokens[i].Pos}");
                    Assert.True((string)expected[i]["morph"]! == tokens[i].Morph, $"morph of '{tokens[i].Text}' in '{text}': {tokens[i].Morph}");
                }
            }
            Assert.Equal((string)row["stressed"]!, stresser.StressText((string)row["text"]!));
            Assert.Equal((string)row["chatterbox_stressed"]!, stresser.StressText((string)row["chatterbox_text"]!));
        }
    }
}
