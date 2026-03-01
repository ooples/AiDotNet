using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Vocabulary;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Tokenization;

/// <summary>
/// Deep integration tests for Vocabulary, SpecialTokens: token ID assignment,
/// bijective mapping, unknown token handling, batch operations, presets,
/// clearing, edge cases, and invariant properties.
/// </summary>
public class TokenizationDeepMathIntegrationTests
{
    // ============================
    // Vocabulary Construction Tests
    // ============================

    [Fact]
    public void Vocabulary_DefaultConstruction_HasUnkToken()
    {
        var vocab = new Vocabulary();

        Assert.Equal(1, vocab.Size);
        Assert.True(vocab.ContainsToken("[UNK]"));
    }

    [Fact]
    public void Vocabulary_CustomUnkToken()
    {
        var vocab = new Vocabulary("<unk>");

        Assert.Equal(1, vocab.Size);
        Assert.True(vocab.ContainsToken("<unk>"));
        Assert.False(vocab.ContainsToken("[UNK]"));
    }

    [Fact]
    public void Vocabulary_FromDictionary_PreservesMapping()
    {
        var dict = new Dictionary<string, int>
        {
            { "hello", 0 },
            { "world", 1 },
            { "[UNK]", 2 }
        };

        var vocab = new Vocabulary(dict);

        Assert.Equal(0, vocab.GetTokenId("hello"));
        Assert.Equal(1, vocab.GetTokenId("world"));
        Assert.Equal(2, vocab.GetTokenId("[UNK]"));
    }

    [Fact]
    public void Vocabulary_FromDictionary_NullThrows()
    {
        Dictionary<string, int>? nullDict = null;
        Assert.Throws<ArgumentNullException>(() =>
            new Vocabulary(nullDict!));
    }

    // ============================
    // Token ID Assignment Tests
    // ============================

    [Fact]
    public void AddToken_AssignsSequentialIds()
    {
        var vocab = new Vocabulary();
        // [UNK] is id 0

        var id1 = vocab.AddToken("hello");
        var id2 = vocab.AddToken("world");
        var id3 = vocab.AddToken("foo");

        Assert.Equal(1, id1);
        Assert.Equal(2, id2);
        Assert.Equal(3, id3);
    }

    [Fact]
    public void AddToken_DuplicateReturnsExistingId()
    {
        var vocab = new Vocabulary();

        var id1 = vocab.AddToken("hello");
        var id2 = vocab.AddToken("hello"); // duplicate

        Assert.Equal(id1, id2);
        Assert.Equal(2, vocab.Size); // [UNK] + "hello" = 2, not 3
    }

    [Fact]
    public void AddToken_EmptyString_Throws()
    {
        var vocab = new Vocabulary();
        Assert.Throws<ArgumentException>(() => vocab.AddToken(""));
    }

    [Fact]
    public void AddToken_NullString_Throws()
    {
        var vocab = new Vocabulary();
        Assert.Throws<ArgumentException>(() => vocab.AddToken(null!));
    }

    // ============================
    // Bijective Mapping Tests (token <-> id)
    // ============================

    [Fact]
    public void TokenToId_And_IdToToken_AreBijective()
    {
        var vocab = new Vocabulary();
        var tokens = new[] { "the", "cat", "sat", "on", "mat" };

        foreach (var token in tokens)
            vocab.AddToken(token);

        // For every token, GetTokenId -> GetToken roundtrips
        foreach (var token in tokens)
        {
            var id = vocab.GetTokenId(token);
            var roundtrip = vocab.GetToken(id);
            Assert.Equal(token, roundtrip);
        }
    }

    [Fact]
    public void AllIds_MapToUniqueTokens()
    {
        var vocab = new Vocabulary();
        var tokens = new[] { "a", "b", "c", "d", "e" };

        foreach (var token in tokens)
            vocab.AddToken(token);

        var seenTokens = new HashSet<string>();
        for (int id = 0; id < vocab.Size; id++)
        {
            var token = vocab.GetToken(id);
            Assert.NotNull(token);
            Assert.True(seenTokens.Add(token), $"Duplicate token for id {id}: {token}");
        }
    }

    [Fact]
    public void AllTokens_MapToUniqueIds()
    {
        var vocab = new Vocabulary();
        var tokens = new[] { "alpha", "beta", "gamma", "delta" };

        foreach (var token in tokens)
            vocab.AddToken(token);

        var seenIds = new HashSet<int>();
        foreach (var token in vocab.GetAllTokens())
        {
            var id = vocab.GetTokenId(token);
            Assert.True(seenIds.Add(id), $"Duplicate id for token {token}: {id}");
        }
    }

    // ============================
    // Unknown Token Handling Tests
    // ============================

    [Fact]
    public void GetTokenId_UnknownToken_ReturnsUnkId()
    {
        var vocab = new Vocabulary();
        vocab.AddToken("hello");

        var unknownId = vocab.GetTokenId("nonexistent");
        var unkId = vocab.GetTokenId("[UNK]");

        Assert.Equal(unkId, unknownId);
    }

    [Fact]
    public void GetToken_InvalidId_ReturnsNull()
    {
        var vocab = new Vocabulary();

        var token = vocab.GetToken(999);
        Assert.Null(token);
    }

    [Fact]
    public void ContainsToken_ExistingToken_True()
    {
        var vocab = new Vocabulary();
        vocab.AddToken("test");

        Assert.True(vocab.ContainsToken("test"));
    }

    [Fact]
    public void ContainsToken_NonExistingToken_False()
    {
        var vocab = new Vocabulary();

        Assert.False(vocab.ContainsToken("nonexistent"));
    }

    [Fact]
    public void ContainsId_ExistingId_True()
    {
        var vocab = new Vocabulary();
        Assert.True(vocab.ContainsId(0)); // [UNK] id
    }

    [Fact]
    public void ContainsId_NonExistingId_False()
    {
        var vocab = new Vocabulary();
        Assert.False(vocab.ContainsId(999));
    }

    // ============================
    // Batch Operations Tests
    // ============================

    [Fact]
    public void AddTokens_AddsAllTokens()
    {
        var vocab = new Vocabulary();
        var tokens = new[] { "the", "quick", "brown", "fox" };

        vocab.AddTokens(tokens);

        Assert.Equal(5, vocab.Size); // 4 + [UNK]
        foreach (var token in tokens)
            Assert.True(vocab.ContainsToken(token));
    }

    [Fact]
    public void AddTokens_NullCollection_Throws()
    {
        var vocab = new Vocabulary();
        Assert.Throws<ArgumentNullException>(() => vocab.AddTokens(null!));
    }

    [Fact]
    public void AddTokens_WithDuplicates_CountsUnique()
    {
        var vocab = new Vocabulary();
        vocab.AddTokens(new[] { "a", "b", "a", "c", "b" });

        Assert.Equal(4, vocab.Size); // [UNK], a, b, c
    }

    [Fact]
    public void GetAllTokens_ReturnsAllAddedTokens()
    {
        var vocab = new Vocabulary();
        var tokens = new[] { "x", "y", "z" };

        vocab.AddTokens(tokens);

        var allTokens = new HashSet<string>(vocab.GetAllTokens());
        Assert.Contains("[UNK]", allTokens);
        foreach (var token in tokens)
            Assert.Contains(token, allTokens);
    }

    // ============================
    // Clear Tests
    // ============================

    [Fact]
    public void Clear_ResetsToOnlyUnk()
    {
        var vocab = new Vocabulary();
        vocab.AddTokens(new[] { "a", "b", "c" });

        Assert.Equal(4, vocab.Size); // [UNK] + 3

        vocab.Clear();

        Assert.Equal(1, vocab.Size); // only [UNK]
        Assert.True(vocab.ContainsToken("[UNK]"));
        Assert.False(vocab.ContainsToken("a"));
    }

    [Fact]
    public void Clear_CanAddTokensAgainAfter()
    {
        var vocab = new Vocabulary();
        vocab.AddToken("old");
        vocab.Clear();

        var newId = vocab.AddToken("new");
        Assert.Equal(1, newId); // 0 is [UNK] after clear
    }

    [Fact]
    public void Clear_UnkIdPreserved()
    {
        var vocab = new Vocabulary();
        vocab.AddToken("test");
        vocab.Clear();

        // After clear, [UNK] should still be id 0
        Assert.Equal(0, vocab.GetTokenId("[UNK]"));
    }

    // ============================
    // Vocabulary Size Invariants
    // ============================

    [Fact]
    public void Size_EqualsTokenToIdCount()
    {
        var vocab = new Vocabulary();
        vocab.AddTokens(new[] { "a", "b", "c", "d" });

        Assert.Equal(vocab.Size, vocab.TokenToId.Count);
    }

    [Fact]
    public void Size_EqualsIdToTokenCount()
    {
        var vocab = new Vocabulary();
        vocab.AddTokens(new[] { "a", "b", "c", "d" });

        Assert.Equal(vocab.Size, vocab.IdToToken.Count);
    }

    [Fact]
    public void TokenToId_And_IdToToken_SameSize()
    {
        var vocab = new Vocabulary();
        vocab.AddTokens(new[] { "alpha", "beta", "gamma" });

        Assert.Equal(vocab.TokenToId.Count, vocab.IdToToken.Count);
    }

    // ============================
    // Large Vocabulary Tests
    // ============================

    [Fact]
    public void LargeVocabulary_1000Tokens_AllRetrievable()
    {
        var vocab = new Vocabulary();

        for (int i = 0; i < 1000; i++)
            vocab.AddToken($"token_{i}");

        Assert.Equal(1001, vocab.Size); // 1000 + [UNK]

        for (int i = 0; i < 1000; i++)
        {
            var token = $"token_{i}";
            Assert.True(vocab.ContainsToken(token));

            var id = vocab.GetTokenId(token);
            var roundtrip = vocab.GetToken(id);
            Assert.Equal(token, roundtrip);
        }
    }

    // ============================
    // SpecialTokens Preset Tests
    // ============================

    [Fact]
    public void SpecialTokens_Bert_HasStandardTokens()
    {
        var tokens = SpecialTokens.Bert();

        Assert.Equal("[UNK]", tokens.UnkToken);
        Assert.Equal("[PAD]", tokens.PadToken);
        Assert.Equal("[CLS]", tokens.ClsToken);
        Assert.Equal("[SEP]", tokens.SepToken);
        Assert.Equal("[MASK]", tokens.MaskToken);
    }

    [Fact]
    public void SpecialTokens_Gpt_HasEndOfText()
    {
        var tokens = SpecialTokens.Gpt();

        Assert.Equal("<|endoftext|>", tokens.UnkToken);
        Assert.Equal("<|endoftext|>", tokens.PadToken);
        Assert.Equal("<|endoftext|>", tokens.BosToken);
        Assert.Equal("<|endoftext|>", tokens.EosToken);
    }

    [Fact]
    public void SpecialTokens_T5_HasCorrectTokens()
    {
        var tokens = SpecialTokens.T5();

        Assert.Equal("<unk>", tokens.UnkToken);
        Assert.Equal("<pad>", tokens.PadToken);
        Assert.Equal("</s>", tokens.EosToken);
    }

    [Fact]
    public void SpecialTokens_Clip_HasStartAndEndOfText()
    {
        var tokens = SpecialTokens.Clip();

        Assert.Equal("<|startoftext|>", tokens.BosToken);
        Assert.Equal("<|endoftext|>", tokens.EosToken);
        Assert.Equal("<|startoftext|>", tokens.ClsToken);
    }

    [Fact]
    public void SpecialTokens_Default_IsBert()
    {
        var defaultTokens = SpecialTokens.Default();
        var bert = SpecialTokens.Bert();

        Assert.Equal(bert.UnkToken, defaultTokens.UnkToken);
        Assert.Equal(bert.PadToken, defaultTokens.PadToken);
        Assert.Equal(bert.ClsToken, defaultTokens.ClsToken);
        Assert.Equal(bert.SepToken, defaultTokens.SepToken);
        Assert.Equal(bert.MaskToken, defaultTokens.MaskToken);
    }

    [Fact]
    public void SpecialTokens_GetAllSpecialTokens_ReturnsNonEmpty()
    {
        var bert = SpecialTokens.Bert();
        var all = bert.GetAllSpecialTokens();

        Assert.True(all.Count > 0);
        Assert.Contains("[UNK]", all);
        Assert.Contains("[PAD]", all);
        Assert.Contains("[CLS]", all);
        Assert.Contains("[SEP]", all);
        Assert.Contains("[MASK]", all);
    }

    [Fact]
    public void SpecialTokens_Bert_GetAll_Count5()
    {
        // BERT has 5 non-empty tokens: UNK, PAD, CLS, SEP, MASK
        // BOS and EOS are empty in BERT
        var bert = SpecialTokens.Bert();
        var all = bert.GetAllSpecialTokens();

        Assert.Equal(5, all.Count);
    }

    [Fact]
    public void SpecialTokens_Gpt_GetAll_Count4()
    {
        // GPT has UNK, PAD, BOS, EOS all set to "<|endoftext|>"
        // But they're all duplicate strings in the list
        var gpt = SpecialTokens.Gpt();
        var all = gpt.GetAllSpecialTokens();

        // 4 non-empty: UnkToken, PadToken, BosToken, EosToken
        Assert.Equal(4, all.Count);
    }

    [Fact]
    public void SpecialTokens_AdditionalTokens_Included()
    {
        var tokens = new SpecialTokens();
        tokens.AdditionalSpecialTokens.Add("<extra_1>");
        tokens.AdditionalSpecialTokens.Add("<extra_2>");

        var all = tokens.GetAllSpecialTokens();
        Assert.Contains("<extra_1>", all);
        Assert.Contains("<extra_2>", all);
    }

    // ============================
    // Vocabulary With From-Dictionary Constructor Edge Cases
    // ============================

    [Fact]
    public void FromDictionary_WithoutUnk_AutoAdds()
    {
        var dict = new Dictionary<string, int>
        {
            { "hello", 0 },
            { "world", 1 }
        };

        var vocab = new Vocabulary(dict);

        // [UNK] should be auto-added
        Assert.True(vocab.ContainsToken("[UNK]"));
        Assert.Equal(3, vocab.Size); // hello, world, [UNK]
    }

    [Fact]
    public void FromDictionary_WithUnk_UsesExistingId()
    {
        var dict = new Dictionary<string, int>
        {
            { "[UNK]", 5 },
            { "hello", 0 },
            { "world", 1 }
        };

        var vocab = new Vocabulary(dict);

        Assert.Equal(5, vocab.GetTokenId("[UNK]"));
        Assert.Equal(3, vocab.Size);
    }

    [Fact]
    public void FromDictionary_EmptyDict_OnlyUnk()
    {
        var dict = new Dictionary<string, int>();
        var vocab = new Vocabulary(dict);

        Assert.Equal(1, vocab.Size);
        Assert.True(vocab.ContainsToken("[UNK]"));
    }

    // ============================
    // Token Encoding/Decoding Consistency
    // ============================

    [Fact]
    public void EncodeDecodeSequence_HandVerified()
    {
        var vocab = new Vocabulary();
        vocab.AddTokens(new[] { "the", "cat", "sat", "on", "mat" });

        // Manually encode: "the cat sat on mat"
        var sentence = new[] { "the", "cat", "sat", "on", "mat" };
        var encoded = sentence.Select(t => vocab.GetTokenId(t)).ToArray();

        // IDs should be 1,2,3,4,5 (0 is [UNK])
        Assert.Equal(new[] { 1, 2, 3, 4, 5 }, encoded);

        // Decode back
        var decoded = encoded.Select(id => vocab.GetToken(id)).ToArray();
        Assert.Equal(sentence, decoded);
    }

    [Fact]
    public void EncodeWithUnknowns_HandVerified()
    {
        var vocab = new Vocabulary();
        vocab.AddTokens(new[] { "the", "cat" });

        // "the dog sat" - "dog" and "sat" are unknown
        var sentence = new[] { "the", "dog", "sat" };
        var encoded = sentence.Select(t => vocab.GetTokenId(t)).ToArray();

        var unkId = vocab.GetTokenId("[UNK]"); // 0
        Assert.Equal(new[] { 1, unkId, unkId }, encoded); // "the"=1, "dog"=UNK, "sat"=UNK
    }
}
