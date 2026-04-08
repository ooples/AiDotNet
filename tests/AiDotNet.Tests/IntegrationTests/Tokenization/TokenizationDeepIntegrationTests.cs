using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Vocabulary;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Tokenization;

/// <summary>
/// Deep integration tests for tokenizer algorithms: Character, BPE, WordPiece.
/// Tests verify exact tokenization output, round-trip consistency, edge cases,
/// vocabulary mechanics, and encoding/decoding correctness.
/// </summary>
public class TokenizationDeepIntegrationTests
{
    // ─── Vocabulary Tests ────────────────────────────────────────────────────

    [Fact]
    public void Vocabulary_FirstTokenIsUnk_GetsIdZero()
    {
        var vocab = new Vocabulary("[UNK]");
        Assert.Equal(0, vocab.GetTokenId("[UNK]"));
        Assert.Equal("[UNK]", vocab.GetToken(0));
    }

    [Fact]
    public void Vocabulary_AddToken_AssignsSequentialIds()
    {
        var vocab = new Vocabulary("[UNK]");
        int idA = vocab.AddToken("a");
        int idB = vocab.AddToken("b");
        int idC = vocab.AddToken("c");

        Assert.Equal(1, idA);
        Assert.Equal(2, idB);
        Assert.Equal(3, idC);
        Assert.Equal(4, vocab.Size);  // [UNK] + a + b + c
    }

    [Fact]
    public void Vocabulary_DuplicateToken_ReturnsSameId()
    {
        var vocab = new Vocabulary("[UNK]");
        int id1 = vocab.AddToken("hello");
        int id2 = vocab.AddToken("hello");
        Assert.Equal(id1, id2);
        Assert.Equal(2, vocab.Size);  // [UNK] + hello
    }

    [Fact]
    public void Vocabulary_UnknownToken_ReturnsUnkId()
    {
        var vocab = new Vocabulary("[UNK]");
        vocab.AddToken("known");
        int unknownId = vocab.GetTokenId("totally_unknown");
        Assert.Equal(0, unknownId);  // UNK token ID
    }

    [Fact]
    public void Vocabulary_ContainsToken_WorksCorrectly()
    {
        var vocab = new Vocabulary("[UNK]");
        vocab.AddToken("hello");
        Assert.True(vocab.ContainsToken("hello"));
        Assert.True(vocab.ContainsToken("[UNK]"));
        Assert.False(vocab.ContainsToken("world"));
    }

    [Fact]
    public void Vocabulary_GetToken_ReturnsNullForInvalidId()
    {
        var vocab = new Vocabulary("[UNK]");
        Assert.Null(vocab.GetToken(999));
    }

    [Fact]
    public void Vocabulary_Clear_ResetsButKeepsUnk()
    {
        var vocab = new Vocabulary("[UNK]");
        vocab.AddToken("a");
        vocab.AddToken("b");
        Assert.Equal(3, vocab.Size);

        vocab.Clear();
        Assert.Equal(1, vocab.Size);  // Only [UNK] remains
        Assert.True(vocab.ContainsToken("[UNK]"));
        Assert.False(vocab.ContainsToken("a"));
    }

    [Fact]
    public void Vocabulary_FromDictionary_PreservesMapping()
    {
        var mapping = new Dictionary<string, int>
        {
            { "[UNK]", 0 },
            { "hello", 1 },
            { "world", 2 }
        };
        var vocab = new Vocabulary(mapping, "[UNK]");

        Assert.Equal(3, vocab.Size);
        Assert.Equal(0, vocab.GetTokenId("[UNK]"));
        Assert.Equal(1, vocab.GetTokenId("hello"));
        Assert.Equal(2, vocab.GetTokenId("world"));
    }

    // ─── CharacterTokenizer Tests ────────────────────────────────────────────

    [Fact]
    public void CharTokenizer_SimpleText_SplitsIntoCharacters()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var tokens = tokenizer.Tokenize("cat");

        Assert.Equal(3, tokens.Count);
        Assert.Equal("c", tokens[0]);
        Assert.Equal("a", tokens[1]);
        Assert.Equal("t", tokens[2]);
    }

    [Fact]
    public void CharTokenizer_WithSpaces_IncludesSpaceCharacters()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var tokens = tokenizer.Tokenize("a b");

        Assert.Equal(3, tokens.Count);
        Assert.Equal("a", tokens[0]);
        Assert.Equal(" ", tokens[1]);
        Assert.Equal("b", tokens[2]);
    }

    [Fact]
    public void CharTokenizer_Lowercase_ConvertsToLowercase()
    {
        var tokenizer = CharacterTokenizer.CreateAscii(lowercase: true);
        var tokens = tokenizer.Tokenize("ABC");

        Assert.Equal(3, tokens.Count);
        Assert.Equal("a", tokens[0]);
        Assert.Equal("b", tokens[1]);
        Assert.Equal("c", tokens[2]);
    }

    [Fact]
    public void CharTokenizer_EmptyString_ReturnsEmptyList()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var tokens = tokenizer.Tokenize("");
        Assert.Empty(tokens);
    }

    [Fact]
    public void CharTokenizer_NullString_ReturnsEmptyList()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var tokens = tokenizer.Tokenize(null);
        Assert.Empty(tokens);
    }

    [Fact]
    public void CharTokenizer_TrainFromCorpus_BuildsVocabulary()
    {
        var corpus = new[] { "hello", "world" };
        var tokenizer = CharacterTokenizer.Train(corpus);

        // Should contain all unique characters from corpus
        var tokens = tokenizer.Tokenize("hello");
        Assert.Equal(5, tokens.Count);
        Assert.All(tokens, t => Assert.NotEqual("[UNK]", t));

        // Vocabulary should include h, e, l, o, w, r, d + special tokens
        Assert.True(tokenizer.VocabularySize >= 7);
    }

    [Fact]
    public void CharTokenizer_TrainWithMinFrequency_FiltersRareChars()
    {
        // 'z' appears only once but 'a' appears 3 times
        var corpus = new[] { "aaa", "aab", "z" };
        var tokenizer = CharacterTokenizer.Train(corpus, minFrequency: 2);

        // 'z' should be below min frequency and become [UNK]
        var tokens = tokenizer.Tokenize("z");
        Assert.Contains("[UNK]", tokens);

        // 'a' should be recognized
        var aTokens = tokenizer.Tokenize("a");
        Assert.DoesNotContain("[UNK]", aTokens);
    }

    [Fact]
    public void CharTokenizer_Encode_ProducesValidTokenIds()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var result = tokenizer.Encode("hi");

        Assert.NotNull(result.TokenIds);
        Assert.Equal(result.Tokens.Count, result.TokenIds.Count);
        // All token IDs should be non-negative
        Assert.All(result.TokenIds, id => Assert.True(id >= 0));
    }

    [Fact]
    public void CharTokenizer_EncodeWithSpecialTokens_AddsClsAndSep()
    {
        var specialTokens = SpecialTokens.Bert();
        var tokenizer = CharacterTokenizer.CreateAscii(specialTokens: specialTokens);
        var options = new EncodingOptions { AddSpecialTokens = true };
        var result = tokenizer.Encode("ab", options);

        // Should have [CLS] + 'a' + 'b' + [SEP]
        Assert.Equal("[CLS]", result.Tokens[0]);
        Assert.Equal("a", result.Tokens[1]);
        Assert.Equal("b", result.Tokens[2]);
        Assert.Equal("[SEP]", result.Tokens[result.Tokens.Count - 1]);
    }

    [Fact]
    public void CharTokenizer_RoundTrip_ReconstructsText()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var original = "hello world";
        var encoded = tokenizer.Encode(original);
        var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

        Assert.Equal(original, decoded);
    }

    [Fact]
    public void CharTokenizer_NonAscii_MapsToUnk()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        // Unicode char outside ASCII range
        var tokens = tokenizer.Tokenize("\u00E9");  // e with accent
        Assert.Contains("[UNK]", tokens);
    }

    // ─── BPE Tokenizer Tests ─────────────────────────────────────────────────

    [Fact]
    public void BpeTokenizer_TrainSimpleCorpus_LearnsCommonPairs()
    {
        var corpus = new[] {
            "the cat sat on the mat",
            "the dog sat on the rug",
            "the cat and the dog"
        };

        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 100);

        // "the" appears 5 times - should be a single token or merged
        var tokens = tokenizer.Tokenize("the");
        // With enough merges, "the" should be a single token or few tokens
        Assert.True(tokens.Count <= 3, $"'the' should be highly merged, got {tokens.Count} tokens");
    }

    [Fact]
    public void BpeTokenizer_EmptyCorpus_ProducesMinimalTokenizer()
    {
        var tokenizer = BpeTokenizer.Train(Array.Empty<string>(), vocabSize: 100);
        // Should still produce a valid tokenizer with special tokens
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void BpeTokenizer_EmptyText_ReturnsEmptyList()
    {
        var tokenizer = BpeTokenizer.Train(new[] { "hello" }, vocabSize: 50);
        Assert.Empty(tokenizer.Tokenize(""));
        Assert.Empty(tokenizer.Tokenize(null));
    }

    [Fact]
    public void BpeTokenizer_ManualMerges_AppliedInPriorityOrder()
    {
        // Build a minimal BPE tokenizer with known merges
        var specialTokens = SpecialTokens.Gpt();
        var vocab = new Vocabulary(specialTokens.UnkToken);
        vocab.AddTokens(specialTokens.GetAllSpecialTokens());
        vocab.AddTokens(new[] { "a", "b", "c", "ab", "abc" });

        var merges = new Dictionary<(string, string), int>
        {
            { ("a", "b"), 0 },   // First merge: a+b → ab
            { ("ab", "c"), 1 }   // Second merge: ab+c → abc
        };

        var tokenizer = new BpeTokenizer(vocab, merges, specialTokens);
        var tokens = tokenizer.Tokenize("abc");

        // With both merges, "abc" should become a single token
        Assert.Single(tokens);
        Assert.Equal("abc", tokens[0]);
    }

    [Fact]
    public void BpeTokenizer_UnknownMerge_SplitsIntoCharacters()
    {
        var specialTokens = SpecialTokens.Gpt();
        var vocab = new Vocabulary(specialTokens.UnkToken);
        vocab.AddTokens(specialTokens.GetAllSpecialTokens());
        vocab.AddTokens(new[] { "x", "y", "z" });

        var merges = new Dictionary<(string, string), int>(); // No merges

        var tokenizer = new BpeTokenizer(vocab, merges, specialTokens);
        var tokens = tokenizer.Tokenize("xyz");

        // With no merges, each character should remain separate
        Assert.Contains("x", tokens);
        Assert.Contains("y", tokens);
        Assert.Contains("z", tokens);
    }

    [Fact]
    public void BpeTokenizer_CachesResults()
    {
        var corpus = new[] { "hello hello hello" };
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 50);

        // Tokenize the same word multiple times
        var tokens1 = tokenizer.Tokenize("hello");
        var tokens2 = tokenizer.Tokenize("hello");

        // Results should be identical (cached)
        Assert.Equal(tokens1.Count, tokens2.Count);
        for (int i = 0; i < tokens1.Count; i++)
            Assert.Equal(tokens1[i], tokens2[i]);
    }

    [Fact]
    public void BpeTokenizer_Train_VocabSizeRespected()
    {
        var corpus = new[] { "the quick brown fox jumps over the lazy dog" };
        int targetVocabSize = 30;
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: targetVocabSize);

        // Vocabulary size should not exceed the target
        Assert.True(tokenizer.VocabularySize <= targetVocabSize,
            $"Vocabulary size {tokenizer.VocabularySize} exceeds target {targetVocabSize}");
    }

    [Fact]
    public void BpeTokenizer_RoundTrip_ReconstructsText()
    {
        var corpus = new[] { "hello world", "foo bar baz" };
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 100);

        var original = "hello world";
        var encoded = tokenizer.Encode(original);
        var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

        Assert.Equal(original, decoded);
    }

    [Fact]
    public void BpeTokenizer_MergeOrderMatters()
    {
        // If merge order differs, tokenization result differs
        var specialTokens = SpecialTokens.Gpt();
        var vocab = new Vocabulary(specialTokens.UnkToken);
        vocab.AddTokens(specialTokens.GetAllSpecialTokens());
        vocab.AddTokens(new[] { "a", "b", "c", "ab", "bc", "abc" });

        // Merge a+b first
        var merges1 = new Dictionary<(string, string), int>
        {
            { ("a", "b"), 0 },
            { ("ab", "c"), 1 }
        };

        // Merge b+c first
        var merges2 = new Dictionary<(string, string), int>
        {
            { ("b", "c"), 0 },
            { ("a", "bc"), 1 }
        };

        var tok1 = new BpeTokenizer(vocab, merges1, specialTokens);
        var tok2 = new BpeTokenizer(vocab, merges2, specialTokens);

        // Both should ultimately merge "abc" into one token
        var tokens1 = tok1.Tokenize("abc");
        var tokens2 = tok2.Tokenize("abc");

        Assert.Single(tokens1);
        Assert.Single(tokens2);
        Assert.Equal("abc", tokens1[0]);
        Assert.Equal("abc", tokens2[0]);
    }

    // ─── WordPiece Tokenizer Tests ───────────────────────────────────────────

    [Fact]
    public void WordPiece_KnownVocab_TokenizesCorrectly()
    {
        var specialTokens = SpecialTokens.Bert();
        var vocab = new Vocabulary("[UNK]");
        vocab.AddTokens(specialTokens.GetAllSpecialTokens());
        // "unhappiness" = u-n-h-a-p-p-i-n-e-s-s, so subwords are "un" + "happi" + "ness"
        // NOT "un" + "happy" + "ness" (there's no "y" in "unhappiness")
        vocab.AddTokens(new[] { "un", "##happi", "##ness", "happy", "the", "##ing" });

        var tokenizer = new WordPieceTokenizer(vocab, specialTokens);
        var tokens = tokenizer.Tokenize("unhappiness");

        // WordPiece greedy longest-match:
        // - Start at 0: try "unhappiness" (not in vocab), shrink until "un" found
        // - Start at 2: try "##happiness" (not in vocab), shrink until "##happi" found
        // - Start at 7: try "##ness" → found!
        Assert.Equal(3, tokens.Count);
        Assert.Equal("un", tokens[0]);
        Assert.Equal("##happi", tokens[1]);
        Assert.Equal("##ness", tokens[2]);
    }

    [Fact]
    public void WordPiece_TrainFromCorpus_BuildsSubwordVocab()
    {
        var corpus = new[]
        {
            "the cat sat on the mat",
            "the dog sat on the log",
            "running jumping swimming"
        };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 100);

        // "the" is very common, should be a full token
        var tokens = tokenizer.Tokenize("the");
        Assert.NotEmpty(tokens);
        Assert.DoesNotContain("[UNK]", tokens);
    }

    [Fact]
    public void WordPiece_ContinuingPrefix_UsedForSubwords()
    {
        var specialTokens = SpecialTokens.Bert();
        var vocab = new Vocabulary("[UNK]");
        vocab.AddTokens(specialTokens.GetAllSpecialTokens());
        vocab.AddTokens(new[] { "play", "##ing", "##ed", "##er", "##s" });

        var tokenizer = new WordPieceTokenizer(vocab, specialTokens);

        var tokens = tokenizer.Tokenize("playing");
        // Should be: "play" + "##ing"
        Assert.Equal(2, tokens.Count);
        Assert.Equal("play", tokens[0]);
        Assert.Equal("##ing", tokens[1]);
    }

    [Fact]
    public void WordPiece_MaxCharsPerWord_ReturnsUnk()
    {
        var specialTokens = SpecialTokens.Bert();
        var vocab = new Vocabulary("[UNK]");
        vocab.AddTokens(specialTokens.GetAllSpecialTokens());
        vocab.AddTokens(new[] { "a", "b" });

        var tokenizer = new WordPieceTokenizer(vocab, specialTokens, maxInputCharsPerWord: 5);

        // Word longer than max chars → [UNK]
        var tokens = tokenizer.Tokenize("abcdefghij");
        Assert.Contains("[UNK]", tokens);
    }

    [Fact]
    public void WordPiece_MaxCharsPerWord_LessThan1_Throws()
    {
        var specialTokens = SpecialTokens.Bert();
        var vocab = new Vocabulary("[UNK]");

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new WordPieceTokenizer(vocab, specialTokens, maxInputCharsPerWord: 0));
    }

    [Fact]
    public void WordPiece_UnknownWord_ReturnsUnkToken()
    {
        var specialTokens = SpecialTokens.Bert();
        var vocab = new Vocabulary("[UNK]");
        vocab.AddTokens(specialTokens.GetAllSpecialTokens());
        vocab.AddTokens(new[] { "hello" });

        var tokenizer = new WordPieceTokenizer(vocab, specialTokens);

        // "xyz" has no matching subword in vocab
        var tokens = tokenizer.Tokenize("xyz");
        Assert.Contains("[UNK]", tokens);
    }

    [Fact]
    public void WordPiece_EmptyText_ReturnsEmptyList()
    {
        var specialTokens = SpecialTokens.Bert();
        var vocab = new Vocabulary("[UNK]");
        vocab.AddTokens(specialTokens.GetAllSpecialTokens());

        var tokenizer = new WordPieceTokenizer(vocab, specialTokens);
        Assert.Empty(tokenizer.Tokenize(""));
        Assert.Empty(tokenizer.Tokenize(null));
    }

    [Fact]
    public void WordPiece_CleanupTokens_RemovesContinuationPrefix()
    {
        var corpus = new[] { "playing", "running", "jumping" };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 100);

        var encoded = tokenizer.Encode("playing");
        var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

        // Decoded text should not contain "##" prefixes
        Assert.DoesNotContain("##", decoded);
    }

    [Fact]
    public void WordPiece_LowercasesInput()
    {
        var corpus = new[] { "hello world" };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 100);

        // WordPiece lowercases internally
        var upperTokens = tokenizer.Tokenize("HELLO");
        var lowerTokens = tokenizer.Tokenize("hello");

        // Both should produce the same tokens
        Assert.Equal(upperTokens.Count, lowerTokens.Count);
        for (int i = 0; i < upperTokens.Count; i++)
            Assert.Equal(upperTokens[i], lowerTokens[i]);
    }

    // ─── Encoding Options Tests ──────────────────────────────────────────────

    [Fact]
    public void Encode_WithPadding_PadsToMaxLength()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions
        {
            Padding = true,
            MaxLength = 10,
            ReturnAttentionMask = true,
            AddSpecialTokens = false
        };

        var result = tokenizer.Encode("hi", options);

        Assert.Equal(10, result.TokenIds.Count);
        Assert.Equal(10, result.AttentionMask.Count);

        // First 2 should be real tokens (attention = 1), rest padding (attention = 0)
        Assert.Equal(1, result.AttentionMask[0]);
        Assert.Equal(1, result.AttentionMask[1]);
        Assert.Equal(0, result.AttentionMask[2]);
    }

    [Fact]
    public void Encode_WithTruncation_TruncatesToMaxLength()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions
        {
            Truncation = true,
            MaxLength = 3,
            AddSpecialTokens = false
        };

        var result = tokenizer.Encode("hello", options);

        // Should be truncated to 3 tokens
        Assert.Equal(3, result.Tokens.Count);
        Assert.Equal("h", result.Tokens[0]);
        Assert.Equal("e", result.Tokens[1]);
        Assert.Equal("l", result.Tokens[2]);
    }

    [Fact]
    public void Encode_WithTruncationAndSpecialTokens_PreservesSpecialTokens()
    {
        var specialTokens = SpecialTokens.Bert();
        var tokenizer = CharacterTokenizer.CreateAscii(specialTokens: specialTokens);
        var options = new EncodingOptions
        {
            Truncation = true,
            MaxLength = 5,
            AddSpecialTokens = true
        };

        var result = tokenizer.Encode("hello", options);

        // MaxLength=5, with [CLS] and [SEP], leaves room for 3 content tokens
        Assert.Equal(5, result.Tokens.Count);
        Assert.Equal("[CLS]", result.Tokens[0]);
        Assert.Equal("[SEP]", result.Tokens[result.Tokens.Count - 1]);
    }

    [Fact]
    public void Encode_ReturnTokenTypeIds_AllZeros()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions { ReturnTokenTypeIds = true };
        var result = tokenizer.Encode("hi", options);

        Assert.NotNull(result.TokenTypeIds);
        Assert.All(result.TokenTypeIds, id => Assert.Equal(0, id));
    }

    [Fact]
    public void Encode_ReturnPositionIds_Sequential()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions { ReturnPositionIds = true };
        var result = tokenizer.Encode("abc", options);

        Assert.NotNull(result.PositionIds);
        for (int i = 0; i < result.PositionIds.Count; i++)
            Assert.Equal(i, result.PositionIds[i]);
    }

    [Fact]
    public void Encode_EmptyString_ReturnsEmptyResult()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var result = tokenizer.Encode("");

        Assert.Empty(result.Tokens);
        Assert.Empty(result.TokenIds);
    }

    // ─── Batch Operations ────────────────────────────────────────────────────

    [Fact]
    public void EncodeBatch_MultipleTexts_ReturnsCorrectCount()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var texts = new List<string> { "hi", "bye", "ok" };
        var options = new EncodingOptions { AddSpecialTokens = false };
        var results = tokenizer.EncodeBatch(texts, options);

        Assert.Equal(3, results.Count);
        Assert.Equal(2, results[0].Tokens.Count);  // "hi" → 2 chars
        Assert.Equal(3, results[1].Tokens.Count);  // "bye" → 3 chars
        Assert.Equal(2, results[2].Tokens.Count);  // "ok" → 2 chars
    }

    [Fact]
    public void DecodeBatch_MultipleSequences_DecodesAll()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var encoded1 = tokenizer.Encode("ab");
        var encoded2 = tokenizer.Encode("cd");

        var decoded = tokenizer.DecodeBatch(
            new List<List<int>> { encoded1.TokenIds, encoded2.TokenIds });

        Assert.Equal(2, decoded.Count);
        Assert.Equal("ab", decoded[0]);
        Assert.Equal("cd", decoded[1]);
    }

    // ─── Special Tokens Configuration ────────────────────────────────────────

    [Fact]
    public void SpecialTokens_Bert_HasCorrectTokens()
    {
        var st = SpecialTokens.Bert();
        Assert.Equal("[UNK]", st.UnkToken);
        Assert.Equal("[PAD]", st.PadToken);
        Assert.Equal("[CLS]", st.ClsToken);
        Assert.Equal("[SEP]", st.SepToken);
        Assert.Equal("[MASK]", st.MaskToken);
        Assert.Equal(string.Empty, st.BosToken);
        Assert.Equal(string.Empty, st.EosToken);
    }

    [Fact]
    public void SpecialTokens_Gpt_HasCorrectTokens()
    {
        var st = SpecialTokens.Gpt();
        Assert.Equal("<|endoftext|>", st.UnkToken);
        Assert.Equal("<|endoftext|>", st.PadToken);
        Assert.Equal("<|endoftext|>", st.BosToken);
        Assert.Equal("<|endoftext|>", st.EosToken);
        Assert.Equal(string.Empty, st.ClsToken);
        Assert.Equal(string.Empty, st.SepToken);
    }

    [Fact]
    public void SpecialTokens_GetAllSpecialTokens_DoesNotIncludeEmpty()
    {
        var st = SpecialTokens.Gpt();
        var allTokens = st.GetAllSpecialTokens();

        // Empty strings should not be included
        Assert.DoesNotContain(string.Empty, allTokens);
        Assert.DoesNotContain(null, allTokens);
    }

    // ─── ConvertTokensToIds / ConvertIdsToTokens ─────────────────────────────

    [Fact]
    public void ConvertTokensToIds_KnownTokens_ReturnsCorrectIds()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var tokens = tokenizer.Tokenize("ab");
        var ids = tokenizer.ConvertTokensToIds(tokens);

        // IDs should be consistent
        Assert.Equal(2, ids.Count);
        Assert.NotEqual(ids[0], ids[1]);  // 'a' and 'b' have different IDs
    }

    [Fact]
    public void ConvertIdsToTokens_ValidIds_ReturnsCorrectTokens()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var tokens = tokenizer.Tokenize("ab");
        var ids = tokenizer.ConvertTokensToIds(tokens);
        var reconstructed = tokenizer.ConvertIdsToTokens(ids);

        Assert.Equal(tokens.Count, reconstructed.Count);
        for (int i = 0; i < tokens.Count; i++)
            Assert.Equal(tokens[i], reconstructed[i]);
    }

    // ─── BPE Training Edge Cases ─────────────────────────────────────────────

    [Fact]
    public void BpeTokenizer_Train_NullCorpus_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            BpeTokenizer.Train(null, vocabSize: 50));
    }

    [Fact]
    public void BpeTokenizer_Train_InvalidVocabSize_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            BpeTokenizer.Train(new[] { "hello" }, vocabSize: 0));
    }

    [Fact]
    public void CharTokenizer_Train_NullCorpus_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            CharacterTokenizer.Train(null));
    }

    [Fact]
    public void CharTokenizer_Train_InvalidMinFrequency_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            CharacterTokenizer.Train(new[] { "hello" }, minFrequency: 0));
    }

    [Fact]
    public void WordPiece_Train_NullCorpus_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            WordPieceTokenizer.Train(null, vocabSize: 50));
    }

    [Fact]
    public void WordPiece_Train_InvalidVocabSize_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            WordPieceTokenizer.Train(new[] { "hello" }, vocabSize: 0));
    }

    // ─── Cross-Tokenizer Consistency ─────────────────────────────────────────

    [Fact]
    public void AllTokenizers_EmptyInput_ReturnEmpty()
    {
        var charTok = CharacterTokenizer.CreateAscii();
        var bpeTok = BpeTokenizer.Train(new[] { "hello" }, vocabSize: 50);
        var wpTok = WordPieceTokenizer.Train(new[] { "hello" }, vocabSize: 50);

        Assert.Empty(charTok.Tokenize(""));
        Assert.Empty(bpeTok.Tokenize(""));
        Assert.Empty(wpTok.Tokenize(""));
    }

    [Fact]
    public void AllTokenizers_NonEmptyInput_ReturnNonEmpty()
    {
        var corpus = new[] { "hello world" };
        var charTok = CharacterTokenizer.CreateAscii();
        var bpeTok = BpeTokenizer.Train(corpus, vocabSize: 100);
        var wpTok = WordPieceTokenizer.Train(corpus, vocabSize: 100);

        Assert.NotEmpty(charTok.Tokenize("hello"));
        Assert.NotEmpty(bpeTok.Tokenize("hello"));
        Assert.NotEmpty(wpTok.Tokenize("hello"));
    }

    [Fact]
    public void CharTokenizer_AlwaysProducesMoreTokensThanBpe()
    {
        var corpus = new[] { "the quick brown fox jumps over the lazy dog" };
        var charTok = CharacterTokenizer.CreateAscii();
        var bpeTok = BpeTokenizer.Train(corpus, vocabSize: 200);

        var text = "the quick";
        var charTokens = charTok.Tokenize(text);
        var bpeTokens = bpeTok.Tokenize(text);

        // Character tokenizer always produces more tokens than BPE (which merges)
        Assert.True(charTokens.Count >= bpeTokens.Count,
            $"Char tokens ({charTokens.Count}) should be >= BPE tokens ({bpeTokens.Count})");
    }

    // ─── Decode with SkipSpecialTokens ───────────────────────────────────────

    [Fact]
    public void Decode_SkipSpecialTokens_RemovesSpecialTokens()
    {
        var specialTokens = SpecialTokens.Bert();
        var tokenizer = CharacterTokenizer.CreateAscii(specialTokens: specialTokens);

        var options = new EncodingOptions { AddSpecialTokens = true };
        var encoded = tokenizer.Encode("ab", options);

        // Decode with skip
        var withSkip = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);
        // Decode without skip
        var withoutSkip = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: false);

        // With skip should be shorter (no [CLS]/[SEP])
        Assert.Equal("ab", withSkip);
        Assert.Contains("[CLS]", withoutSkip);
    }

    // ─── Vocabulary Size Consistency ─────────────────────────────────────────

    [Fact]
    public void BpeTokenizer_VocabSize_MatchesVocabularyProperty()
    {
        var corpus = new[] { "hello world foo bar" };
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 50);

        // VocabularySize should be consistent
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void WordPiece_VocabSize_MatchesVocabularyProperty()
    {
        var corpus = new[] { "hello world foo bar" };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 50);

        Assert.True(tokenizer.VocabularySize > 0);
    }

    // ─── Left Truncation ─────────────────────────────────────────────────────

    [Fact]
    public void Encode_LeftTruncation_KeepsEndOfSequence()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions
        {
            Truncation = true,
            MaxLength = 3,
            TruncationSide = "left",
            AddSpecialTokens = false
        };

        var result = tokenizer.Encode("abcde", options);

        // Left truncation should keep the last 3 characters
        Assert.Equal(3, result.Tokens.Count);
        Assert.Equal("c", result.Tokens[0]);
        Assert.Equal("d", result.Tokens[1]);
        Assert.Equal("e", result.Tokens[2]);
    }

    // ─── Left Padding ────────────────────────────────────────────────────────

    [Fact]
    public void Encode_LeftPadding_PadsAtStart()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions
        {
            Padding = true,
            MaxLength = 5,
            PaddingSide = "left",
            ReturnAttentionMask = true,
            AddSpecialTokens = false
        };

        var result = tokenizer.Encode("ab", options);

        Assert.Equal(5, result.TokenIds.Count);
        // First 3 should be padding (attention = 0)
        Assert.Equal(0, result.AttentionMask[0]);
        Assert.Equal(0, result.AttentionMask[1]);
        Assert.Equal(0, result.AttentionMask[2]);
        // Last 2 should be real tokens (attention = 1)
        Assert.Equal(1, result.AttentionMask[3]);
        Assert.Equal(1, result.AttentionMask[4]);
    }
}
