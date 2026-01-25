using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Configuration;
using AiDotNet.Tokenization.Core;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Specialized;
using AiDotNet.Tokenization.Vocabulary;

namespace AiDotNet.Tests.IntegrationTests.Tokenization;

/// <summary>
/// Comprehensive integration tests for the Tokenization module.
/// Tests cover all tokenizer implementations, vocabulary management, encoding options,
/// and specialized tokenizers (MIDI, Phoneme).
/// </summary>
public class TokenizationIntegrationTests
{
    #region Vocabulary Tests

    [Fact]
    public void Vocabulary_Creation_InitializesWithUnkToken()
    {
        var vocabulary = new Vocabulary();

        Assert.Equal(1, vocabulary.Size);
        Assert.True(vocabulary.ContainsToken("[UNK]"));
        Assert.Equal(0, vocabulary.GetTokenId("[UNK]"));
    }

    [Fact]
    public void Vocabulary_CustomUnkToken_Works()
    {
        var vocabulary = new Vocabulary("<unk>");

        Assert.True(vocabulary.ContainsToken("<unk>"));
        Assert.Equal(0, vocabulary.GetTokenId("<unk>"));
    }

    [Fact]
    public void Vocabulary_AddToken_AssignsIncrementalIds()
    {
        var vocabulary = new Vocabulary();

        var id1 = vocabulary.AddToken("hello");
        var id2 = vocabulary.AddToken("world");

        Assert.Equal(1, id1);
        Assert.Equal(2, id2);
        Assert.Equal(3, vocabulary.Size);
    }

    [Fact]
    public void Vocabulary_AddToken_ReturnsSameIdForDuplicates()
    {
        var vocabulary = new Vocabulary();

        var id1 = vocabulary.AddToken("test");
        var id2 = vocabulary.AddToken("test");

        Assert.Equal(id1, id2);
        Assert.Equal(2, vocabulary.Size); // [UNK] + "test"
    }

    [Fact]
    public void Vocabulary_AddTokens_AddsMultipleTokens()
    {
        var vocabulary = new Vocabulary();
        var tokens = new[] { "a", "b", "c", "d", "e" };

        vocabulary.AddTokens(tokens);

        Assert.Equal(6, vocabulary.Size); // [UNK] + 5 tokens
        foreach (var token in tokens)
        {
            Assert.True(vocabulary.ContainsToken(token));
        }
    }

    [Fact]
    public void Vocabulary_GetTokenId_ReturnsUnkForMissingToken()
    {
        var vocabulary = new Vocabulary();
        vocabulary.AddToken("known");

        var unknownId = vocabulary.GetTokenId("unknown_token");
        var unkTokenId = vocabulary.GetTokenId("[UNK]");

        Assert.Equal(unkTokenId, unknownId);
    }

    [Fact]
    public void Vocabulary_GetToken_ReturnsNullForInvalidId()
    {
        var vocabulary = new Vocabulary();

        var token = vocabulary.GetToken(999);

        Assert.Null(token);
    }

    [Fact]
    public void Vocabulary_GetToken_ReturnsCorrectToken()
    {
        var vocabulary = new Vocabulary();
        var id = vocabulary.AddToken("hello");

        var token = vocabulary.GetToken(id);

        Assert.Equal("hello", token);
    }

    [Fact]
    public void Vocabulary_ContainsId_ReturnsTrueForValidId()
    {
        var vocabulary = new Vocabulary();
        vocabulary.AddToken("test");

        Assert.True(vocabulary.ContainsId(0)); // [UNK]
        Assert.True(vocabulary.ContainsId(1)); // "test"
        Assert.False(vocabulary.ContainsId(999));
    }

    [Fact]
    public void Vocabulary_GetAllTokens_ReturnsAllTokens()
    {
        var vocabulary = new Vocabulary();
        vocabulary.AddTokens(new[] { "a", "b", "c" });

        var allTokens = vocabulary.GetAllTokens().ToList();

        Assert.Equal(4, allTokens.Count);
        Assert.Contains("[UNK]", allTokens);
        Assert.Contains("a", allTokens);
        Assert.Contains("b", allTokens);
        Assert.Contains("c", allTokens);
    }

    [Fact]
    public void Vocabulary_Clear_ResetsVocabulary()
    {
        var vocabulary = new Vocabulary();
        vocabulary.AddTokens(new[] { "a", "b", "c", "d", "e" });
        Assert.Equal(6, vocabulary.Size);

        vocabulary.Clear();

        Assert.Equal(1, vocabulary.Size); // Only [UNK] remains
        Assert.True(vocabulary.ContainsToken("[UNK]"));
        Assert.False(vocabulary.ContainsToken("a"));
    }

    [Fact]
    public void Vocabulary_TokenToId_ReturnsCorrectMappings()
    {
        var vocabulary = new Vocabulary();
        vocabulary.AddTokens(new[] { "x", "y", "z" });

        var mapping = vocabulary.TokenToId;

        Assert.Equal(0, mapping["[UNK]"]);
        Assert.True(mapping.ContainsKey("x"));
        Assert.True(mapping.ContainsKey("y"));
        Assert.True(mapping.ContainsKey("z"));
    }

    [Fact]
    public void Vocabulary_IdToToken_ReturnsCorrectMappings()
    {
        var vocabulary = new Vocabulary();
        vocabulary.AddTokens(new[] { "x", "y" });

        var mapping = vocabulary.IdToToken;

        Assert.Equal("[UNK]", mapping[0]);
        Assert.Equal("x", mapping[1]);
        Assert.Equal("y", mapping[2]);
    }

    [Fact]
    public void Vocabulary_FromDictionary_InitializesCorrectly()
    {
        var tokenToId = new Dictionary<string, int>
        {
            { "hello", 0 },
            { "world", 1 },
            { "[UNK]", 2 }
        };

        var vocabulary = new Vocabulary(tokenToId, "[UNK]");

        Assert.Equal(3, vocabulary.Size);
        Assert.Equal(0, vocabulary.GetTokenId("hello"));
        Assert.Equal(1, vocabulary.GetTokenId("world"));
        Assert.Equal(2, vocabulary.GetTokenId("[UNK]"));
    }

    [Fact]
    public void Vocabulary_AddToken_ThrowsOnNullOrEmpty()
    {
        var vocabulary = new Vocabulary();

        Assert.Throws<ArgumentException>(() => vocabulary.AddToken(null!));
        Assert.Throws<ArgumentException>(() => vocabulary.AddToken(""));
    }

    #endregion

    #region SpecialTokens Tests

    [Fact]
    public void SpecialTokens_Default_UsesBertStyle()
    {
        var tokens = SpecialTokens.Default();

        Assert.Equal("[UNK]", tokens.UnkToken);
        Assert.Equal("[PAD]", tokens.PadToken);
        Assert.Equal("[CLS]", tokens.ClsToken);
        Assert.Equal("[SEP]", tokens.SepToken);
        Assert.Equal("[MASK]", tokens.MaskToken);
    }

    [Fact]
    public void SpecialTokens_Bert_ConfiguresCorrectly()
    {
        var tokens = SpecialTokens.Bert();

        Assert.Equal("[UNK]", tokens.UnkToken);
        Assert.Equal("[PAD]", tokens.PadToken);
        Assert.Equal("[CLS]", tokens.ClsToken);
        Assert.Equal("[SEP]", tokens.SepToken);
        Assert.Equal("[MASK]", tokens.MaskToken);
        Assert.Equal(string.Empty, tokens.BosToken);
        Assert.Equal(string.Empty, tokens.EosToken);
    }

    [Fact]
    public void SpecialTokens_Gpt_ConfiguresCorrectly()
    {
        var tokens = SpecialTokens.Gpt();

        Assert.Equal("<|endoftext|>", tokens.UnkToken);
        Assert.Equal("<|endoftext|>", tokens.PadToken);
        Assert.Equal("<|endoftext|>", tokens.BosToken);
        Assert.Equal("<|endoftext|>", tokens.EosToken);
        Assert.Equal(string.Empty, tokens.ClsToken);
        Assert.Equal(string.Empty, tokens.SepToken);
    }

    [Fact]
    public void SpecialTokens_T5_ConfiguresCorrectly()
    {
        var tokens = SpecialTokens.T5();

        Assert.Equal("<unk>", tokens.UnkToken);
        Assert.Equal("<pad>", tokens.PadToken);
        Assert.Equal("</s>", tokens.EosToken);
    }

    [Fact]
    public void SpecialTokens_Clip_ConfiguresCorrectly()
    {
        var tokens = SpecialTokens.Clip();

        Assert.Equal("<|endoftext|>", tokens.UnkToken);
        Assert.Equal("<|startoftext|>", tokens.BosToken);
        Assert.Equal("<|endoftext|>", tokens.EosToken);
    }

    [Fact]
    public void SpecialTokens_GetAllSpecialTokens_ReturnsAllNonEmpty()
    {
        var tokens = SpecialTokens.Bert();

        var allTokens = tokens.GetAllSpecialTokens();

        Assert.Contains("[UNK]", allTokens);
        Assert.Contains("[PAD]", allTokens);
        Assert.Contains("[CLS]", allTokens);
        Assert.Contains("[SEP]", allTokens);
        Assert.Contains("[MASK]", allTokens);
        Assert.DoesNotContain(string.Empty, allTokens);
    }

    [Fact]
    public void SpecialTokens_AdditionalTokens_AreIncluded()
    {
        var tokens = new SpecialTokens
        {
            UnkToken = "[UNK]",
            AdditionalSpecialTokens = new List<string> { "[CUSTOM1]", "[CUSTOM2]" }
        };

        var allTokens = tokens.GetAllSpecialTokens();

        Assert.Contains("[CUSTOM1]", allTokens);
        Assert.Contains("[CUSTOM2]", allTokens);
    }

    #endregion

    #region EncodingOptions Tests

    [Fact]
    public void EncodingOptions_DefaultValues_AreCorrect()
    {
        var options = new EncodingOptions();

        Assert.True(options.AddSpecialTokens);
        Assert.Null(options.MaxLength);
        Assert.False(options.Padding);
        Assert.Equal("right", options.PaddingSide);
        Assert.False(options.Truncation);
        Assert.Equal("right", options.TruncationSide);
        Assert.True(options.ReturnAttentionMask);
        Assert.False(options.ReturnTokenTypeIds);
        Assert.False(options.ReturnPositionIds);
        Assert.False(options.ReturnOffsets);
    }

    [Fact]
    public void EncodingOptions_CanBeConfigured()
    {
        var options = new EncodingOptions
        {
            AddSpecialTokens = false,
            MaxLength = 512,
            Padding = true,
            PaddingSide = "left",
            Truncation = true,
            TruncationSide = "left",
            ReturnTokenTypeIds = true,
            ReturnPositionIds = true
        };

        Assert.False(options.AddSpecialTokens);
        Assert.Equal(512, options.MaxLength);
        Assert.True(options.Padding);
        Assert.Equal("left", options.PaddingSide);
        Assert.True(options.Truncation);
        Assert.Equal("left", options.TruncationSide);
        Assert.True(options.ReturnTokenTypeIds);
        Assert.True(options.ReturnPositionIds);
    }

    #endregion

    #region TokenizationResult Tests

    [Fact]
    public void TokenizationResult_DefaultConstructor_CreatesEmptyLists()
    {
        var result = new TokenizationResult();

        Assert.Empty(result.TokenIds);
        Assert.Empty(result.Tokens);
        Assert.Empty(result.AttentionMask);
        Assert.Empty(result.TokenTypeIds);
        Assert.Empty(result.PositionIds);
        Assert.Empty(result.Offsets);
        Assert.Empty(result.Metadata);
    }

    [Fact]
    public void TokenizationResult_ParameterizedConstructor_InitializesCorrectly()
    {
        var tokens = new List<string> { "hello", "world" };
        var tokenIds = new List<int> { 1, 2 };

        var result = new TokenizationResult(tokens, tokenIds);

        Assert.Equal(tokens, result.Tokens);
        Assert.Equal(tokenIds, result.TokenIds);
        Assert.Equal(new List<int> { 1, 1 }, result.AttentionMask);
    }

    [Fact]
    public void TokenizationResult_ParameterizedConstructor_ThrowsOnMismatch()
    {
        var tokens = new List<string> { "hello", "world" };
        var tokenIds = new List<int> { 1 }; // Mismatched count

        Assert.Throws<ArgumentException>(() => new TokenizationResult(tokens, tokenIds));
    }

    [Fact]
    public void TokenizationResult_Length_ReturnsNonPaddedCount()
    {
        var result = new TokenizationResult
        {
            TokenIds = new List<int> { 1, 2, 3, 0, 0 },
            AttentionMask = new List<int> { 1, 1, 1, 0, 0 }
        };

        Assert.Equal(3, result.Length);
    }

    [Fact]
    public void TokenizationResult_TotalLength_ReturnsTotalCount()
    {
        var result = new TokenizationResult
        {
            TokenIds = new List<int> { 1, 2, 3, 0, 0 }
        };

        Assert.Equal(5, result.TotalLength);
    }

    #endregion

    #region TokenizationConfig Tests

    [Fact]
    public void TokenizationConfig_DefaultValues_AreCorrect()
    {
        var config = new TokenizationConfig();

        Assert.True(config.AddSpecialTokens);
        Assert.Null(config.MaxLength);
        Assert.False(config.Padding);
        Assert.False(config.Truncation);
        Assert.Equal("right", config.PaddingSide);
        Assert.Equal("right", config.TruncationSide);
        Assert.True(config.ReturnAttentionMask);
        Assert.False(config.ReturnTokenTypeIds);
        Assert.False(config.EnableCaching);
        Assert.True(config.EnableParallelBatchProcessing);
        Assert.Equal(32, config.ParallelBatchThreshold);
    }

    [Fact]
    public void TokenizationConfig_ForBert_ConfiguresCorrectly()
    {
        var config = TokenizationConfig.ForBert(256);

        Assert.Equal(256, config.MaxLength);
        Assert.True(config.AddSpecialTokens);
        Assert.True(config.Padding);
        Assert.True(config.Truncation);
        Assert.True(config.ReturnAttentionMask);
        Assert.True(config.ReturnTokenTypeIds);
    }

    [Fact]
    public void TokenizationConfig_ForGpt_ConfiguresCorrectly()
    {
        var config = TokenizationConfig.ForGpt(2048);

        Assert.Equal(2048, config.MaxLength);
        Assert.False(config.AddSpecialTokens);
        Assert.False(config.Padding);
        Assert.True(config.Truncation);
        Assert.Equal("left", config.TruncationSide);
        Assert.True(config.ReturnAttentionMask);
        Assert.False(config.ReturnTokenTypeIds);
    }

    [Fact]
    public void TokenizationConfig_ForCode_ConfiguresCorrectly()
    {
        var config = TokenizationConfig.ForCode();

        Assert.Equal(2048, config.MaxLength);
        Assert.True(config.AddSpecialTokens);
        Assert.False(config.Padding);
        Assert.True(config.Truncation);
        Assert.True(config.ReturnAttentionMask);
        Assert.False(config.ReturnTokenTypeIds);
    }

    [Fact]
    public void TokenizationConfig_ToEncodingOptions_TransfersCorrectly()
    {
        var config = new TokenizationConfig
        {
            AddSpecialTokens = false,
            MaxLength = 100,
            Padding = true,
            Truncation = true,
            PaddingSide = "left",
            TruncationSide = "left",
            ReturnAttentionMask = false,
            ReturnTokenTypeIds = true
        };

        var options = config.ToEncodingOptions();

        Assert.False(options.AddSpecialTokens);
        Assert.Equal(100, options.MaxLength);
        Assert.True(options.Padding);
        Assert.True(options.Truncation);
        Assert.Equal("left", options.PaddingSide);
        Assert.Equal("left", options.TruncationSide);
        Assert.False(options.ReturnAttentionMask);
        Assert.True(options.ReturnTokenTypeIds);
    }

    #endregion

    #region CharacterTokenizer Tests

    [Fact]
    public void CharacterTokenizer_CreateAscii_CreatesValidTokenizer()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();

        Assert.NotNull(tokenizer);
        Assert.NotNull(tokenizer.Vocabulary);
        Assert.True(tokenizer.VocabularySize > 90); // ASCII + special tokens
    }

    [Fact]
    public void CharacterTokenizer_Tokenize_SplitsIntoCharacters()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();

        var tokens = tokenizer.Tokenize("Hello");

        Assert.Equal(new List<string> { "H", "e", "l", "l", "o" }, tokens);
    }

    [Fact]
    public void CharacterTokenizer_Tokenize_HandlesCaseSensitivity()
    {
        var tokenizer = CharacterTokenizer.CreateAscii(lowercase: true);

        var tokens = tokenizer.Tokenize("HeLLo");

        Assert.Equal(new List<string> { "h", "e", "l", "l", "o" }, tokens);
    }

    [Fact]
    public void CharacterTokenizer_Tokenize_EmptyString_ReturnsEmptyList()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();

        var tokens = tokenizer.Tokenize("");

        Assert.Empty(tokens);
    }

    [Fact]
    public void CharacterTokenizer_Tokenize_NullString_ReturnsEmptyList()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();

        var tokens = tokenizer.Tokenize(null!);

        Assert.Empty(tokens);
    }

    [Fact]
    public void CharacterTokenizer_Encode_ReturnsValidResult()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();

        var result = tokenizer.Encode("Hi");

        Assert.NotNull(result);
        Assert.Equal(4, result.Tokens.Count); // [CLS] + H + i + [SEP]
        Assert.Equal(4, result.TokenIds.Count);
    }

    [Fact]
    public void CharacterTokenizer_Encode_WithoutSpecialTokens()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions { AddSpecialTokens = false };

        var result = tokenizer.Encode("Hi", options);

        Assert.Equal(2, result.Tokens.Count); // H + i only
        Assert.Equal(new List<string> { "H", "i" }, result.Tokens);
    }

    [Fact]
    public void CharacterTokenizer_Encode_WithPadding()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions
        {
            AddSpecialTokens = false,
            Padding = true,
            MaxLength = 10
        };

        var result = tokenizer.Encode("Hi", options);

        Assert.Equal(10, result.TokenIds.Count);
        Assert.Equal(2, result.AttentionMask.Take(2).Sum()); // 2 real tokens
        Assert.Equal(0, result.AttentionMask.Skip(2).Sum()); // Rest is padding
    }

    [Fact]
    public void CharacterTokenizer_Encode_WithTruncation()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions
        {
            AddSpecialTokens = false,
            Truncation = true,
            MaxLength = 3
        };

        var result = tokenizer.Encode("Hello World", options);

        Assert.Equal(3, result.Tokens.Count);
    }

    [Fact]
    public void CharacterTokenizer_Decode_ReconstructsText()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var encoded = tokenizer.Encode("Test", new EncodingOptions { AddSpecialTokens = false });

        var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

        Assert.Equal("Test", decoded);
    }

    [Fact]
    public void CharacterTokenizer_EncodeBatch_EncodesMultipleTexts()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var texts = new List<string> { "Hi", "Bye" };
        var options = new EncodingOptions { AddSpecialTokens = false };

        var results = tokenizer.EncodeBatch(texts, options);

        Assert.Equal(2, results.Count);
        Assert.Equal(2, results[0].Tokens.Count);
        Assert.Equal(3, results[1].Tokens.Count);
    }

    [Fact]
    public void CharacterTokenizer_DecodeBatch_DecodesMultipleSequences()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var encoded1 = tokenizer.Encode("Hi", new EncodingOptions { AddSpecialTokens = false });
        var encoded2 = tokenizer.Encode("Bye", new EncodingOptions { AddSpecialTokens = false });

        var decoded = tokenizer.DecodeBatch(new List<List<int>> { encoded1.TokenIds, encoded2.TokenIds });

        Assert.Equal(2, decoded.Count);
        Assert.Equal("Hi", decoded[0]);
        Assert.Equal("Bye", decoded[1]);
    }

    [Fact]
    public void CharacterTokenizer_ConvertTokensToIds_WorksCorrectly()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var tokens = new List<string> { "A", "B", "C" };

        var ids = tokenizer.ConvertTokensToIds(tokens);

        Assert.Equal(3, ids.Count);
        Assert.All(ids, id => Assert.True(id >= 0));
    }

    [Fact]
    public void CharacterTokenizer_ConvertIdsToTokens_WorksCorrectly()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var tokens = new List<string> { "X", "Y", "Z" };
        var ids = tokenizer.ConvertTokensToIds(tokens);

        var convertedTokens = tokenizer.ConvertIdsToTokens(ids);

        Assert.Equal(tokens, convertedTokens);
    }

    [Fact]
    public void CharacterTokenizer_Train_CreatesFromCorpus()
    {
        var corpus = new[] { "hello world", "hello there", "world peace" };

        var tokenizer = CharacterTokenizer.Train(corpus);

        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.Vocabulary.ContainsToken("h"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("e"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("l"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("o"));
    }

    [Fact]
    public void CharacterTokenizer_Train_WithMinFrequency_FiltersRareChars()
    {
        var corpus = new[] { "aaaa bbbb cccc x" }; // 'x' appears only once

        var tokenizer = CharacterTokenizer.Train(corpus, minFrequency: 2);

        // 'x' appears only once, so it should be filtered out
        Assert.False(tokenizer.Vocabulary.ContainsToken("x"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("a"));
    }

    [Fact]
    public void CharacterTokenizer_Train_ThrowsOnNullCorpus()
    {
        Assert.Throws<ArgumentNullException>(() => CharacterTokenizer.Train(null!));
    }

    [Fact]
    public void CharacterTokenizer_UnknownCharacter_MapsToUnk()
    {
        var vocabulary = new Vocabulary();
        vocabulary.AddTokens(new[] { "a", "b", "c" });
        var specialTokens = SpecialTokens.Default();
        var tokenizer = new CharacterTokenizer(vocabulary, specialTokens);

        var tokens = tokenizer.Tokenize("xyz"); // x, y, z not in vocabulary

        Assert.All(tokens, t => Assert.Equal("[UNK]", t));
    }

    #endregion

    #region WordPieceTokenizer Tests

    [Fact]
    public void WordPieceTokenizer_Train_CreatesValidTokenizer()
    {
        var corpus = new[] { "hello world", "hello there", "world peace" };

        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 100);

        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void WordPieceTokenizer_Tokenize_SplitsIntoSubwords()
    {
        var corpus = new[] { "hello world", "hello there" };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 50);

        var tokens = tokenizer.Tokenize("hello");

        Assert.NotEmpty(tokens);
        // Should tokenize into subwords
    }

    [Fact]
    public void WordPieceTokenizer_Tokenize_EmptyString_ReturnsEmptyList()
    {
        var corpus = new[] { "hello world" };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 50);

        var tokens = tokenizer.Tokenize("");

        Assert.Empty(tokens);
    }

    [Fact]
    public void WordPieceTokenizer_Tokenize_HandlesUnknownWords()
    {
        var corpus = new[] { "hello world" };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 50);

        // Token for very long unknown word
        var tokens = tokenizer.Tokenize("supercalifragilisticexpialidociousextralongwordthatexceedsmaxchars".PadRight(200, 'x'));

        Assert.Contains("[UNK]", tokens);
    }

    [Fact]
    public void WordPieceTokenizer_Encode_ReturnsValidResult()
    {
        var corpus = new[] { "hello world", "hello there" };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 50);

        var result = tokenizer.Encode("hello");

        Assert.NotNull(result);
        Assert.NotEmpty(result.TokenIds);
    }

    [Fact]
    public void WordPieceTokenizer_Decode_ReconstructsText()
    {
        var corpus = new[] { "hello world", "hello there" };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 50);
        var options = new EncodingOptions { AddSpecialTokens = false };
        var encoded = tokenizer.Encode("hello", options);

        var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

        Assert.NotEmpty(decoded);
    }

    [Fact]
    public void WordPieceTokenizer_UsesBertSpecialTokensByDefault()
    {
        var corpus = new[] { "hello world" };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 50);

        Assert.Equal("[UNK]", tokenizer.SpecialTokens.UnkToken);
        Assert.Equal("[CLS]", tokenizer.SpecialTokens.ClsToken);
        Assert.Equal("[SEP]", tokenizer.SpecialTokens.SepToken);
    }

    [Fact]
    public void WordPieceTokenizer_ContinuingSubwordPrefix_IsConfigurable()
    {
        var corpus = new[] { "hello world" };
        var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 50, continuingSubwordPrefix: "@@");

        // The tokenizer should use @@ prefix for subwords
        Assert.NotNull(tokenizer);
    }

    #endregion

    #region BpeTokenizer Tests

    [Fact]
    public void BpeTokenizer_Train_CreatesValidTokenizer()
    {
        var corpus = new[] { "hello world", "hello there", "world peace" };

        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 100);

        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void BpeTokenizer_Train_EmptyCorpus_CreatesMinimalTokenizer()
    {
        var corpus = Array.Empty<string>();

        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 100);

        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0); // At least special tokens
    }

    [Fact]
    public void BpeTokenizer_Train_ThrowsOnNullCorpus()
    {
        Assert.Throws<ArgumentNullException>(() => BpeTokenizer.Train(null!, vocabSize: 100));
    }

    [Fact]
    public void BpeTokenizer_Tokenize_SplitsIntoSubwords()
    {
        var corpus = new[] { "hello world", "hello there" };
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 50);

        var tokens = tokenizer.Tokenize("hello");

        Assert.NotEmpty(tokens);
    }

    [Fact]
    public void BpeTokenizer_Tokenize_EmptyString_ReturnsEmptyList()
    {
        var corpus = new[] { "hello world" };
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 50);

        var tokens = tokenizer.Tokenize("");

        Assert.Empty(tokens);
    }

    [Fact]
    public void BpeTokenizer_Tokenize_NullString_ReturnsEmptyList()
    {
        var corpus = new[] { "hello world" };
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 50);

        var tokens = tokenizer.Tokenize(null!);

        Assert.Empty(tokens);
    }

    [Fact]
    public void BpeTokenizer_Encode_ReturnsValidResult()
    {
        var corpus = new[] { "hello world", "hello there" };
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 50);

        var result = tokenizer.Encode("hello world");

        Assert.NotNull(result);
        Assert.NotEmpty(result.TokenIds);
    }

    [Fact]
    public void BpeTokenizer_Decode_ReconstructsText()
    {
        var corpus = new[] { "hello world", "hello there" };
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 50);
        var options = new EncodingOptions { AddSpecialTokens = false };
        var encoded = tokenizer.Encode("hello", options);

        var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

        Assert.NotEmpty(decoded);
    }

    [Fact]
    public void BpeTokenizer_UsesGptSpecialTokensByDefault()
    {
        var corpus = new[] { "hello world" };
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 50);

        Assert.Equal("<|endoftext|>", tokenizer.SpecialTokens.UnkToken);
    }

    [Fact]
    public void BpeTokenizer_CachesPreviouslyTokenizedWords()
    {
        var corpus = new[] { "hello world", "hello there" };
        var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 50);

        // First tokenization
        var tokens1 = tokenizer.Tokenize("hello");
        // Second tokenization should use cache
        var tokens2 = tokenizer.Tokenize("hello");

        Assert.Equal(tokens1, tokens2);
    }

    #endregion

    #region MidiTokenizer Tests

    [Fact]
    public void MidiTokenizer_CreateREMI_CreatesValidTokenizer()
    {
        var tokenizer = MidiTokenizer.CreateREMI();

        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 100);
    }

    [Fact]
    public void MidiTokenizer_CreateCPWord_CreatesValidTokenizer()
    {
        var tokenizer = MidiTokenizer.CreateCPWord();

        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 100);
    }

    [Fact]
    public void MidiTokenizer_CreateSimpleNote_CreatesValidTokenizer()
    {
        var tokenizer = MidiTokenizer.CreateSimpleNote();

        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 100);
    }

    [Fact]
    public void MidiTokenizer_Tokenize_NoteEvent_REMI()
    {
        var tokenizer = MidiTokenizer.CreateREMI();

        var tokens = tokenizer.Tokenize("NOTE:60:480:64");

        Assert.NotEmpty(tokens);
        Assert.Contains(tokens, t => t.StartsWith("Pitch_"));
        Assert.Contains(tokens, t => t.StartsWith("Velocity_"));
        Assert.Contains(tokens, t => t.StartsWith("Duration_"));
    }

    [Fact]
    public void MidiTokenizer_Tokenize_NoteEvent_CPWord()
    {
        var tokenizer = MidiTokenizer.CreateCPWord();

        var tokens = tokenizer.Tokenize("NOTE:60:480:64");

        Assert.NotEmpty(tokens);
        Assert.Contains(tokens, t => t.StartsWith("Note_"));
    }

    [Fact]
    public void MidiTokenizer_Tokenize_NoteEvent_SimpleNote()
    {
        var tokenizer = MidiTokenizer.CreateSimpleNote();

        var tokens = tokenizer.Tokenize("NOTE:60:480:64");

        Assert.NotEmpty(tokens);
        Assert.Contains(tokens, t => t.StartsWith("Pitch_"));
        Assert.Contains(tokens, t => t.StartsWith("Duration_"));
    }

    [Fact]
    public void MidiTokenizer_Tokenize_RestEvent()
    {
        var tokenizer = MidiTokenizer.CreateREMI();

        var tokens = tokenizer.Tokenize("REST:240");

        Assert.NotEmpty(tokens);
        Assert.Contains(tokens, t => t.StartsWith("TimeShift_"));
    }

    [Fact]
    public void MidiTokenizer_Tokenize_BarEvent()
    {
        var tokenizer = MidiTokenizer.CreateREMI();

        var tokens = tokenizer.Tokenize("BAR");

        Assert.Contains("Bar", tokens);
    }

    [Fact]
    public void MidiTokenizer_Tokenize_EmptyString_ReturnsEmptyList()
    {
        var tokenizer = MidiTokenizer.CreateREMI();

        var tokens = tokenizer.Tokenize("");

        Assert.Empty(tokens);
    }

    [Fact]
    public void MidiTokenizer_Tokenize_MultipleEvents()
    {
        var tokenizer = MidiTokenizer.CreateREMI();

        var tokens = tokenizer.Tokenize("BAR;NOTE:60:480:64;REST:120;NOTE:64:240:80");

        Assert.NotEmpty(tokens);
        Assert.True(tokens.Count > 3); // Multiple tokens from multiple events
    }

    [Fact]
    public void MidiTokenizer_TokenizeNotes_REMI()
    {
        var tokenizer = MidiTokenizer.CreateREMI();
        var notes = new List<MidiTokenizer.MidiNote>
        {
            new MidiTokenizer.MidiNote { StartTick = 0, Duration = 480, Pitch = 60, Velocity = 64 },
            new MidiTokenizer.MidiNote { StartTick = 480, Duration = 480, Pitch = 64, Velocity = 80 }
        };

        var tokens = tokenizer.TokenizeNotes(notes);

        Assert.NotEmpty(tokens);
        Assert.Contains(tokens, t => t.Contains("60")); // Pitch 60
        Assert.Contains(tokens, t => t.Contains("64")); // Pitch 64
    }

    [Fact]
    public void MidiTokenizer_TokenizeNotes_CPWord()
    {
        var tokenizer = MidiTokenizer.CreateCPWord();
        var notes = new List<MidiTokenizer.MidiNote>
        {
            new MidiTokenizer.MidiNote { StartTick = 0, Duration = 480, Pitch = 60, Velocity = 64 }
        };

        var tokens = tokenizer.TokenizeNotes(notes);

        Assert.NotEmpty(tokens);
        // CPWord creates compound tokens
        Assert.Contains(tokens, t => t.StartsWith("Note_60_"));
    }

    [Fact]
    public void MidiTokenizer_TokenizeNotes_SimpleNote()
    {
        var tokenizer = MidiTokenizer.CreateSimpleNote();
        var notes = new List<MidiTokenizer.MidiNote>
        {
            new MidiTokenizer.MidiNote { StartTick = 0, Duration = 480, Pitch = 60, Velocity = 64 }
        };

        var tokens = tokenizer.TokenizeNotes(notes);

        Assert.NotEmpty(tokens);
        Assert.Contains(tokens, t => t.StartsWith("Pitch_60"));
        Assert.Contains(tokens, t => t.StartsWith("Duration_"));
    }

    [Fact]
    public void MidiTokenizer_Encode_ReturnsValidResult()
    {
        var tokenizer = MidiTokenizer.CreateREMI();

        var result = tokenizer.Encode("NOTE:60:480:64");

        Assert.NotNull(result);
        Assert.NotEmpty(result.TokenIds);
    }

    [Fact]
    public void MidiTokenizer_Decode_ReconstructsTokens()
    {
        var tokenizer = MidiTokenizer.CreateREMI();
        var options = new EncodingOptions { AddSpecialTokens = false };
        var encoded = tokenizer.Encode("NOTE:60:480:64", options);

        var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

        Assert.NotEmpty(decoded);
    }

    [Fact]
    public void MidiTokenizer_VelocityBins_AreConfigurable()
    {
        var tokenizer = MidiTokenizer.CreateREMI(numVelocityBins: 16);

        // With 16 bins, velocity tokens should be Velocity_0 through Velocity_15
        Assert.True(tokenizer.Vocabulary.ContainsToken("Velocity_0"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("Velocity_15"));
    }

    [Fact]
    public void MidiTokenizer_TicksPerBeat_IsConfigurable()
    {
        // Different ticks per beat
        var tokenizer = MidiTokenizer.CreateREMI(ticksPerBeat: 960);

        Assert.NotNull(tokenizer);
    }

    #endregion

    #region PhonemeTokenizer Tests

    [Fact]
    public void PhonemeTokenizer_CreateARPAbet_CreatesValidTokenizer()
    {
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 30);
    }

    [Fact]
    public void PhonemeTokenizer_Tokenize_ConvertsToPhonemes()
    {
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        var tokens = tokenizer.Tokenize("hello");

        Assert.NotEmpty(tokens);
        // Should contain ARPAbet phonemes
    }

    [Fact]
    public void PhonemeTokenizer_Tokenize_EmptyString_ReturnsEmptyList()
    {
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        var tokens = tokenizer.Tokenize("");

        Assert.Empty(tokens);
    }

    [Fact]
    public void PhonemeTokenizer_Tokenize_NullString_ReturnsEmptyList()
    {
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        var tokens = tokenizer.Tokenize(null!);

        Assert.Empty(tokens);
    }

    [Fact]
    public void PhonemeTokenizer_Tokenize_HandlesMultipleWords()
    {
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        var tokens = tokenizer.Tokenize("hello world");

        Assert.NotEmpty(tokens);
        Assert.Contains("<space>", tokens); // Space separator
    }

    [Fact]
    public void PhonemeTokenizer_Tokenize_HandlesCommonDigraphs()
    {
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        var tokens = tokenizer.Tokenize("the");

        Assert.NotEmpty(tokens);
        Assert.Contains("TH", tokens); // "th" digraph
    }

    [Fact]
    public void PhonemeTokenizer_Encode_ReturnsValidResult()
    {
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        var result = tokenizer.Encode("test");

        Assert.NotNull(result);
        Assert.NotEmpty(result.TokenIds);
    }

    [Fact]
    public void PhonemeTokenizer_Decode_ReconstructsPhonemes()
    {
        var tokenizer = PhonemeTokenizer.CreateARPAbet();
        var options = new EncodingOptions { AddSpecialTokens = false };
        var encoded = tokenizer.Encode("hello", options);

        var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

        Assert.NotEmpty(decoded);
    }

    [Fact]
    public void PhonemeTokenizer_ARPAbet_ContainsStandardPhonemes()
    {
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        // Check for standard ARPAbet phonemes
        Assert.True(tokenizer.Vocabulary.ContainsToken("AA")); // vowel
        Assert.True(tokenizer.Vocabulary.ContainsToken("B"));  // consonant
        Assert.True(tokenizer.Vocabulary.ContainsToken("CH")); // digraph
        Assert.True(tokenizer.Vocabulary.ContainsToken("SH")); // digraph
    }

    [Fact]
    public void PhonemeTokenizer_HandlesPunctuation()
    {
        var tokenizer = PhonemeTokenizer.CreateARPAbet();

        var tokens = tokenizer.Tokenize("hello!");

        // Should handle punctuation without crashing
        Assert.NotEmpty(tokens);
    }

    #endregion

    #region ITokenizer Interface Tests

    [Fact]
    public void ITokenizer_AllImplementations_HaveVocabulary()
    {
        var tokenizers = new ITokenizer[]
        {
            CharacterTokenizer.CreateAscii(),
            WordPieceTokenizer.Train(new[] { "test" }, vocabSize: 50),
            BpeTokenizer.Train(new[] { "test" }, vocabSize: 50),
            MidiTokenizer.CreateREMI(),
            PhonemeTokenizer.CreateARPAbet()
        };

        foreach (var tokenizer in tokenizers)
        {
            Assert.NotNull(tokenizer.Vocabulary);
            Assert.True(tokenizer.VocabularySize > 0);
        }
    }

    [Fact]
    public void ITokenizer_AllImplementations_HaveSpecialTokens()
    {
        var tokenizers = new ITokenizer[]
        {
            CharacterTokenizer.CreateAscii(),
            WordPieceTokenizer.Train(new[] { "test" }, vocabSize: 50),
            BpeTokenizer.Train(new[] { "test" }, vocabSize: 50),
            MidiTokenizer.CreateREMI(),
            PhonemeTokenizer.CreateARPAbet()
        };

        foreach (var tokenizer in tokenizers)
        {
            Assert.NotNull(tokenizer.SpecialTokens);
            Assert.NotNull(tokenizer.SpecialTokens.UnkToken);
        }
    }

    [Fact]
    public void ITokenizer_AllImplementations_HandleEmptyInput()
    {
        var tokenizers = new ITokenizer[]
        {
            CharacterTokenizer.CreateAscii(),
            WordPieceTokenizer.Train(new[] { "test" }, vocabSize: 50),
            BpeTokenizer.Train(new[] { "test" }, vocabSize: 50),
            MidiTokenizer.CreateREMI(),
            PhonemeTokenizer.CreateARPAbet()
        };

        foreach (var tokenizer in tokenizers)
        {
            var tokens = tokenizer.Tokenize("");
            Assert.Empty(tokens);
        }
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void Tokenizer_Encode_WithTokenTypeIds_ReturnsCorrectIds()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions
        {
            AddSpecialTokens = false,
            ReturnTokenTypeIds = true
        };

        var result = tokenizer.Encode("Hi", options);

        Assert.NotEmpty(result.TokenTypeIds);
        Assert.All(result.TokenTypeIds, id => Assert.Equal(0, id));
    }

    [Fact]
    public void Tokenizer_Encode_WithPositionIds_ReturnsCorrectIds()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions
        {
            AddSpecialTokens = false,
            ReturnPositionIds = true
        };

        var result = tokenizer.Encode("Hi", options);

        Assert.Equal(new List<int> { 0, 1 }, result.PositionIds);
    }

    [Fact]
    public void Tokenizer_Encode_LeftPadding_PadsCorrectly()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions
        {
            AddSpecialTokens = false,
            Padding = true,
            PaddingSide = "left",
            MaxLength = 5
        };

        var result = tokenizer.Encode("Hi", options);

        Assert.Equal(5, result.TokenIds.Count);
        // First 3 should be padding (attention mask = 0)
        Assert.Equal(0, result.AttentionMask[0]);
        Assert.Equal(0, result.AttentionMask[1]);
        Assert.Equal(0, result.AttentionMask[2]);
        // Last 2 should be real tokens (attention mask = 1)
        Assert.Equal(1, result.AttentionMask[3]);
        Assert.Equal(1, result.AttentionMask[4]);
    }

    [Fact]
    public void Tokenizer_Encode_LeftTruncation_TruncatesCorrectly()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var options = new EncodingOptions
        {
            AddSpecialTokens = false,
            Truncation = true,
            TruncationSide = "left",
            MaxLength = 2
        };

        var result = tokenizer.Encode("Hello", options);

        Assert.Equal(2, result.Tokens.Count);
        Assert.Equal(new List<string> { "l", "o" }, result.Tokens); // Last 2 chars
    }

    [Fact]
    public void Tokenizer_Decode_EmptyList_ReturnsEmptyString()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();

        var decoded = tokenizer.Decode(new List<int>());

        Assert.Equal(string.Empty, decoded);
    }

    [Fact]
    public void Tokenizer_Decode_NullList_ReturnsEmptyString()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();

        var decoded = tokenizer.Decode(null!);

        Assert.Equal(string.Empty, decoded);
    }

    [Fact]
    public void Tokenizer_Decode_WithSkipSpecialTokens_RemovesSpecialTokens()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var encoded = tokenizer.Encode("Hi"); // Includes [CLS] and [SEP]

        var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

        Assert.DoesNotContain("[CLS]", decoded);
        Assert.DoesNotContain("[SEP]", decoded);
    }

    [Fact]
    public void Tokenizer_Decode_WithoutSkipSpecialTokens_IncludesSpecialTokens()
    {
        var tokenizer = CharacterTokenizer.CreateAscii();
        var encoded = tokenizer.Encode("Hi"); // Includes [CLS] and [SEP]

        var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: false);

        Assert.Contains("[CLS]", decoded);
        Assert.Contains("[SEP]", decoded);
    }

    [Fact]
    public void Vocabulary_LargeVocabulary_HandlesCorrectly()
    {
        var vocabulary = new Vocabulary();

        // Add 10000 tokens
        for (int i = 0; i < 10000; i++)
        {
            vocabulary.AddToken($"token_{i}");
        }

        Assert.Equal(10001, vocabulary.Size); // [UNK] + 10000 tokens
        Assert.True(vocabulary.ContainsToken("token_9999"));
        Assert.Equal(10000, vocabulary.GetTokenId("token_9999"));
    }

    [Fact]
    public void Tokenizer_UnicodeCharacters_HandlesCorrectly()
    {
        var vocabulary = new Vocabulary();
        vocabulary.AddTokens(new[] { "é", "ü", "中", "日" });
        var tokenizer = new CharacterTokenizer(vocabulary, SpecialTokens.Default());

        var tokens = tokenizer.Tokenize("éü中日");

        Assert.Equal(4, tokens.Count);
        Assert.Contains("é", tokens);
        Assert.Contains("中", tokens);
    }

    #endregion
}
