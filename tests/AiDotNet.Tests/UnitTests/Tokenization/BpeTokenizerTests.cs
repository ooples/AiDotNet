using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// Unit tests for BPE (Byte-Pair Encoding) tokenizer.
/// </summary>
public class BpeTokenizerTests
{
    private readonly BpeTokenizer _tokenizer;
    private readonly List<string> _trainingCorpus;

    public BpeTokenizerTests()
    {
        _trainingCorpus = new List<string>
        {
            "Hello world, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "Natural language processing enables computers to understand text.",
            "Tokenization is the first step in text processing pipelines.",
            "Deep learning models require proper tokenization.",
            "Subword tokenization handles out-of-vocabulary words.",
            "BPE tokenizer merges frequent character pairs."
        };

        _tokenizer = BpeTokenizer.Train(_trainingCorpus, 500);
    }

    [Fact]
    public void Train_CreatesVocabulary_WithCorrectSize()
    {
        // Arrange & Act
        var tokenizer = BpeTokenizer.Train(_trainingCorpus, 100);

        // Assert
        Assert.True(tokenizer.VocabularySize > 0);
        Assert.True(tokenizer.VocabularySize <= 100 + 10); // vocab size + special tokens
    }

    [Fact]
    public void Tokenize_SimpleText_ReturnsTokens()
    {
        // Arrange
        var text = "Hello world";

        // Act
        var tokens = _tokenizer.Tokenize(text);

        // Assert
        Assert.NotNull(tokens);
        Assert.NotEmpty(tokens);
    }

    [Fact]
    public void Tokenize_EmptyText_ReturnsEmptyList()
    {
        // Arrange
        var text = "";

        // Act
        var tokens = _tokenizer.Tokenize(text);

        // Assert
        Assert.NotNull(tokens);
        Assert.Empty(tokens);
    }

    [Fact]
    public void Tokenize_NullText_ReturnsEmptyList()
    {
        // Arrange
        string? text = null;

        // Act
        var tokens = _tokenizer.Tokenize(text!);

        // Assert
        Assert.NotNull(tokens);
        Assert.Empty(tokens);
    }

    [Fact]
    public void Encode_ReturnsTokenizationResult_WithTokenIds()
    {
        // Arrange
        var text = "Hello world";

        // Act
        var result = _tokenizer.Encode(text);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.Tokens);
        Assert.NotNull(result.TokenIds);
        Assert.Equal(result.Tokens.Count, result.TokenIds.Count);
    }

    [Fact]
    public void Encode_WithSpecialTokens_AddsClsAndSep()
    {
        // Arrange
        var text = "Hello world";
        var options = new EncodingOptions { AddSpecialTokens = true };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        Assert.NotNull(result.Tokens);
        // Only check for special tokens if they are non-empty
        if (!string.IsNullOrEmpty(_tokenizer.SpecialTokens.ClsToken))
        {
            Assert.Contains(_tokenizer.SpecialTokens.ClsToken, result.Tokens);
        }
        if (!string.IsNullOrEmpty(_tokenizer.SpecialTokens.SepToken))
        {
            Assert.Contains(_tokenizer.SpecialTokens.SepToken, result.Tokens);
        }
    }

    [Fact]
    public void Encode_WithAttentionMask_ReturnsValidMask()
    {
        // Arrange
        var text = "Hello world";
        var options = new EncodingOptions { ReturnAttentionMask = true };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        Assert.NotNull(result.AttentionMask);
        Assert.Equal(result.TokenIds.Count, result.AttentionMask.Count);
        Assert.All(result.AttentionMask, mask => Assert.Equal(1, mask));
    }

    [Fact]
    public void Encode_WithTokenTypeIds_ReturnsValidTypeIds()
    {
        // Arrange
        var text = "Hello world";
        var options = new EncodingOptions { ReturnTokenTypeIds = true };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        Assert.NotNull(result.TokenTypeIds);
        Assert.Equal(result.TokenIds.Count, result.TokenTypeIds.Count);
    }

    [Fact]
    public void Encode_WithPositionIds_ReturnsSequentialIds()
    {
        // Arrange
        var text = "Hello world";
        var options = new EncodingOptions { ReturnPositionIds = true };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        Assert.NotNull(result.PositionIds);
        Assert.Equal(result.TokenIds.Count, result.PositionIds.Count);
        for (int i = 0; i < result.PositionIds.Count; i++)
        {
            Assert.Equal(i, result.PositionIds[i]);
        }
    }

    [Fact]
    public void Encode_WithPadding_PadsToMaxLength()
    {
        // Arrange
        var text = "Hi";
        var options = new EncodingOptions
        {
            Padding = true,
            MaxLength = 10,
            PaddingSide = "right"
        };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        Assert.Equal(10, result.TokenIds.Count);
    }

    [Fact]
    public void Encode_WithTruncation_TruncatesToMaxLength()
    {
        // Arrange
        var text = "This is a very long sentence that should be truncated to fit the maximum length";
        var options = new EncodingOptions
        {
            Truncation = true,
            MaxLength = 5,
            TruncationSide = "right"
        };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        Assert.True(result.TokenIds.Count <= 5);
    }

    [Fact]
    public void Decode_ReturnsOriginalText_Approximately()
    {
        // Arrange
        var text = "Hello world";
        var encoded = _tokenizer.Encode(text);

        // Act
        var decoded = _tokenizer.Decode(encoded.TokenIds);

        // Assert
        Assert.NotNull(decoded);
        Assert.NotEmpty(decoded);
        // BPE may not produce exact roundtrip, but should contain key words
        Assert.Contains("Hello", decoded, System.StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Decode_SkipsSpecialTokens_ByDefault()
    {
        // Arrange
        var text = "Hello world";
        var options = new EncodingOptions { AddSpecialTokens = true };
        var encoded = _tokenizer.Encode(text, options);

        // Act
        var decoded = _tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

        // Assert - Only check if special tokens are non-empty
        if (!string.IsNullOrEmpty(_tokenizer.SpecialTokens.ClsToken))
        {
            Assert.DoesNotContain(_tokenizer.SpecialTokens.ClsToken, decoded);
        }
        if (!string.IsNullOrEmpty(_tokenizer.SpecialTokens.SepToken))
        {
            Assert.DoesNotContain(_tokenizer.SpecialTokens.SepToken, decoded);
        }
    }

    [Fact]
    public void EncodeBatch_ProcessesMultipleTexts()
    {
        // Arrange
        var texts = new List<string> { "Hello world", "Test text", "Another sentence" };

        // Act
        var results = _tokenizer.EncodeBatch(texts);

        // Assert
        Assert.Equal(3, results.Count);
        Assert.All(results, r => Assert.NotEmpty(r.TokenIds));
    }

    [Fact]
    public void DecodeBatch_ProcessesMultipleTokenIdLists()
    {
        // Arrange
        var texts = new List<string> { "Hello world", "Test text" };
        var encoded = _tokenizer.EncodeBatch(texts);
        var tokenIdsBatch = encoded.Select(e => e.TokenIds).ToList();

        // Act
        var decoded = _tokenizer.DecodeBatch(tokenIdsBatch);

        // Assert
        Assert.Equal(2, decoded.Count);
        Assert.All(decoded, d => Assert.NotEmpty(d));
    }

    [Fact]
    public void ConvertTokensToIds_ReturnsValidIds()
    {
        // Arrange
        var tokens = _tokenizer.Tokenize("Hello world");

        // Act
        var ids = _tokenizer.ConvertTokensToIds(tokens);

        // Assert
        Assert.Equal(tokens.Count, ids.Count);
        Assert.All(ids, id => Assert.True(id >= 0));
    }

    [Fact]
    public void ConvertIdsToTokens_ReturnsValidTokens()
    {
        // Arrange
        var text = "Hello world";
        var encoded = _tokenizer.Encode(text);

        // Act
        var tokens = _tokenizer.ConvertIdsToTokens(encoded.TokenIds);

        // Assert
        Assert.Equal(encoded.TokenIds.Count, tokens.Count);
        Assert.All(tokens, t => Assert.NotNull(t));
    }

    [Fact]
    public void Roundtrip_TokenizeAndDetokenize_PreservesContent()
    {
        // Arrange
        var originalText = "machine learning";

        // Act
        var tokens = _tokenizer.Tokenize(originalText);
        var ids = _tokenizer.ConvertTokensToIds(tokens);
        var decoded = _tokenizer.Decode(ids);

        // Assert
        // Normalize whitespace for comparison
        var normalizedOriginal = originalText.ToLowerInvariant().Trim();
        var normalizedDecoded = decoded.ToLowerInvariant().Trim();

        // Should contain the core content
        Assert.Contains("machine", normalizedDecoded);
        Assert.Contains("learning", normalizedDecoded);
    }

    [Fact]
    public void SpecialTokens_AreInVocabulary()
    {
        // Assert
        Assert.True(_tokenizer.Vocabulary.ContainsToken(_tokenizer.SpecialTokens.UnkToken));
        Assert.True(_tokenizer.Vocabulary.ContainsToken(_tokenizer.SpecialTokens.PadToken));
    }

    [Fact]
    public void Train_WithEmptyCorpus_CreatesMinimalTokenizer()
    {
        // Arrange
        var emptyCorpus = new List<string>();

        // Act - Training with empty corpus creates a tokenizer with only special tokens
        var tokenizer = BpeTokenizer.Train(emptyCorpus, 100);

        // Assert - Tokenizer is created but with minimal vocabulary
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0); // At least special tokens exist
    }

    [Fact]
    public void VocabularySize_ReturnsCorrectCount()
    {
        // Assert
        Assert.True(_tokenizer.VocabularySize > 0);
        Assert.Equal(_tokenizer.Vocabulary.Size, _tokenizer.VocabularySize);
    }

    #region PR #757 Bug Fix Tests - Parameter Validation

    [Fact]
    public void Train_NullCorpus_ThrowsArgumentNullException()
    {
        Assert.Throws<System.ArgumentNullException>(() =>
            BpeTokenizer.Train(null!, 100));
    }

    [Fact]
    public void Train_InvalidVocabSize_ThrowsArgumentOutOfRangeException()
    {
        var corpus = new List<string> { "Hello world" };

        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            BpeTokenizer.Train(corpus, 0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            BpeTokenizer.Train(corpus, -1));
    }

    #endregion
}
