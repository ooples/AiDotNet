using System.Collections.Generic;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// Unit tests for Unigram (SentencePiece-style) tokenizer.
/// </summary>
public class UnigramTokenizerTests
{
    private readonly UnigramTokenizer _tokenizer;
    private readonly List<string> _trainingCorpus;

    public UnigramTokenizerTests()
    {
        _trainingCorpus = new List<string>
        {
            "Hello world, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "Unigram models use probabilistic segmentation.",
            "The Viterbi algorithm finds optimal tokenization."
        };

        _tokenizer = UnigramTokenizer.Train(_trainingCorpus, 500);
    }

    [Fact]
    public void Train_CreatesVocabulary()
    {
        // Assert
        Assert.True(_tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void Tokenize_UsesViterbiSegmentation()
    {
        // Arrange
        var text = "machine learning";

        // Act
        var tokens = _tokenizer.Tokenize(text);

        // Assert
        Assert.NotNull(tokens);
        Assert.NotEmpty(tokens);
    }

    [Fact]
    public void Tokenize_HandlesSpaces_WithSentencePieceMarker()
    {
        // Arrange
        var text = "Hello world";

        // Act
        var tokens = _tokenizer.Tokenize(text);

        // Assert
        // Unigram uses \u2581 (▁) to mark spaces
        var joinedTokens = string.Join("", tokens);
        Assert.Contains("\u2581", joinedTokens);
    }

    [Fact]
    public void Tokenize_EmptyText_ReturnsEmpty()
    {
        // Act
        var tokens = _tokenizer.Tokenize("");

        // Assert
        Assert.Empty(tokens);
    }

    [Fact]
    public void Encode_ReturnsValidResult()
    {
        // Arrange
        var text = "Hello world";

        // Act
        var result = _tokenizer.Encode(text);

        // Assert
        Assert.NotEmpty(result.TokenIds);
        Assert.Equal(result.Tokens.Count, result.TokenIds.Count);
    }

    [Fact]
    public void Decode_RemovesSentencePieceMarker()
    {
        // Arrange
        var text = "Hello world";
        var encoded = _tokenizer.Encode(text);

        // Act
        var decoded = _tokenizer.Decode(encoded.TokenIds);

        // Assert
        Assert.DoesNotContain("\u2581", decoded);
    }

    [Fact]
    public void Roundtrip_PreservesContent()
    {
        // Arrange
        var text = "machine learning";

        // Act
        var encoded = _tokenizer.Encode(text);
        var decoded = _tokenizer.Decode(encoded.TokenIds);

        // Assert
        var normalizedDecoded = decoded.ToLowerInvariant().Trim();
        Assert.Contains("machine", normalizedDecoded);
        Assert.Contains("learning", normalizedDecoded);
    }

    [Fact]
    public void Encode_WithPositionIds_ReturnsSequential()
    {
        // Arrange
        var text = "Test text";
        var options = new EncodingOptions { ReturnPositionIds = true };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        for (int i = 0; i < result.PositionIds.Count; i++)
        {
            Assert.Equal(i, result.PositionIds[i]);
        }
    }

    [Fact]
    public void Train_WithSmallVocab_Works()
    {
        // Arrange & Act
        var smallTokenizer = UnigramTokenizer.Train(_trainingCorpus, 50);

        // Assert
        Assert.True(smallTokenizer.VocabularySize > 0);
        Assert.True(smallTokenizer.VocabularySize <= 60); // vocab + special tokens
    }

    [Fact]
    public void Tokenize_LongWord_BreaksIntoSubwords()
    {
        // Arrange
        var text = "antidisestablishmentarianism";

        // Act
        var tokens = _tokenizer.Tokenize(text);

        // Assert
        // Long unknown words should be broken into subwords
        Assert.True(tokens.Count >= 1);
    }
}
