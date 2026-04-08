using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// Unit tests for Character-level tokenizer.
/// </summary>
public class CharacterTokenizerTests
{
    [Fact]
    public void CreateAscii_CreatesValidTokenizer()
    {
        // Act
        var tokenizer = CharacterTokenizer.CreateAscii();

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void Tokenize_SplitsIntoCharacters()
    {
        // Arrange
        var tokenizer = CharacterTokenizer.CreateAscii();
        var text = "Hello";

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        Assert.Equal(5, tokens.Count);
        Assert.Equal("H", tokens[0]);
        Assert.Equal("e", tokens[1]);
        Assert.Equal("l", tokens[2]);
        Assert.Equal("l", tokens[3]);
        Assert.Equal("o", tokens[4]);
    }

    [Fact]
    public void Tokenize_WithLowercase_ConvertsToLower()
    {
        // Arrange
        var tokenizer = CharacterTokenizer.CreateAscii(lowercase: true);
        var text = "Hello";

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        Assert.Equal("h", tokens[0]);
        Assert.All(tokens, t => Assert.Equal(t.ToLowerInvariant(), t));
    }

    [Fact]
    public void Tokenize_PreservesWhitespace()
    {
        // Arrange
        var tokenizer = CharacterTokenizer.CreateAscii();
        var text = "Hi there";

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        Assert.Equal(8, tokens.Count);
        Assert.Equal(" ", tokens[2]); // Space between Hi and there
    }

    [Fact]
    public void Tokenize_EmptyText_ReturnsEmpty()
    {
        // Arrange
        var tokenizer = CharacterTokenizer.CreateAscii();

        // Act
        var tokens = tokenizer.Tokenize("");

        // Assert
        Assert.Empty(tokens);
    }

    [Fact]
    public void Encode_ReturnsCorrectTokenIds()
    {
        // Arrange
        var tokenizer = CharacterTokenizer.CreateAscii();
        var text = "AB";
        var options = new EncodingOptions { AddSpecialTokens = false };

        // Act
        var result = tokenizer.Encode(text, options);

        // Assert
        Assert.Equal(2, result.TokenIds.Count);
        Assert.NotEqual(result.TokenIds[0], result.TokenIds[1]); // A and B have different IDs
    }

    [Fact]
    public void Decode_ReconstructsOriginalText()
    {
        // Arrange
        var tokenizer = CharacterTokenizer.CreateAscii();
        var text = "Hello World";

        // Act
        var encoded = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(encoded.TokenIds);

        // Assert
        Assert.Equal(text, decoded);
    }

    [Fact]
    public void Roundtrip_IsExact()
    {
        // Arrange
        var tokenizer = CharacterTokenizer.CreateAscii();
        var text = "Test 123!";

        // Act
        var tokens = tokenizer.Tokenize(text);
        var ids = tokenizer.ConvertTokensToIds(tokens);
        var decoded = tokenizer.Decode(ids);

        // Assert
        Assert.Equal(text, decoded);
    }

    [Fact]
    public void Train_FromCorpus_CreatesVocabulary()
    {
        // Arrange
        var corpus = new List<string> { "Hello", "World", "Test" };

        // Act
        var tokenizer = CharacterTokenizer.Train(corpus);

        // Assert
        Assert.True(tokenizer.VocabularySize > 0);
        Assert.True(tokenizer.Vocabulary.ContainsToken("H"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("e"));
    }

    [Fact]
    public void Train_WithMinFrequency_FiltersRareChars()
    {
        // Arrange
        var corpus = new List<string> { "aaa", "bbb", "c" }; // c appears only once

        // Act
        var tokenizer = CharacterTokenizer.Train(corpus, minFrequency: 2);

        // Assert
        Assert.True(tokenizer.Vocabulary.ContainsToken("a"));
        Assert.True(tokenizer.Vocabulary.ContainsToken("b"));
        // c should NOT be in vocabulary since it appears only once (below minFrequency of 2)
        Assert.False(tokenizer.Vocabulary.ContainsToken("c"));
    }

    [Fact]
    public void Tokenize_NonAsciiChar_ReturnsUnk()
    {
        // Arrange
        var tokenizer = CharacterTokenizer.CreateAscii();
        var text = "Hello\u4e2d"; // Chinese character

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        Assert.Equal(6, tokens.Count);
        Assert.Equal(tokenizer.SpecialTokens.UnkToken, tokens[5]);
    }

    [Fact]
    public void Encode_WithPositionIds_ReturnsSequential()
    {
        // Arrange
        var tokenizer = CharacterTokenizer.CreateAscii();
        var text = "ABC";
        var options = new EncodingOptions { ReturnPositionIds = true, AddSpecialTokens = false };

        // Act
        var result = tokenizer.Encode(text, options);

        // Assert
        Assert.Equal(new List<int> { 0, 1, 2 }, result.PositionIds);
    }

    [Fact]
    public void VocabularySize_MatchesAsciiPrintable()
    {
        // Arrange
        var tokenizer = CharacterTokenizer.CreateAscii();

        // Assert
        // ASCII printable: 95 chars (32-126) + special tokens
        Assert.True(tokenizer.VocabularySize >= 95);
    }

    #region PR #757 Bug Fix Tests - Parameter Validation

    [Fact]
    public void Train_NullCorpus_ThrowsArgumentNullException()
    {
        Assert.Throws<System.ArgumentNullException>(() =>
            CharacterTokenizer.Train(null!));
    }

    [Fact]
    public void Train_InvalidMinFrequency_ThrowsArgumentOutOfRangeException()
    {
        var corpus = new List<string> { "Hello world" };

        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            CharacterTokenizer.Train(corpus, minFrequency: 0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            CharacterTokenizer.Train(corpus, minFrequency: -1));
    }

    #endregion
}
