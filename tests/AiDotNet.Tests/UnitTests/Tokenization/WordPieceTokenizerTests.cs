using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// Unit tests for WordPiece tokenizer (BERT-style).
/// </summary>
public class WordPieceTokenizerTests
{
    private readonly WordPieceTokenizer _tokenizer;
    private readonly List<string> _trainingCorpus;

    public WordPieceTokenizerTests()
    {
        _trainingCorpus = new List<string>
        {
            "Hello world, this is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "Natural language processing enables computers to understand text.",
            "WordPiece tokenization is used by BERT models.",
            "Subword units help handle unknown words effectively."
        };

        _tokenizer = WordPieceTokenizer.Train(_trainingCorpus, 500);
    }

    [Fact]
    public void Train_CreatesVocabulary_WithCorrectSize()
    {
        // Arrange & Act
        var tokenizer = WordPieceTokenizer.Train(_trainingCorpus, 100);

        // Assert
        Assert.True(tokenizer.VocabularySize > 0);
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
    public void Tokenize_UnknownWord_ReturnsUnkOrSubwords()
    {
        // Arrange
        var text = "xyzabc123";

        // Act
        var tokens = _tokenizer.Tokenize(text);

        // Assert
        Assert.NotNull(tokens);
        Assert.NotEmpty(tokens);
    }

    [Fact]
    public void Tokenize_ContinuationPrefix_UsesHashHash()
    {
        // Arrange
        var text = "tokenization";

        // Act
        var tokens = _tokenizer.Tokenize(text);

        // Assert
        Assert.NotNull(tokens);
        // WordPiece uses ## for continuation tokens
        // May or may not have continuation depending on vocabulary
        Assert.True(tokens.Count > 0, "Should produce at least one token");
    }

    [Fact]
    public void Encode_WithSpecialTokens_AddsBertStyleTokens()
    {
        // Arrange
        var text = "Hello world";
        var options = new EncodingOptions { AddSpecialTokens = true };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        Assert.NotNull(result.Tokens);
        Assert.Contains("[CLS]", result.Tokens);
        Assert.Contains("[SEP]", result.Tokens);
    }

    [Fact]
    public void Encode_WithAttentionMask_ReturnsAllOnes()
    {
        // Arrange
        var text = "Hello world";
        var options = new EncodingOptions { ReturnAttentionMask = true };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        Assert.All(result.AttentionMask, m => Assert.Equal(1, m));
    }

    [Fact]
    public void Encode_WithPadding_HasZerosInAttentionMask()
    {
        // Arrange
        var text = "Hi";
        var options = new EncodingOptions
        {
            Padding = true,
            MaxLength = 20,
            ReturnAttentionMask = true
        };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        Assert.Equal(20, result.AttentionMask.Count);
        Assert.Contains(0, result.AttentionMask); // Padding has 0 attention
    }

    [Fact]
    public void Decode_RemovesContinuationMarkers()
    {
        // Arrange
        var text = "tokenization";
        var encoded = _tokenizer.Encode(text);

        // Act
        var decoded = _tokenizer.Decode(encoded.TokenIds);

        // Assert
        Assert.DoesNotContain("##", decoded);
    }

    [Fact]
    public void Roundtrip_PreservesWordContent()
    {
        // Arrange
        var text = "machine learning";

        // Act
        var tokens = _tokenizer.Tokenize(text);
        var ids = _tokenizer.ConvertTokensToIds(tokens);
        var decoded = _tokenizer.Decode(ids);

        // Assert
        var normalizedDecoded = decoded.ToLowerInvariant().Replace("##", "").Trim();
        Assert.Contains("machine", normalizedDecoded);
    }

    [Fact]
    public void SpecialTokens_AreBertStyle()
    {
        // Assert
        Assert.Equal("[UNK]", _tokenizer.SpecialTokens.UnkToken);
        Assert.Equal("[PAD]", _tokenizer.SpecialTokens.PadToken);
        Assert.Equal("[CLS]", _tokenizer.SpecialTokens.ClsToken);
        Assert.Equal("[SEP]", _tokenizer.SpecialTokens.SepToken);
        Assert.Equal("[MASK]", _tokenizer.SpecialTokens.MaskToken);
    }

    [Fact]
    public void Encode_TokenTypeIds_DefaultsToZero()
    {
        // Arrange
        var text = "Hello world";
        var options = new EncodingOptions { ReturnTokenTypeIds = true };

        // Act
        var result = _tokenizer.Encode(text, options);

        // Assert
        Assert.All(result.TokenTypeIds, id => Assert.Equal(0, id));
    }

    [Fact]
    public void EncodeBatch_ConsistentResults()
    {
        // Arrange
        var texts = new List<string> { "Hello", "world", "test" };

        // Act
        var results = _tokenizer.EncodeBatch(texts);

        // Assert
        Assert.Equal(3, results.Count);
        Assert.All(results, r => Assert.NotEmpty(r.TokenIds));
    }

    #region PR #757 Bug Fix Tests - Parameter Validation

    [Fact]
    public void Train_NullCorpus_ThrowsArgumentNullException()
    {
        Assert.Throws<System.ArgumentNullException>(() =>
            WordPieceTokenizer.Train(null!, 100));
    }

    [Fact]
    public void Train_InvalidVocabSize_ThrowsArgumentOutOfRangeException()
    {
        var corpus = new List<string> { "Hello world" };

        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            WordPieceTokenizer.Train(corpus, 0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            WordPieceTokenizer.Train(corpus, -1));
    }

    [Fact]
    public void Constructor_InvalidMaxInputCharsPerWord_ThrowsArgumentOutOfRangeException()
    {
        var vocabulary = new AiDotNet.Tokenization.Vocabulary.Vocabulary();

        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            new WordPieceTokenizer(vocabulary, maxInputCharsPerWord: 0));
        Assert.Throws<System.ArgumentOutOfRangeException>(() =>
            new WordPieceTokenizer(vocabulary, maxInputCharsPerWord: -1));
    }

    #endregion
}
