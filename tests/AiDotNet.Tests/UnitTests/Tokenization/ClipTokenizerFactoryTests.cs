using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Models;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Tokenization;

/// <summary>
/// Unit tests for CLIP tokenizer factory.
/// </summary>
public class ClipTokenizerFactoryTests
{
    [Fact]
    public void CreateSimple_ReturnsValidTokenizer()
    {
        // Act
        var tokenizer = ClipTokenizerFactory.CreateSimple();

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
    }

    [Fact]
    public void CreateSimple_WithCustomCorpus_TrainsOnCorpus()
    {
        // Arrange
        var corpus = new[]
        {
            "a photo of a cat",
            "a photo of a dog",
            "an image of a bird"
        };

        // Act
        var tokenizer = ClipTokenizerFactory.CreateSimple(corpus, vocabSize: 200);

        // Assert
        Assert.NotNull(tokenizer);
        Assert.True(tokenizer.VocabularySize > 0);
        Assert.True(tokenizer.VocabularySize <= 210); // vocab size + special tokens
    }

    [Fact]
    public void CreateSimple_HasClipSpecialTokens()
    {
        // Act
        var tokenizer = ClipTokenizerFactory.CreateSimple();

        // Assert
        Assert.Equal("<|startoftext|>", tokenizer.SpecialTokens.BosToken);
        Assert.Equal("<|endoftext|>", tokenizer.SpecialTokens.EosToken);
        Assert.Equal("<|startoftext|>", tokenizer.SpecialTokens.ClsToken);
    }

    [Fact]
    public void CreateSimple_CanTokenizeText()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var text = "a photo of a cat";

        // Act
        var result = tokenizer.Encode(text);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.TokenIds);
        Assert.NotEmpty(result.TokenIds);
    }

    [Fact]
    public void CreateSimple_WithSpecialTokens_AddsStartToken()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var text = "hello world";
        var options = new EncodingOptions { AddSpecialTokens = true };

        // Act
        var result = tokenizer.Encode(text, options);

        // Assert
        Assert.NotNull(result.Tokens);
        // CLIP BOS/CLS token should be added at the start
        Assert.Contains("<|startoftext|>", result.Tokens);
    }

    [Fact]
    public void GetDefaultEncodingOptions_ReturnsCorrectDefaults()
    {
        // Act
        var options = ClipTokenizerFactory.GetDefaultEncodingOptions();

        // Assert
        Assert.Equal(77, options.MaxLength);
        Assert.True(options.Padding);
        Assert.Equal("right", options.PaddingSide);
        Assert.True(options.Truncation);
        Assert.Equal("right", options.TruncationSide);
        Assert.True(options.AddSpecialTokens);
        Assert.True(options.ReturnAttentionMask);
    }

    [Fact]
    public void GetDefaultEncodingOptions_WithCustomMaxLength_UsesCustomValue()
    {
        // Act
        var options = ClipTokenizerFactory.GetDefaultEncodingOptions(maxLength: 128);

        // Assert
        Assert.Equal(128, options.MaxLength);
    }

    [Fact]
    public void IsClipCompatible_WithSimpleTokenizer_ChecksCompatibility()
    {
        // Arrange - Simple tokenizer with 1000 vocab size
        var tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: 1000);

        // Act
        var isCompatible = ClipTokenizerFactory.IsClipCompatible(tokenizer);

        // Assert - Simple tokenizer is CLIP-compatible if it has right special tokens and vocab >= 1000
        Assert.NotNull(tokenizer);
        // The tokenizer has correct special tokens and is considered CLIP-compatible
        Assert.Equal("<|startoftext|>", tokenizer.SpecialTokens.BosToken);
        Assert.Equal("<|endoftext|>", tokenizer.SpecialTokens.EosToken);
        Assert.True(isCompatible);
    }

    [Fact]
    public void IsClipCompatible_WithNullTokenizer_ReturnsFalse()
    {
        // Act
        var isCompatible = ClipTokenizerFactory.IsClipCompatible(null!);

        // Assert
        Assert.False(isCompatible);
    }

    [Fact]
    public void DefaultVocabSize_Is49408()
    {
        // Assert
        Assert.Equal(49408, ClipTokenizerFactory.DefaultVocabSize);
    }

    [Fact]
    public void DefaultMaxLength_Is77()
    {
        // Assert
        Assert.Equal(77, ClipTokenizerFactory.DefaultMaxLength);
    }

    [Fact]
    public void ClipPattern_IsNotNullOrEmpty()
    {
        // Assert
        Assert.False(string.IsNullOrEmpty(ClipTokenizerFactory.ClipPattern));
    }

    [Fact]
    public void CreateSimple_TokenizerCanEncodeDecode()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var text = "machine learning";

        // Act
        var encoded = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(encoded.TokenIds);

        // Assert
        Assert.NotEmpty(decoded);
        // Content should be preserved (may have different case/spacing)
        Assert.Contains("machine", decoded.ToLowerInvariant());
    }

    [Fact]
    public void CreateSimple_WithPaddingOptions_PadsToMaxLength()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var text = "cat";
        var options = ClipTokenizerFactory.GetDefaultEncodingOptions();

        // Act
        var result = tokenizer.Encode(text, options);

        // Assert
        Assert.Equal(77, result.TokenIds.Count);
        Assert.Equal(77, result.AttentionMask.Count);
    }

    [Fact]
    public void CreateSimple_WithTruncationOptions_TruncatesToMaxLength()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var longText = string.Join(" ", Enumerable.Repeat("word", 100));
        var options = ClipTokenizerFactory.GetDefaultEncodingOptions();

        // Act
        var result = tokenizer.Encode(longText, options);

        // Assert
        Assert.Equal(77, result.TokenIds.Count);
    }

    [Fact]
    public void FromPretrained_WithNullVocabPath_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            ClipTokenizerFactory.FromPretrained(null!, "merges.txt"));
    }

    [Fact]
    public void FromPretrained_WithNullMergesPath_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            ClipTokenizerFactory.FromPretrained("vocab.json", null!));
    }

    [Fact]
    public void FromPretrained_WithEmptyVocabPath_ThrowsArgumentException()
    {
        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            ClipTokenizerFactory.FromPretrained("", "merges.txt"));
    }

    [Fact]
    public void FromPretrained_WithNonExistentVocabFile_ThrowsFileNotFoundException()
    {
        // Arrange
        var nonExistentPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString(), "vocab.json");

        // Act & Assert
        Assert.Throws<FileNotFoundException>(() =>
            ClipTokenizerFactory.FromPretrained(nonExistentPath, "merges.txt"));
    }

    [Fact]
    public void CreateSimple_BatchEncode_ProcessesMultipleTexts()
    {
        // Arrange
        var tokenizer = ClipTokenizerFactory.CreateSimple();
        var texts = new List<string>
        {
            "a photo of a cat",
            "a photo of a dog",
            "an image of a bird"
        };

        // Act
        var results = tokenizer.EncodeBatch(texts);

        // Assert
        Assert.Equal(3, results.Count);
        Assert.All(results, result => Assert.NotEmpty(result.TokenIds));
    }
}
