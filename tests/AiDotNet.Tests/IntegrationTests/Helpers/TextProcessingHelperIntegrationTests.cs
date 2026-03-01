using AiDotNet.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for TextProcessingHelper:
/// SplitIntoSentences, Tokenize.
/// </summary>
public class TextProcessingHelperIntegrationTests
{
    #region SplitIntoSentences

    [Fact]
    public void SplitIntoSentences_MultipleSentences()
    {
        string text = "Hello world. How are you? I am fine!";
        var sentences = TextProcessingHelper.SplitIntoSentences(text);
        Assert.NotEmpty(sentences);
        Assert.True(sentences.Count >= 2, "Should split into multiple sentences");
    }

    [Fact]
    public void SplitIntoSentences_SingleSentence()
    {
        string text = "Hello world";
        var sentences = TextProcessingHelper.SplitIntoSentences(text);
        Assert.Single(sentences);
        Assert.Equal("Hello world", sentences[0]);
    }

    [Fact]
    public void SplitIntoSentences_EmptyString_ReturnsEmpty()
    {
        var sentences = TextProcessingHelper.SplitIntoSentences("");
        Assert.Empty(sentences);
    }

    [Fact]
    public void SplitIntoSentences_NullString_ReturnsEmpty()
    {
        var sentences = TextProcessingHelper.SplitIntoSentences(null!);
        Assert.Empty(sentences);
    }

    [Fact]
    public void SplitIntoSentences_WhitespaceOnly_ReturnsEmpty()
    {
        var sentences = TextProcessingHelper.SplitIntoSentences("   ");
        Assert.Empty(sentences);
    }

    [Fact]
    public void SplitIntoSentences_WithNewlines()
    {
        string text = "First sentence.\nSecond sentence.";
        var sentences = TextProcessingHelper.SplitIntoSentences(text);
        Assert.True(sentences.Count >= 1);
    }

    #endregion

    #region Tokenize

    [Fact]
    public void Tokenize_SimpleText()
    {
        string text = "Hello World";
        var tokens = TextProcessingHelper.Tokenize(text);
        Assert.Equal(2, tokens.Count);
        Assert.Equal("hello", tokens[0]);
        Assert.Equal("world", tokens[1]);
    }

    [Fact]
    public void Tokenize_ConvertToLowercase()
    {
        string text = "HELLO World";
        var tokens = TextProcessingHelper.Tokenize(text);
        Assert.All(tokens, t => Assert.Equal(t.ToLowerInvariant(), t));
    }

    [Fact]
    public void Tokenize_RemovesPunctuation()
    {
        string text = "Hello, world! How are you?";
        var tokens = TextProcessingHelper.Tokenize(text);
        Assert.DoesNotContain(",", tokens);
        Assert.DoesNotContain("!", tokens);
        Assert.DoesNotContain("?", tokens);
    }

    [Fact]
    public void Tokenize_EmptyString_ReturnsEmpty()
    {
        var tokens = TextProcessingHelper.Tokenize("");
        Assert.Empty(tokens);
    }

    [Fact]
    public void Tokenize_NullString_ReturnsEmpty()
    {
        var tokens = TextProcessingHelper.Tokenize(null!);
        Assert.Empty(tokens);
    }

    [Fact]
    public void Tokenize_WithTabs()
    {
        string text = "Hello\tWorld";
        var tokens = TextProcessingHelper.Tokenize(text);
        Assert.Equal(2, tokens.Count);
    }

    #endregion
}
