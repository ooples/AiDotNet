#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.Helpers
{
    public class TextProcessingHelperTests
    {
        [Fact]
        public void SplitIntoSentences_WithNull_ReturnsEmptyList()
        {
            // Act
            var result = TextProcessingHelper.SplitIntoSentences(null);

            // Assert
            Assert.NotNull(result);
            Assert.Empty(result);
        }

        [Fact]
        public void SplitIntoSentences_WithEmptyString_ReturnsEmptyList()
        {
            // Act
            var result = TextProcessingHelper.SplitIntoSentences(string.Empty);

            // Assert
            Assert.NotNull(result);
            Assert.Empty(result);
        }

        [Fact]
        public void SplitIntoSentences_WithWhitespace_ReturnsEmptyList()
        {
            // Act
            var result = TextProcessingHelper.SplitIntoSentences("   \t\n  ");

            // Assert
            Assert.NotNull(result);
            Assert.Empty(result);
        }

        [Fact]
        public void SplitIntoSentences_WithSingleSentence_ReturnsSingleItem()
        {
            // Arrange
            var text = "This is a sentence.";

            // Act
            var result = TextProcessingHelper.SplitIntoSentences(text);

            // Assert
            Assert.Single(result);
            Assert.Equal("This is a sentence.", result[0]);
        }

        [Fact]
        public void SplitIntoSentences_WithMultipleSentences_SplitsCorrectly()
        {
            // Arrange
            var text = "First sentence. Second sentence. Third sentence.";

            // Act
            var result = TextProcessingHelper.SplitIntoSentences(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("First sentence.", result[0]);
            Assert.Equal("Second sentence.", result[1]);
            Assert.Equal("Third sentence.", result[2]);
        }

        [Fact]
        public void SplitIntoSentences_WithExclamationMark_SplitsCorrectly()
        {
            // Arrange
            var text = "Hello there! How are you?";

            // Act
            var result = TextProcessingHelper.SplitIntoSentences(text);

            // Assert
            Assert.Equal(2, result.Count);
            Assert.Equal("Hello there!", result[0]);
            Assert.Equal("How are you?", result[1]);
        }

        [Fact]
        public void SplitIntoSentences_WithQuestionMark_SplitsCorrectly()
        {
            // Arrange
            var text = "What is this? It is a test. Really?";

            // Act
            var result = TextProcessingHelper.SplitIntoSentences(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("What is this?", result[0]);
            Assert.Equal("It is a test.", result[1]);
            Assert.Equal("Really?", result[2]);
        }

        [Fact]
        public void SplitIntoSentences_WithNewlines_SplitsCorrectly()
        {
            // Arrange
            var text = "First line.\nSecond line.\nThird line.";

            // Act
            var result = TextProcessingHelper.SplitIntoSentences(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("First line.", result[0]);
            Assert.Equal("Second line.", result[1]);
            Assert.Equal("Third line.", result[2]);
        }

        [Fact]
        public void SplitIntoSentences_WithMixedPunctuation_SplitsCorrectly()
        {
            // Arrange
            var text = "Statement one. Question two? Exclamation three!";

            // Act
            var result = TextProcessingHelper.SplitIntoSentences(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("Statement one.", result[0]);
            Assert.Equal("Question two?", result[1]);
            Assert.Equal("Exclamation three!", result[2]);
        }

        [Fact]
        public void SplitIntoSentences_WithNoTrailingPunctuation_IncludesLastSentence()
        {
            // Arrange
            var text = "First sentence. Second sentence";

            // Act
            var result = TextProcessingHelper.SplitIntoSentences(text);

            // Assert
            Assert.Equal(2, result.Count);
            Assert.Equal("First sentence.", result[0]);
            Assert.Equal("Second sentence", result[1]);
        }

        [Fact]
        public void Tokenize_WithNull_ReturnsEmptyList()
        {
            // Act
            var result = TextProcessingHelper.Tokenize(null);

            // Assert
            Assert.NotNull(result);
            Assert.Empty(result);
        }

        [Fact]
        public void Tokenize_WithEmptyString_ReturnsEmptyList()
        {
            // Act
            var result = TextProcessingHelper.Tokenize(string.Empty);

            // Assert
            Assert.NotNull(result);
            Assert.Empty(result);
        }

        [Fact]
        public void Tokenize_WithSingleWord_ReturnsSingleToken()
        {
            // Arrange
            var text = "hello";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Single(result);
            Assert.Equal("hello", result[0]);
        }

        [Fact]
        public void Tokenize_WithMultipleWords_ReturnsAllTokens()
        {
            // Arrange
            var text = "hello world test";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("hello", result[0]);
            Assert.Equal("world", result[1]);
            Assert.Equal("test", result[2]);
        }

        [Fact]
        public void Tokenize_WithPunctuation_SplitsCorrectly()
        {
            // Arrange
            var text = "Hello, world!";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(2, result.Count);
            Assert.Equal("hello", result[0]);
            Assert.Equal("world", result[1]);
        }

        [Fact]
        public void Tokenize_WithUpperCase_ConvertsToLowerCase()
        {
            // Arrange
            var text = "HELLO WORLD";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(2, result.Count);
            Assert.Equal("hello", result[0]);
            Assert.Equal("world", result[1]);
        }

        [Fact]
        public void Tokenize_WithMixedCase_ConvertsToLowerCase()
        {
            // Arrange
            var text = "HeLLo WoRLd";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(2, result.Count);
            Assert.Equal("hello", result[0]);
            Assert.Equal("world", result[1]);
        }

        [Fact]
        public void Tokenize_WithTabs_SplitsCorrectly()
        {
            // Arrange
            var text = "hello\tworld\ttest";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("hello", result[0]);
            Assert.Equal("world", result[1]);
            Assert.Equal("test", result[2]);
        }

        [Fact]
        public void Tokenize_WithNewlines_SplitsCorrectly()
        {
            // Arrange
            var text = "hello\nworld\ntest";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("hello", result[0]);
            Assert.Equal("world", result[1]);
            Assert.Equal("test", result[2]);
        }

        [Fact]
        public void Tokenize_WithMultipleSpaces_IgnoresEmptyTokens()
        {
            // Arrange
            var text = "hello    world    test";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("hello", result[0]);
            Assert.Equal("world", result[1]);
            Assert.Equal("test", result[2]);
        }

        [Fact]
        public void Tokenize_WithSentence_RemovesPunctuation()
        {
            // Arrange
            var text = "This is a test. It works!";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(6, result.Count);
            Assert.Equal("this", result[0]);
            Assert.Equal("is", result[1]);
            Assert.Equal("a", result[2]);
            Assert.Equal("test", result[3]);
            Assert.Equal("it", result[4]);
            Assert.Equal("works", result[5]);
        }

        [Fact]
        public void Tokenize_WithQuestionMarks_RemovesThem()
        {
            // Arrange
            var text = "What? Why?";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(2, result.Count);
            Assert.Equal("what", result[0]);
            Assert.Equal("why", result[1]);
        }

        [Fact]
        public void Tokenize_WithCommas_RemovesThem()
        {
            // Arrange
            var text = "one, two, three";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("one", result[0]);
            Assert.Equal("two", result[1]);
            Assert.Equal("three", result[2]);
        }

        [Fact]
        public void SplitIntoSentences_WithLongText_HandlesCorrectly()
        {
            // Arrange
            var text = "This is the first sentence. This is the second sentence! " +
                       "Is this the third sentence? Yes, this is the fourth sentence.";

            // Act
            var result = TextProcessingHelper.SplitIntoSentences(text);

            // Assert
            Assert.Equal(4, result.Count);
            Assert.Contains("first sentence", result[0]);
            Assert.Contains("second sentence", result[1]);
            Assert.Contains("third sentence", result[2]);
            Assert.Contains("fourth sentence", result[3]);
        }

        [Fact]
        public void Tokenize_WithNumbers_IncludesThem()
        {
            // Arrange
            var text = "test 123 hello 456";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(4, result.Count);
            Assert.Contains("test", result);
            Assert.Contains("123", result);
            Assert.Contains("hello", result);
            Assert.Contains("456", result);
        }

        [Fact]
        public void SplitIntoSentences_WithConsecutivePunctuation_HandlesCorrectly()
        {
            // Arrange
            var text = "What?! Really! Yes.";

            // Act
            var result = TextProcessingHelper.SplitIntoSentences(text);

            // Assert
            Assert.NotEmpty(result);
            Assert.All(result, sentence => Assert.False(string.IsNullOrWhiteSpace(sentence)));
        }

        [Fact]
        public void Tokenize_WithHyphenatedWords_PreservesHyphenatedWords()
        {
            // Arrange
            var text = "state-of-the-art technology";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            // The Tokenize method does not split on hyphens, so hyphenated words are preserved
            Assert.Contains("state-of-the-art", result);
            Assert.Contains("technology", result);
        }

        [Fact]
        public void SplitIntoSentences_WithExtraSpaces_TrimsCorrectly()
        {
            // Arrange
            var text = "First.   Second.    Third.";

            // Act
            var result = TextProcessingHelper.SplitIntoSentences(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.All(result, sentence => Assert.False(sentence.StartsWith(" ")));
            Assert.All(result, sentence => Assert.False(sentence.EndsWith("  ")));
        }

        [Fact]
        public void Tokenize_WithCarriageReturn_SplitsCorrectly()
        {
            // Arrange
            var text = "hello\rworld\rtest";

            // Act
            var result = TextProcessingHelper.Tokenize(text);

            // Assert
            Assert.Equal(3, result.Count);
            Assert.Equal("hello", result[0]);
            Assert.Equal("world", result[1]);
            Assert.Equal("test", result[2]);
        }
    }
}
