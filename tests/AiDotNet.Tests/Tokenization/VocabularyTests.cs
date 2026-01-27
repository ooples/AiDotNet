using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tokenization.Vocabulary;
using Xunit;

namespace AiDotNet.Tests.Tokenization
{
    public class VocabularyTests
    {
        [Fact]
        public void Constructor_CreatesVocabularyWithUnkToken()
        {
            // Arrange & Act
            var vocab = new Vocabulary("[UNK]");

            // Assert
            Assert.Equal(1, vocab.Size);
            Assert.True(vocab.ContainsToken("[UNK]"));
        }

        [Fact]
        public void AddToken_AddsNewToken()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");

            // Act
            var id = vocab.AddToken("hello");

            // Assert
            Assert.Equal(2, vocab.Size);
            Assert.True(vocab.ContainsToken("hello"));
            Assert.Equal(id, vocab.GetTokenId("hello"));
        }

        [Fact]
        public void AddToken_ReturnsSameIdForDuplicateToken()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            var id1 = vocab.AddToken("hello");

            // Act
            var id2 = vocab.AddToken("hello");

            // Assert
            Assert.Equal(id1, id2);
            Assert.Equal(2, vocab.Size);
        }

        [Fact]
        public void GetTokenId_ReturnsUnkIdForUnknownToken()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            var unkId = vocab.GetTokenId("[UNK]");

            // Act
            var unknownId = vocab.GetTokenId("unknown");

            // Assert
            Assert.Equal(unkId, unknownId);
        }

        [Fact]
        public void GetToken_ReturnsTokenForValidId()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            var id = vocab.AddToken("hello");

            // Act
            var token = vocab.GetToken(id);

            // Assert
            Assert.Equal("hello", token);
        }

        [Fact]
        public void GetToken_ReturnsNullForInvalidId()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");

            // Act
            var token = vocab.GetToken(999);

            // Assert
            Assert.Null(token);
        }

        [Fact]
        public void AddTokens_AddsMultipleTokens()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            var tokens = new[] { "hello", "world", "test" };

            // Act
            vocab.AddTokens(tokens);

            // Assert
            Assert.Equal(4, vocab.Size); // [UNK] + 3 tokens
            Assert.True(vocab.ContainsToken("hello"));
            Assert.True(vocab.ContainsToken("world"));
            Assert.True(vocab.ContainsToken("test"));
        }

        [Fact]
        public void GetAllTokens_ReturnsAllTokens()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "hello", "world" });

            // Act
            var allTokens = vocab.GetAllTokens().ToList();

            // Assert
            Assert.Equal(3, allTokens.Count);
            Assert.Contains("[UNK]", allTokens);
            Assert.Contains("hello", allTokens);
            Assert.Contains("world", allTokens);
        }

        [Fact]
        public void Clear_RemovesAllTokens()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "hello", "world" });

            // Act
            vocab.Clear();

            // Assert - Clear() re-adds UNK token to maintain consistency
            Assert.Equal(1, vocab.Size);
            Assert.True(vocab.ContainsToken("[UNK]"));
        }

        #region PR #757 Bug Fix Tests - Parameter Validation

        [Fact]
        public void Constructor_FromDictionary_ThrowsOnNullTokenToId()
        {
            Assert.Throws<ArgumentNullException>(() =>
                new Vocabulary((Dictionary<string, int>)null!));
        }

        [Fact]
        public void AddTokens_ThrowsOnNullTokens()
        {
            var vocab = new Vocabulary("[UNK]");

            Assert.Throws<ArgumentNullException>(() =>
                vocab.AddTokens(null!));
        }

        #endregion
    }
}
