using System.Collections.Generic;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Vocabulary;
using Xunit;

namespace AiDotNet.Tests.Tokenization
{
    public class WordPieceTokenizerTests
    {
        [Fact]
        public void Train_CreatesTokenizerWithSubwords()
        {
            // Arrange
            var corpus = new List<string>
            {
                "hello world",
                "hello there",
                "world peace"
            };

            // Act
            var tokenizer = WordPieceTokenizer.Train(corpus, vocabSize: 100, specialTokens: SpecialTokens.Bert());

            // Assert
            Assert.NotNull(tokenizer);
            Assert.True(tokenizer.VocabularySize > 0);
        }

        [Fact]
        public void Tokenize_SplitsTextIntoSubwords()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "[PAD]", "[CLS]", "[SEP]", "[MASK]" });
            vocab.AddTokens(new[] { "hello", "world", "##ing", "##ed" });

            var tokenizer = new WordPieceTokenizer(vocab, SpecialTokens.Bert());

            // Act
            var tokens = tokenizer.Tokenize("hello world");

            // Assert
            Assert.NotEmpty(tokens);
            Assert.Contains("hello", tokens);
            Assert.Contains("world", tokens);
        }

        [Fact]
        public void Tokenize_HandlesUnknownWords()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "hello", "world" });

            var tokenizer = new WordPieceTokenizer(vocab, SpecialTokens.Bert());

            // Act
            var tokens = tokenizer.Tokenize("hello unknownword");

            // Assert
            Assert.Contains("[UNK]", tokens);
        }

        [Fact]
        public void Encode_AddsSpecialTokens()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "[PAD]", "[CLS]", "[SEP]", "[MASK]", "hello", "world" });

            var tokenizer = new WordPieceTokenizer(vocab, SpecialTokens.Bert());

            var options = new EncodingOptions
            {
                AddSpecialTokens = true
            };

            // Act
            var result = tokenizer.Encode("hello world", options);

            // Assert
            Assert.Contains("[CLS]", result.Tokens);
            Assert.Contains("[SEP]", result.Tokens);
        }

        [Fact]
        public void Decode_ReconstructsText()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "[PAD]", "[CLS]", "[SEP]", "hello", "##ing" });

            var tokenizer = new WordPieceTokenizer(vocab, SpecialTokens.Bert());

            var tokens = new List<string> { "hello", "##ing" };
            var tokenIds = tokenizer.ConvertTokensToIds(tokens);

            // Act
            var decoded = tokenizer.Decode(tokenIds, skipSpecialTokens: true);

            // Assert
            Assert.Equal("helloing", decoded);
        }

        [Fact]
        public void Decode_SkipsSpecialTokens()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "[PAD]", "[CLS]", "[SEP]", "hello", "world" });

            var tokenizer = new WordPieceTokenizer(vocab, SpecialTokens.Bert());

            var tokens = new List<string> { "[CLS]", "hello", "world", "[SEP]" };
            var tokenIds = tokenizer.ConvertTokensToIds(tokens);

            // Act
            var decoded = tokenizer.Decode(tokenIds, skipSpecialTokens: true);

            // Assert
            Assert.DoesNotContain("[CLS]", decoded);
            Assert.DoesNotContain("[SEP]", decoded);
            Assert.Contains("hello", decoded);
            Assert.Contains("world", decoded);
        }

        [Fact]
        public void EncodeBatch_EncodesMultipleTexts()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "[PAD]", "[CLS]", "[SEP]", "hello", "world", "test" });

            var tokenizer = new WordPieceTokenizer(vocab, SpecialTokens.Bert());

            var texts = new List<string> { "hello world", "test" };

            // Act
            var results = tokenizer.EncodeBatch(texts);

            // Assert
            Assert.Equal(2, results.Count);
            Assert.NotEmpty(results[0].TokenIds);
            Assert.NotEmpty(results[1].TokenIds);
        }
    }
}
