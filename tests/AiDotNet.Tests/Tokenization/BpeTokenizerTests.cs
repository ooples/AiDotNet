using System.Collections.Generic;
using AiDotNet.Tokenization.Algorithms;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Vocabulary;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.Tokenization
{
    public class BpeTokenizerTests
    {
        [Fact(Timeout = 60000)]
        public async Task Train_CreatesTokenizerWithMerges()
        {
            // Arrange
            var corpus = new List<string>
            {
                "hello world",
                "hello there",
                "world peace"
            };

            // Act
            var tokenizer = BpeTokenizer.Train(corpus, vocabSize: 50, specialTokens: SpecialTokens.Gpt());

            // Assert
            Assert.NotNull(tokenizer);
            Assert.True(tokenizer.VocabularySize > 0);
        }

        [Fact(Timeout = 60000)]
        public async Task Tokenize_SplitsTextIntoTokens()
        {
            // Arrange
            var vocab = new Vocabulary("[UNK]");
            vocab.AddTokens(new[] { "h", "e", "l", "o", " ", "w", "r", "d", "hello", "world" });

            var merges = new Dictionary<(string, string), int>
            {
                { ("h", "e"), 0 },
                { ("he", "l"), 1 },
                { ("hel", "l"), 2 },
                { ("hell", "o"), 3 }
            };

            var tokenizer = new BpeTokenizer(vocab, merges, SpecialTokens.Gpt());

            // Act
            var tokens = tokenizer.Tokenize("hello");

            // Assert
            Assert.NotEmpty(tokens);
            Assert.Contains("hello", tokens);
        }

        [Fact(Timeout = 60000)]
        public async Task Encode_ReturnsTokenizationResult()
        {
            // Arrange
            var vocab = new Vocabulary("<|endoftext|>");
            vocab.AddTokens(new[] { "h", "e", "l", "o", " ", "w", "r", "d" });

            var merges = new Dictionary<(string, string), int>();
            var tokenizer = new BpeTokenizer(vocab, merges, SpecialTokens.Gpt());

            var options = new EncodingOptions
            {
                AddSpecialTokens = true,
                Padding = false
            };

            // Act
            var result = tokenizer.Encode("hello", options);

            // Assert
            Assert.NotNull(result);
            Assert.NotEmpty(result.TokenIds);
            Assert.Equal(result.TokenIds.Count, result.Tokens.Count);
        }

        [Fact(Timeout = 60000)]
        public async Task Decode_ReconstructsText()
        {
            // Arrange
            var vocab = new Vocabulary("<|endoftext|>");
            vocab.AddTokens(new[] { "h", "e", "l", "o", " ", "w", "r", "d" });

            var merges = new Dictionary<(string, string), int>();
            var tokenizer = new BpeTokenizer(vocab, merges, SpecialTokens.Gpt());

            var text = "hello";
            var encoded = tokenizer.Encode(text, new EncodingOptions { AddSpecialTokens = false });

            // Act
            var decoded = tokenizer.Decode(encoded.TokenIds, skipSpecialTokens: true);

            // Assert
            Assert.Equal(text, decoded);
        }

        [Fact(Timeout = 60000)]
        public async Task Encode_WithPadding_AddsPaddingTokens()
        {
            // Arrange
            var vocab = new Vocabulary("<|endoftext|>");
            vocab.AddTokens(new[] { "h", "e", "l", "o" });

            var merges = new Dictionary<(string, string), int>();
            var tokenizer = new BpeTokenizer(vocab, merges, SpecialTokens.Gpt());

            var options = new EncodingOptions
            {
                AddSpecialTokens = false,
                Padding = true,
                MaxLength = 10
            };

            // Act
            var result = tokenizer.Encode("hello", options);

            // Assert
            Assert.Equal(10, result.TokenIds.Count);
            Assert.Equal(10, result.AttentionMask.Count);
            Assert.Contains(0, result.AttentionMask); // Has padding
        }

        [Fact(Timeout = 60000)]
        public async Task Encode_WithTruncation_TruncatesSequence()
        {
            // Arrange
            var vocab = new Vocabulary("<|endoftext|>");
            vocab.AddTokens(new[] { "h", "e", "l", "o", " ", "w", "r", "d" });

            var merges = new Dictionary<(string, string), int>();
            var tokenizer = new BpeTokenizer(vocab, merges, SpecialTokens.Gpt());

            var options = new EncodingOptions
            {
                AddSpecialTokens = false,
                Truncation = true,
                MaxLength = 3
            };

            // Act
            var result = tokenizer.Encode("hello world", options);

            // Assert
            // Verify truncation: token count should be exactly maxLength (or less if input is shorter)
            Assert.True(result.TokenIds.Count <= 3, $"Token count {result.TokenIds.Count} should be <= 3");
            // Verify that truncation actually happened if the original would have been longer
            var untruncatedResult = tokenizer.Encode("hello world", new EncodingOptions { AddSpecialTokens = false });
            if (untruncatedResult.TokenIds.Count > 3)
            {
                Assert.Equal(3, result.TokenIds.Count);
            }
        }
    }
}
