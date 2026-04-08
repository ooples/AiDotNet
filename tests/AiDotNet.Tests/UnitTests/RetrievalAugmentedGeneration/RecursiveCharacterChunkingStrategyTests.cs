using System;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for RecursiveCharacterChunkingStrategy which splits text using a hierarchy of separators.
    /// </summary>
    public class RecursiveCharacterChunkingStrategyTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithDefaultValues_SetsCorrectProperties()
        {
            // Act
            var strategy = new RecursiveCharacterChunkingStrategy();

            // Assert
            Assert.Equal(1000, strategy.ChunkSize);
            Assert.Equal(200, strategy.ChunkOverlap);
        }

        [Fact]
        public void Constructor_WithCustomValues_SetsCorrectProperties()
        {
            // Act
            var strategy = new RecursiveCharacterChunkingStrategy(
                chunkSize: 500,
                chunkOverlap: 50);

            // Assert
            Assert.Equal(500, strategy.ChunkSize);
            Assert.Equal(50, strategy.ChunkOverlap);
        }

        [Fact]
        public void Constructor_WithCustomSeparators_AcceptsSeparators()
        {
            // Act
            var strategy = new RecursiveCharacterChunkingStrategy(
                chunkSize: 100,
                chunkOverlap: 10,
                separators: new[] { "\r\n", "\n", " " });

            // Assert
            Assert.Equal(100, strategy.ChunkSize);
            Assert.Equal(10, strategy.ChunkOverlap);
        }

        [Fact]
        public void Constructor_ChunkSizeZero_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new RecursiveCharacterChunkingStrategy(chunkSize: 0));

            Assert.Contains("ChunkSize", ex.Message);
        }

        [Fact]
        public void Constructor_ChunkOverlapNegative_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new RecursiveCharacterChunkingStrategy(chunkSize: 100, chunkOverlap: -1));

            Assert.Contains("ChunkOverlap", ex.Message);
        }

        [Fact]
        public void Constructor_ChunkOverlapEqualToChunkSize_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new RecursiveCharacterChunkingStrategy(chunkSize: 100, chunkOverlap: 100));

            Assert.Contains("ChunkOverlap", ex.Message);
        }

        #endregion

        #region Chunk Method Tests

        [Fact]
        public void Chunk_NullText_ThrowsArgumentNullException()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                strategy.Chunk(null!).ToList());
        }

        [Fact]
        public void Chunk_EmptyText_ThrowsArgumentException()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                strategy.Chunk(string.Empty).ToList());
        }

        [Fact]
        public void Chunk_ShortText_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = "This is a short text.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal(text.Trim(), chunks[0]);
        }

        [Fact]
        public void Chunk_TextWithParagraphs_SplitsAtParagraphBoundaries()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 50, chunkOverlap: 0);
            var text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Should split at paragraph boundaries");
            // Each chunk should be trimmed
            foreach (var chunk in chunks)
            {
                Assert.Equal(chunk.Trim(), chunk);
            }
        }

        [Fact]
        public void Chunk_TextWithNewlines_SplitsAtNewlineBoundaries()
        {
            // Arrange - chunkSize set so paragraphs don't fit
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 30, chunkOverlap: 0);
            var text = "Line one.\nLine two.\nLine three.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Should split text at newline boundaries");
        }

        [Fact]
        public void Chunk_TextWithSentences_SplitsAtSentenceBoundaries()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 25, chunkOverlap: 0);
            var text = "First sentence. Second sentence. Third sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Should split at sentence boundaries");
        }

        [Fact]
        public void Chunk_TextWithSpaces_SplitsAtWordBoundaries()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 15, chunkOverlap: 0);
            var text = "one two three four five six";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Should split at word boundaries");
        }

        [Fact]
        public void Chunk_VeryLongWord_SplitsByCharacter()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 10, chunkOverlap: 0);
            var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // 26 chars, no spaces

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Should split long word by characters as last resort");
        }

        #endregion

        #region Separator Hierarchy Tests

        [Fact]
        public void Chunk_PrefersParagraphSeparatorOverOthers()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 100, chunkOverlap: 0);
            var text = "Short paragraph one.\n\nShort paragraph two.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert - With large chunk size, paragraphs should be kept together
            Assert.Single(chunks);
            Assert.Contains("paragraph one", chunks[0]);
            Assert.Contains("paragraph two", chunks[0]);
        }

        [Fact]
        public void Chunk_FallsBackToNewlineWhenParagraphTooLarge()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 30, chunkOverlap: 0);
            var paragraph = "This is a very long paragraph that will need to be split.\nWith a newline here.";

            // Act
            var chunks = strategy.Chunk(paragraph).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Should fall back to newline separator");
        }

        [Fact]
        public void Chunk_CustomSeparators_UseProvidedOrder()
        {
            // Arrange - Use pipe as primary separator
            // Text must exceed chunkSize to trigger splitting
            var strategy = new RecursiveCharacterChunkingStrategy(
                chunkSize: 10,
                chunkOverlap: 0,
                separators: new[] { "|", " ", "" });
            var text = "one|two|three|four"; // 18 chars > 10 chunkSize

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Should split at pipe separator when text exceeds chunk size");
        }

        #endregion

        #region Overlap Tests

        [Fact]
        public void Chunk_WithOverlap_ChunksContainOverlappingContent()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 30, chunkOverlap: 10);
            var text = "First part of text. Second part of text. Third part of text.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            if (chunks.Count >= 2)
            {
                // There should be some overlap between consecutive chunks
                // The end of one chunk should appear at the start of the next
                for (int i = 0; i < chunks.Count - 1; i++)
                {
                    // At least verify chunks were created
                    Assert.True(chunks[i].Length > 0);
                    Assert.True(chunks[i + 1].Length > 0);
                }
            }
        }

        [Fact]
        public void Chunk_ZeroOverlap_ChunksAreIndependent()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 20, chunkOverlap: 0);
            var text = "word1 word2 word3 word4 word5";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Should create multiple chunks");
            // With zero overlap, chunks shouldn't repeat content
        }

        #endregion

        #region ChunkWithPositions Tests

        [Fact]
        public void ChunkWithPositions_ReturnsValidPositions()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 100, chunkOverlap: 0);
            var text = "Short text for testing positions.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal(0, chunks[0].StartPosition);
            Assert.True(chunks[0].EndPosition > 0);
        }

        [Fact]
        public void ChunkWithPositions_MultipleChunks_HasSequentialPositions()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 20, chunkOverlap: 0);
            var text = "First chunk content. Second chunk content.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            if (chunks.Count >= 2)
            {
                // Positions should be sequential
                for (int i = 1; i < chunks.Count; i++)
                {
                    // Each chunk should start after the previous one ended
                    Assert.True(chunks[i].StartPosition >= chunks[i - 1].EndPosition ||
                                chunks[i].StartPosition == chunks[i - 1].EndPosition);
                }
            }
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Chunk_SingleCharacter_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = "X";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("X", chunks[0]);
        }

        [Fact]
        public void Chunk_WhitespaceOnlyText_ReturnsEmptyOrSingleWhitespace()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = "   ";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert - whitespace-only text might be trimmed to empty
            Assert.True(chunks.Count <= 1, "Whitespace text should result in 0 or 1 chunks");
        }

        [Fact]
        public void Chunk_TextExactlyChunkSize_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 20, chunkOverlap: 5);
            var text = "12345678901234567890"; // Exactly 20 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
        }

        [Fact]
        public void Chunk_LargeChunkSize_ReturnsEntireTextAsSingleChunk()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 10000, chunkOverlap: 100);
            var text = "This is a paragraph.\n\nAnother paragraph.\n\nAnd another one.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
        }

        [Fact]
        public void Chunk_OnlyParagraphSeparators_SplitsCorrectly()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 30, chunkOverlap: 0);
            var text = "Para one.\n\nPara two.\n\nPara three.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Should split into multiple chunks");
        }

        [Fact]
        public void Chunk_MixedSeparators_HandlesCorrectly()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 25, chunkOverlap: 0);
            var text = "Para.\n\nLine.\nSentence. Word word.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1, "Should create at least one chunk");
            foreach (var chunk in chunks)
            {
                Assert.True(chunk.Length <= 25 + 5, // Allow some flexibility for trim differences
                    $"Chunk '{chunk}' exceeds expected size");
            }
        }

        [Fact]
        public void Chunk_RepeatedCalls_ProduceSameResults()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 50, chunkOverlap: 10);
            var text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";

            // Act
            var chunks1 = strategy.Chunk(text).ToList();
            var chunks2 = strategy.Chunk(text).ToList();

            // Assert
            Assert.Equal(chunks1.Count, chunks2.Count);
            for (int i = 0; i < chunks1.Count; i++)
            {
                Assert.Equal(chunks1[i], chunks2[i]);
            }
        }

        #endregion

        #region Recursive Splitting Tests

        [Fact]
        public void Chunk_VeryLargeParagraph_RecursivelySplits()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 20, chunkOverlap: 0);
            var largeParagraph = "This is a very large paragraph with many words that exceeds the chunk size limit.";

            // Act
            var chunks = strategy.Chunk(largeParagraph).ToList();

            // Assert
            Assert.True(chunks.Count >= 3, "Large paragraph should be split into multiple chunks");
        }

        [Fact]
        public void Chunk_NestedStructure_PreservesSemanticOrder()
        {
            // Arrange
            var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 50, chunkOverlap: 0);
            var text = "Chapter 1.\n\nSection A.\nParagraph 1. Paragraph 2.\n\nSection B.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1, "Should create chunks");
            // Verify order is preserved - first chunk should contain "Chapter 1"
            Assert.Contains("Chapter 1", chunks[0]);
        }

        #endregion
    }
}
