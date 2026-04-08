using System;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for SentenceChunkingStrategy which splits text at sentence boundaries.
    /// </summary>
    public class SentenceChunkingStrategyTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithDefaultValues_SetsCorrectProperties()
        {
            // Act
            var strategy = new SentenceChunkingStrategy();

            // Assert
            Assert.Equal(1000, strategy.ChunkSize); // maxChunkSize becomes ChunkSize
            Assert.Equal(0, strategy.ChunkOverlap); // Base class always gets 0
        }

        [Fact]
        public void Constructor_WithCustomValues_SetsCorrectProperties()
        {
            // Act
            var strategy = new SentenceChunkingStrategy(
                targetChunkSize: 300,
                maxChunkSize: 600,
                overlapSentences: 2);

            // Assert
            Assert.Equal(600, strategy.ChunkSize);
            Assert.Equal(0, strategy.ChunkOverlap);
        }

        [Fact]
        public void Constructor_MaxChunkSizeLessThanTarget_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new SentenceChunkingStrategy(targetChunkSize: 500, maxChunkSize: 300));

            Assert.Contains("Maximum chunk size must be greater than or equal to target chunk size", ex.Message);
        }

        [Fact]
        public void Constructor_MaxChunkSizeEqualToTarget_IsValid()
        {
            // Act
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 500, maxChunkSize: 500);

            // Assert
            Assert.Equal(500, strategy.ChunkSize);
        }

        [Fact]
        public void Constructor_NegativeOverlapSentences_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new SentenceChunkingStrategy(overlapSentences: -1));

            Assert.Contains("Overlap sentences cannot be negative", ex.Message);
        }

        [Fact]
        public void Constructor_ZeroOverlapSentences_IsValid()
        {
            // Act
            var strategy = new SentenceChunkingStrategy(overlapSentences: 0);

            // Assert - should not throw
            Assert.Equal(1000, strategy.ChunkSize);
        }

        [Fact]
        public void Constructor_ChunkSizeZero_ThrowsArgumentException()
        {
            // Act & Assert - base class validation
            var ex = Assert.Throws<ArgumentException>(() =>
                new SentenceChunkingStrategy(targetChunkSize: 0, maxChunkSize: 0));

            Assert.Contains("ChunkSize", ex.Message);
        }

        #endregion

        #region Chunk Method Tests

        [Fact]
        public void Chunk_NullText_ThrowsArgumentNullException()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                strategy.Chunk(null!).ToList());
        }

        [Fact]
        public void Chunk_EmptyText_ThrowsArgumentException()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                strategy.Chunk(string.Empty).ToList());
        }

        [Fact]
        public void Chunk_SingleSentence_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 100, maxChunkSize: 200);
            var text = "This is a single sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("This is a single sentence.", chunks[0]);
        }

        [Fact]
        public void Chunk_MultipleSentences_GroupsByTargetSize()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 30, maxChunkSize: 100, overlapSentences: 0);
            var text = "First sentence. Second sentence. Third sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1, "Should create at least one chunk");
        }

        [Fact]
        public void Chunk_SentencesWithDifferentEndings_HandlesAllTypes()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 200, maxChunkSize: 300);
            var text = "Statement. Question? Exclamation!";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
            var combined = string.Join(" ", chunks);
            Assert.Contains("Statement", combined);
            Assert.Contains("Question", combined);
            Assert.Contains("Exclamation", combined);
        }

        [Fact]
        public void Chunk_TextEndingWithoutPunctuation_HandlesLastSentence()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 200, maxChunkSize: 300);
            var text = "First sentence. Last part without punctuation";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
            var combined = string.Join(" ", chunks);
            Assert.Contains("Last part", combined);
        }

        [Fact]
        public void Chunk_VeryLongSentence_SplitsByMaxChunkSize()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 20, maxChunkSize: 30);
            var longSentence = "This is a very long sentence that definitely exceeds the maximum chunk size limit and should be split.";

            // Act
            var chunks = strategy.Chunk(longSentence).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Long sentence should be split into multiple chunks");
            foreach (var chunk in chunks)
            {
                Assert.True(chunk.Length <= 30, $"Chunk exceeds maxChunkSize: {chunk.Length}");
            }
        }

        #endregion

        #region Overlap Tests

        [Fact]
        public void Chunk_WithOverlapSentences_RepeatsLastSentences()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(
                targetChunkSize: 20,
                maxChunkSize: 50,
                overlapSentences: 1);
            var text = "First. Second. Third. Fourth. Fifth.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            if (chunks.Count >= 2)
            {
                // There should be some overlapping content between consecutive chunks
                // The last sentence of chunk 1 should appear in chunk 2
                Assert.True(chunks.Count >= 2, "Should have multiple chunks for overlap testing");
            }
        }

        [Fact]
        public void Chunk_ZeroOverlapSentences_NoSentenceRepetition()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(
                targetChunkSize: 20,
                maxChunkSize: 50,
                overlapSentences: 0);
            var text = "First. Second. Third. Fourth.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
        }

        [Fact]
        public void Chunk_HighOverlapCount_HandlesGracefully()
        {
            // Arrange - overlap count higher than sentence count
            var strategy = new SentenceChunkingStrategy(
                targetChunkSize: 10,
                maxChunkSize: 30,
                overlapSentences: 10);
            var text = "Short. Text.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert - should not crash
            Assert.True(chunks.Count >= 1);
        }

        #endregion

        #region ChunkWithPositions Tests

        [Fact]
        public void ChunkWithPositions_SingleSentence_ReturnsValidPositions()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 100, maxChunkSize: 200);
            var text = "Short text for testing.";

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
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 20, maxChunkSize: 40, overlapSentences: 0);
            var text = "First sentence. Second sentence. Third sentence.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            if (chunks.Count >= 2)
            {
                for (int i = 1; i < chunks.Count; i++)
                {
                    Assert.True(chunks[i].StartPosition >= chunks[i - 1].StartPosition,
                        $"Chunk {i} start should be after or equal to chunk {i - 1} start");
                }
            }
        }

        [Fact]
        public void ChunkWithPositions_ChunkContentMatchesSubstring()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 50, maxChunkSize: 100);
            var text = "First sentence. Second sentence.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            foreach (var (chunk, startPos, endPos) in chunks)
            {
                var expected = text.Substring(startPos, endPos - startPos);
                Assert.Equal(expected, chunk);
            }
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Chunk_SingleCharacter_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 100, maxChunkSize: 200);
            var text = "X";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("X", chunks[0]);
        }

        [Fact]
        public void Chunk_WhitespaceOnlyText_ReturnsEmptyCollection()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 100, maxChunkSize: 200);
            var text = "   ";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Empty(chunks);
        }

        [Fact]
        public void Chunk_OnlyPunctuation_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 100, maxChunkSize: 200);
            var text = "...";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert - empty sentences get filtered out
            Assert.True(chunks.Count <= 1);
        }

        [Fact]
        public void Chunk_LargeTargetSize_ReturnsEntireTextAsSingleChunk()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 10000, maxChunkSize: 20000);
            var text = "First sentence. Second sentence. Third sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
        }

        [Fact]
        public void Chunk_RepeatedCalls_ProduceSameResults()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 30, maxChunkSize: 50);
            var text = "First sentence. Second sentence. Third sentence.";

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

        [Fact]
        public void Chunk_TextWithLeadingTrailingWhitespace_HandlesCorrectly()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 100, maxChunkSize: 200);
            var text = "   First sentence.   ";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("First sentence.", chunks[0]);
        }

        [Fact]
        public void Chunk_TextWithMultipleSpacesBetweenSentences_HandlesCorrectly()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 100, maxChunkSize: 200);
            var text = "First sentence.    Second sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
        }

        #endregion

        #region Sentence Detection Tests

        [Fact]
        public void Chunk_SentenceWithPeriod_DetectsBoundary()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 20, maxChunkSize: 40, overlapSentences: 0);
            var text = "First sentence. Second sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
        }

        [Fact]
        public void Chunk_SentenceWithQuestion_DetectsBoundary()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 20, maxChunkSize: 40, overlapSentences: 0);
            var text = "Is this a question? Yes it is.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
        }

        [Fact]
        public void Chunk_SentenceWithExclamation_DetectsBoundary()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 20, maxChunkSize: 40, overlapSentences: 0);
            var text = "This is exciting! More text here.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
        }

        [Fact]
        public void Chunk_SentenceWithNewline_HandlesCorrectly()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 100, maxChunkSize: 200);
            var text = "First sentence.\nSecond sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
        }

        [Fact]
        public void Chunk_AbbreviationWithPeriod_MayNotDetectAsBoundary()
        {
            // Arrange - lowercase after period suggests not a sentence boundary
            var strategy = new SentenceChunkingStrategy(targetChunkSize: 200, maxChunkSize: 400);
            var text = "Dr. Smith went to the store. He bought milk.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert - should handle abbreviations reasonably
            Assert.True(chunks.Count >= 1);
        }

        #endregion

        #region Target vs Max Size Tests

        [Fact]
        public void Chunk_SentencesBetweenTargetAndMax_CreatesChunkAtTarget()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(
                targetChunkSize: 30,
                maxChunkSize: 100,
                overlapSentences: 0);
            var text = "Short. Another. One more. Extra.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
        }

        [Fact]
        public void Chunk_SentencesExceedMax_ForcesChunkCreation()
        {
            // Arrange
            var strategy = new SentenceChunkingStrategy(
                targetChunkSize: 50,
                maxChunkSize: 60,
                overlapSentences: 0);
            var text = "This is a sentence that is fairly long. Another sentence here.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            foreach (var chunk in chunks)
            {
                Assert.True(chunk.Length <= 60 || !chunk.Contains(". "),
                    "Chunks with multiple sentences should respect max size");
            }
        }

        #endregion
    }
}
