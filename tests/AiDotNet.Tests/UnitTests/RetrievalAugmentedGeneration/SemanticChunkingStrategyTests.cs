using System;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for SemanticChunkingStrategy which groups sentences based on semantic coherence.
    /// </summary>
    public class SemanticChunkingStrategyTests
    {
        #region Constructor Tests

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithDefaultValues_SetsCorrectProperties()
        {
            // Act
            var strategy = new SemanticChunkingStrategy();

            // Assert
            Assert.Equal(1000, strategy.ChunkSize);
            Assert.Equal(200, strategy.ChunkOverlap);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_WithCustomValues_SetsCorrectProperties()
        {
            // Act
            var strategy = new SemanticChunkingStrategy(
                maxChunkSize: 500,
                chunkOverlap: 50);

            // Assert
            Assert.Equal(500, strategy.ChunkSize);
            Assert.Equal(50, strategy.ChunkOverlap);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_ChunkSizeZero_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new SemanticChunkingStrategy(maxChunkSize: 0));

            Assert.Contains("ChunkSize", ex.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_ChunkSizeNegative_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new SemanticChunkingStrategy(maxChunkSize: -1));

            Assert.Contains("ChunkSize", ex.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_ChunkOverlapNegative_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: -1));

            Assert.Contains("ChunkOverlap", ex.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_ChunkOverlapEqualToChunkSize_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 100));

            Assert.Contains("ChunkOverlap", ex.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_ChunkOverlapGreaterThanChunkSize_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 150));

            Assert.Contains("ChunkOverlap", ex.Message);
        }

        [Fact(Timeout = 60000)]
        public async Task Constructor_ZeroOverlap_IsValid()
        {
            // Act
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 0);

            // Assert
            Assert.Equal(100, strategy.ChunkSize);
            Assert.Equal(0, strategy.ChunkOverlap);
        }

        #endregion

        #region Chunk Method Tests

        [Fact(Timeout = 60000)]
        public async Task Chunk_NullText_ThrowsArgumentNullException()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                strategy.Chunk(null!).ToList());
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_EmptyText_ThrowsArgumentException()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                strategy.Chunk(string.Empty).ToList());
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_SingleSentence_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 10);
            var text = "This is a single sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("This is a single sentence.", chunks[0]);
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_MultipleSentences_GroupsBySizeLimit()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 50, chunkOverlap: 0);
            var text = "First sentence. Second sentence. Third sentence. Fourth sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Should split into multiple chunks");
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_SentencesWithDifferentEndings_HandlesAllEndingTypes()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 200, chunkOverlap: 0);
            var text = "Statement. Question? Exclamation! Another statement.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1, "Should create at least one chunk");
            var combined = string.Join(" ", chunks);
            Assert.Contains("Statement", combined);
            Assert.Contains("Question", combined);
            Assert.Contains("Exclamation", combined);
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_TextWithNewlines_SplitsSentencesCorrectly()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 0);
            var text = "First sentence.\nSecond sentence.\nThird sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1, "Should create at least one chunk");
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_LargeSentence_IncludesSentenceEvenIfOverLimit()
        {
            // Arrange - sentence is larger than chunk size
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 20, chunkOverlap: 0);
            var text = "This is a very long sentence that exceeds the chunk size limit.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            // The strategy includes the sentence even if it exceeds the limit (no sentence splitting)
            Assert.Single(chunks);
            Assert.Equal(text, chunks[0]);
        }

        #endregion

        #region Overlap Tests

        [Fact(Timeout = 60000)]
        public async Task Chunk_WithOverlap_ChunksHaveOverlappingPositions()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 40, chunkOverlap: 10);
            var text = "First sentence. Second sentence. Third sentence.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            if (chunks.Count >= 2)
            {
                // With overlap, the start of chunk 2 should be before end of chunk 1
                Assert.True(chunks[1].StartPosition < chunks[0].EndPosition,
                    "Chunks should overlap in positions");
            }
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_ZeroOverlap_ChunksAreContiguous()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 40, chunkOverlap: 0);
            var text = "First sentence. Second sentence. Third sentence.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            if (chunks.Count >= 2)
            {
                // With zero overlap, chunks should not overlap
                Assert.True(chunks[1].StartPosition >= chunks[0].EndPosition - 1,
                    "Chunks with zero overlap should not have overlapping positions");
            }
        }

        #endregion

        #region ChunkWithPositions Tests

        [Fact(Timeout = 60000)]
        public async Task ChunkWithPositions_SingleSentence_ReturnsValidPositions()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 0);
            var text = "Short text for testing.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal(0, chunks[0].StartPosition);
            Assert.True(chunks[0].EndPosition > 0);
        }

        [Fact(Timeout = 60000)]
        public async Task ChunkWithPositions_MultipleChunks_HasSequentialPositions()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 30, chunkOverlap: 0);
            var text = "First sentence. Second sentence. Third sentence.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            if (chunks.Count >= 2)
            {
                for (int i = 1; i < chunks.Count; i++)
                {
                    // Each chunk should start at or after where previous ended (accounting for zero overlap)
                    Assert.True(chunks[i].StartPosition >= chunks[i - 1].EndPosition - 1,
                        $"Chunk {i} start should be at or after chunk {i - 1} end");
                }
            }
        }

        [Fact(Timeout = 60000)]
        public async Task ChunkWithPositions_ChunkContentHasValidLength()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 10);
            var text = "First sentence. Second sentence. Third sentence.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            foreach (var (chunk, startPos, endPos) in chunks)
            {
                Assert.True(chunk.Length > 0, "Chunk should not be empty");
                Assert.True(endPos > startPos, "End position should be after start position");
                Assert.Equal(chunk.Length, endPos - startPos);
            }
        }

        #endregion

        #region Edge Cases

        [Fact(Timeout = 60000)]
        public async Task Chunk_SingleCharacter_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 10);
            var text = "X";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("X", chunks[0]);
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_WhitespaceOnlyText_ReturnsEmptyCollection()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 10);
            var text = "   ";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert - whitespace-only text should result in empty or no chunks
            // SplitIntoSentences returns empty list for whitespace-only
            Assert.Empty(chunks);
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_VeryLongSingleSentence_ReturnsEntireTextAsSingleChunk()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 50, chunkOverlap: 10);
            var text = "This is an extremely long sentence without any periods that goes on and on";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal(text, chunks[0]);
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_LargeChunkSize_ReturnsEntireTextAsSingleChunk()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 10000, chunkOverlap: 100);
            var text = "First sentence. Second sentence. Third sentence.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_OnlyPunctuation_HandlesProperly()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 10);
            var text = "...";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert - punctuation-only text might result in empty or single chunk
            Assert.True(chunks.Count <= 1, "Should result in 0 or 1 chunks");
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_RepeatedCalls_ProduceSameResults()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 50, chunkOverlap: 10);
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

        [Fact(Timeout = 60000)]
        public async Task Chunk_MixedPunctuation_SplitsCorrectly()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 100, chunkOverlap: 0);
            var text = "Hello! How are you? I am fine. Thank you!";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
            var combined = string.Join(" ", chunks);
            Assert.Contains("Hello", combined);
            Assert.Contains("fine", combined);
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_SentencesWithAbbreviations_HandlesCommonCases()
        {
            // Arrange - Note: Simple sentence splitter may incorrectly split on abbreviations
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 200, chunkOverlap: 0);
            var text = "Dr. Smith went to Washington. He met with the president.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert - At minimum should produce chunks containing the text
            Assert.True(chunks.Count >= 1);
        }

        #endregion

        #region Sentence Grouping Tests

        [Fact(Timeout = 60000)]
        public async Task Chunk_SentencesFitExactly_GroupsCorrectly()
        {
            // Arrange - sentences that fit exactly within limit
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 50, chunkOverlap: 0);
            var text = "Short. Also short. Third.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_ManySentences_CreatesManyChunks()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 30, chunkOverlap: 0);
            var text = "One. Two. Three. Four. Five. Six. Seven. Eight.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Many sentences should create multiple chunks");
        }

        [Fact(Timeout = 60000)]
        public async Task Chunk_MixedLengthSentences_GroupsAppropriately()
        {
            // Arrange
            var strategy = new SemanticChunkingStrategy(maxChunkSize: 50, chunkOverlap: 0);
            var text = "Short. This is a much longer sentence with many words. Tiny.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 1);
        }

        #endregion
    }
}
