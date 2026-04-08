using System;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for FixedSizeChunkingStrategy which splits text into fixed-size character-based chunks.
    /// </summary>
    public class FixedSizeChunkingStrategyTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithDefaultValues_SetsCorrectProperties()
        {
            // Act
            var strategy = new FixedSizeChunkingStrategy();

            // Assert
            Assert.Equal(500, strategy.ChunkSize);
            Assert.Equal(50, strategy.ChunkOverlap);
        }

        [Fact]
        public void Constructor_WithCustomValues_SetsCorrectProperties()
        {
            // Act
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 1000, chunkOverlap: 100);

            // Assert
            Assert.Equal(1000, strategy.ChunkSize);
            Assert.Equal(100, strategy.ChunkOverlap);
        }

        [Fact]
        public void Constructor_ChunkSizeZero_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new FixedSizeChunkingStrategy(chunkSize: 0, chunkOverlap: 10));

            Assert.Contains("ChunkSize", ex.Message);
        }

        [Fact]
        public void Constructor_ChunkSizeNegative_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new FixedSizeChunkingStrategy(chunkSize: -1, chunkOverlap: 10));

            Assert.Contains("ChunkSize", ex.Message);
        }

        [Fact]
        public void Constructor_ChunkOverlapNegative_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: -1));

            Assert.Contains("ChunkOverlap", ex.Message);
        }

        [Fact]
        public void Constructor_ChunkOverlapEqualToChunkSize_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 100));

            Assert.Contains("ChunkOverlap", ex.Message);
        }

        [Fact]
        public void Constructor_ChunkOverlapGreaterThanChunkSize_ThrowsArgumentException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentException>(() =>
                new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 150));

            Assert.Contains("ChunkOverlap", ex.Message);
        }

        [Fact]
        public void Constructor_ZeroOverlap_IsValid()
        {
            // Act
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 0);

            // Assert
            Assert.Equal(100, strategy.ChunkSize);
            Assert.Equal(0, strategy.ChunkOverlap);
        }

        #endregion

        #region Chunk Method Tests

        [Fact]
        public void Chunk_NullText_ThrowsArgumentNullException()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                strategy.Chunk(null!).ToList());
        }

        [Fact]
        public void Chunk_EmptyText_ThrowsArgumentException()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                strategy.Chunk(string.Empty).ToList());
        }

        [Fact]
        public void Chunk_ShortText_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = "This is a short text.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal(text, chunks[0]);
        }

        [Fact]
        public void Chunk_ExactChunkSizeText_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 20, chunkOverlap: 5);
            var text = "12345678901234567890"; // Exactly 20 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal(text, chunks[0]);
        }

        [Fact]
        public void Chunk_LongText_ReturnsMultipleChunks()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 20, chunkOverlap: 5);
            var text = "The quick brown fox jumps over the lazy dog and runs away";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count > 1, "Expected multiple chunks for long text");
        }

        [Fact]
        public void Chunk_VerifyOverlap_ChunksOverlapCorrectly()
        {
            // Arrange
            var chunkSize = 20;
            var chunkOverlap = 5;
            var strategy = new FixedSizeChunkingStrategy(chunkSize: chunkSize, chunkOverlap: chunkOverlap);
            var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdefghij"; // 46 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Expected at least 2 chunks");

            // Verify the end of chunk 1 matches the beginning of chunk 2
            if (chunks.Count >= 2)
            {
                var endOfChunk1 = chunks[0].Substring(chunks[0].Length - chunkOverlap);
                var startOfChunk2 = chunks[1].Substring(0, chunkOverlap);
                Assert.Equal(endOfChunk1, startOfChunk2);
            }
        }

        [Fact]
        public void Chunk_NoOverlap_ChunksAreContiguous()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 0);
            var text = "0123456789ABCDEFGHIJ"; // 20 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Equal(2, chunks.Count);
            Assert.Equal("0123456789", chunks[0]);
            Assert.Equal("ABCDEFGHIJ", chunks[1]);
        }

        [Fact]
        public void Chunk_ReturnsChunksInOrder()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 2);
            var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count > 1, "Expected multiple chunks");
            // First chunk should start with 'A'
            Assert.StartsWith("A", chunks[0]);
        }

        #endregion

        #region ChunkWithPositions Tests

        [Fact]
        public void ChunkWithPositions_NullText_ThrowsArgumentNullException()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                strategy.ChunkWithPositions(null!).ToList());
        }

        [Fact]
        public void ChunkWithPositions_EmptyText_ThrowsArgumentException()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                strategy.ChunkWithPositions(string.Empty).ToList());
        }

        [Fact]
        public void ChunkWithPositions_ShortText_ReturnsCorrectPositions()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = "Short text";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal(text, chunks[0].Chunk);
            Assert.Equal(0, chunks[0].StartPosition);
            Assert.Equal(text.Length, chunks[0].EndPosition);
        }

        [Fact]
        public void ChunkWithPositions_TracksPositionsCorrectly()
        {
            // Arrange
            var chunkSize = 10;
            var chunkOverlap = 2;
            var strategy = new FixedSizeChunkingStrategy(chunkSize: chunkSize, chunkOverlap: chunkOverlap);
            var text = "0123456789ABCDEFGHIJ"; // 20 characters

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Expected at least 2 chunks");

            // First chunk: position 0-10
            Assert.Equal(0, chunks[0].StartPosition);
            Assert.Equal(10, chunks[0].EndPosition);
            Assert.Equal("0123456789", chunks[0].Chunk);

            // Second chunk: should start at chunkSize - chunkOverlap = 8
            Assert.Equal(8, chunks[1].StartPosition);
        }

        [Fact]
        public void ChunkWithPositions_ChunkContentMatchesOriginalAtPositions()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 15, chunkOverlap: 3);
            var text = "The quick brown fox jumps over the lazy dog";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            foreach (var (chunk, startPos, endPos) in chunks)
            {
                var expectedSubstring = text.Substring(startPos, endPos - startPos);
                Assert.Equal(expectedSubstring, chunk);
            }
        }

        [Fact]
        public void ChunkWithPositions_LastChunk_EndsAtTextLength()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 2);
            var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // 26 characters

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            var lastChunk = chunks.Last();
            Assert.Equal(text.Length, lastChunk.EndPosition);
        }

        [Fact]
        public void ChunkWithPositions_OverlappingChunks_PositionsOverlap()
        {
            // Arrange
            var chunkSize = 10;
            var chunkOverlap = 3;
            var strategy = new FixedSizeChunkingStrategy(chunkSize: chunkSize, chunkOverlap: chunkOverlap);
            var text = "0123456789ABCDEFGHIJKLMNO"; // 25 characters

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2, "Expected at least 2 chunks");

            // Check that consecutive chunks overlap
            for (int i = 0; i < chunks.Count - 1; i++)
            {
                var currentEnd = chunks[i].EndPosition;
                var nextStart = chunks[i + 1].StartPosition;
                Assert.True(nextStart < currentEnd,
                    $"Chunk {i} end ({currentEnd}) should be after chunk {i + 1} start ({nextStart})");
            }
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Chunk_SingleCharacter_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = "X";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("X", chunks[0]);
        }

        [Fact]
        public void Chunk_WhitespaceOnlyText_ReturnsChunk()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
            var text = "   ";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("   ", chunks[0]);
        }

        [Fact]
        public void Chunk_VeryLargeChunkSize_ReturnsEntireTextAsSingleChunk()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 10000, chunkOverlap: 100);
            var text = "This is a moderately sized text that won't fill the chunk size.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal(text, chunks[0]);
        }

        [Fact]
        public void Chunk_SmallChunkSize_CreatesManyChunks()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 5, chunkOverlap: 1);
            var text = "0123456789ABCDEF"; // 16 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 3, "Small chunk size should create many chunks");
        }

        [Fact]
        public void Chunk_TextJustOverChunkSize_ReturnsTwoChunks()
        {
            // Arrange
            var chunkSize = 10;
            var chunkOverlap = 2;
            var strategy = new FixedSizeChunkingStrategy(chunkSize: chunkSize, chunkOverlap: chunkOverlap);
            var text = "12345678901"; // 11 characters - just over chunk size

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Equal(2, chunks.Count);
        }

        [Fact]
        public void ChunkWithPositions_LargeText_MaintainsConsistentStepSize()
        {
            // Arrange
            var chunkSize = 100;
            var chunkOverlap = 20;
            var expectedStep = chunkSize - chunkOverlap; // 80
            var strategy = new FixedSizeChunkingStrategy(chunkSize: chunkSize, chunkOverlap: chunkOverlap);
            var text = new string('X', 500);

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            for (int i = 0; i < chunks.Count - 1; i++)
            {
                var currentStart = chunks[i].StartPosition;
                var nextStart = chunks[i + 1].StartPosition;
                Assert.Equal(expectedStep, nextStart - currentStart);
            }
        }

        [Fact]
        public void Chunk_RepeatedCalls_ProduceSameResults()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 20, chunkOverlap: 5);
            var text = "The quick brown fox jumps over the lazy dog";

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

        #region Coverage Tests

        [Fact]
        public void Chunk_CoverAllText_NoCharactersMissed()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 0);
            var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // 26 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            var reconstructed = string.Join("", chunks);
            Assert.Equal(text, reconstructed);
        }

        [Fact]
        public void ChunkWithPositions_StartPositionsAreAscending()
        {
            // Arrange
            var strategy = new FixedSizeChunkingStrategy(chunkSize: 15, chunkOverlap: 3);
            var text = "The quick brown fox jumps over the lazy dog runs fast";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            for (int i = 1; i < chunks.Count; i++)
            {
                Assert.True(chunks[i].StartPosition > chunks[i - 1].StartPosition,
                    $"Start position of chunk {i} should be greater than chunk {i - 1}");
            }
        }

        [Fact]
        public void ChunkWithPositions_AllChunksHaveValidLength()
        {
            // Arrange
            var chunkSize = 20;
            var strategy = new FixedSizeChunkingStrategy(chunkSize: chunkSize, chunkOverlap: 5);
            var text = "The quick brown fox jumps over the lazy dog";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            foreach (var (chunk, startPos, endPos) in chunks)
            {
                Assert.True(chunk.Length <= chunkSize,
                    $"Chunk length ({chunk.Length}) should not exceed chunk size ({chunkSize})");
                Assert.True(chunk.Length > 0, "Chunk should not be empty");
                Assert.Equal(chunk.Length, endPos - startPos);
            }
        }

        #endregion
    }
}
