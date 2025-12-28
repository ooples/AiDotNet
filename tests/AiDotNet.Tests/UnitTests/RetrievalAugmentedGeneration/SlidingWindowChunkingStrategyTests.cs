using System;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using Xunit;

namespace AiDotNetTests.UnitTests.RetrievalAugmentedGeneration
{
    /// <summary>
    /// Tests for SlidingWindowChunkingStrategy which uses a configurable window size and stride.
    /// </summary>
    public class SlidingWindowChunkingStrategyTests
    {
        #region Constructor Tests

        [Fact]
        public void Constructor_WithDefaultValues_SetsCorrectProperties()
        {
            // Act
            var strategy = new SlidingWindowChunkingStrategy();

            // Assert
            Assert.Equal(1000, strategy.ChunkSize);
            Assert.Equal(500, strategy.ChunkOverlap); // windowSize - stride = 1000 - 500
        }

        [Fact]
        public void Constructor_WithCustomValues_SetsCorrectProperties()
        {
            // Act
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 200, stride: 50);

            // Assert
            Assert.Equal(200, strategy.ChunkSize);
            Assert.Equal(150, strategy.ChunkOverlap); // 200 - 50
        }

        [Fact]
        public void Constructor_StrideZero_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SlidingWindowChunkingStrategy(windowSize: 100, stride: 0));

            Assert.Contains("Stride must be greater than zero", ex.Message);
        }

        [Fact]
        public void Constructor_StrideNegative_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SlidingWindowChunkingStrategy(windowSize: 100, stride: -1));

            Assert.Contains("Stride must be greater than zero", ex.Message);
        }

        [Fact]
        public void Constructor_StrideExceedsWindowSize_ThrowsArgumentOutOfRangeException()
        {
            // Act & Assert
            var ex = Assert.Throws<ArgumentOutOfRangeException>(() =>
                new SlidingWindowChunkingStrategy(windowSize: 100, stride: 150));

            Assert.Contains("Stride cannot exceed the window size", ex.Message);
        }

        [Fact]
        public void Constructor_StrideEqualsWindowSize_IsValid()
        {
            // Act - stride == windowSize means overlap = 0
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 100, stride: 100);

            // Assert
            Assert.Equal(100, strategy.ChunkSize);
            Assert.Equal(0, strategy.ChunkOverlap);
        }

        [Fact]
        public void Constructor_WindowSizeZero_ThrowsArgumentException()
        {
            // Act & Assert - base class validation
            Assert.Throws<ArgumentException>(() =>
                new SlidingWindowChunkingStrategy(windowSize: 0, stride: 0));
        }

        [Fact]
        public void Constructor_WindowSizeNegative_ThrowsArgumentException()
        {
            // Act & Assert - base class validation
            Assert.Throws<ArgumentException>(() =>
                new SlidingWindowChunkingStrategy(windowSize: -1, stride: 1));
        }

        [Fact]
        public void Constructor_SmallStrideCreatesLargeOverlap()
        {
            // Act
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 100, stride: 10);

            // Assert
            Assert.Equal(100, strategy.ChunkSize);
            Assert.Equal(90, strategy.ChunkOverlap); // 100 - 10 = 90
        }

        #endregion

        #region Chunk Method Tests

        [Fact]
        public void Chunk_NullText_ThrowsArgumentNullException()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                strategy.Chunk(null!).ToList());
        }

        [Fact]
        public void Chunk_EmptyText_ThrowsArgumentException()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy();

            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                strategy.Chunk(string.Empty).ToList());
        }

        [Fact]
        public void Chunk_ShortText_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 100, stride: 50);
            var text = "Short text.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("Short text.", chunks[0]);
        }

        [Fact]
        public void Chunk_TextExactlyWindowSize_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 5);
            var text = "1234567890"; // Exactly 10 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("1234567890", chunks[0]);
        }

        [Fact]
        public void Chunk_LongText_ReturnsMultipleChunks()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 5);
            var text = "0123456789ABCDEFGHIJ"; // 20 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count > 1, "Should create multiple overlapping chunks");
        }

        [Fact]
        public void Chunk_WithStride_ChunksOverlapCorrectly()
        {
            // Arrange
            var windowSize = 10;
            var stride = 5;
            var strategy = new SlidingWindowChunkingStrategy(windowSize: windowSize, stride: stride);
            var text = "0123456789ABCDE"; // 15 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count >= 2);
            // With stride=5, chunk 1 ends at 10, chunk 2 starts at 5
            // So overlap should be positions 5-9 (5 characters)
            if (chunks.Count >= 2)
            {
                var endOfChunk1 = chunks[0].Substring(chunks[0].Length - stride);
                var startOfChunk2 = chunks[1].Substring(0, stride);
                Assert.Equal(endOfChunk1, startOfChunk2);
            }
        }

        [Fact]
        public void Chunk_StrideEqualsWindowSize_NoOverlap()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 10);
            var text = "0123456789ABCDEFGHIJ"; // 20 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Equal(2, chunks.Count);
            Assert.Equal("0123456789", chunks[0]);
            Assert.Equal("ABCDEFGHIJ", chunks[1]);
        }

        #endregion

        #region ChunkWithPositions Tests

        [Fact]
        public void ChunkWithPositions_SingleChunk_ReturnsValidPositions()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 100, stride: 50);
            var text = "Short text.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal(0, chunks[0].StartPosition);
            Assert.Equal(text.Length, chunks[0].EndPosition);
        }

        [Fact]
        public void ChunkWithPositions_MultipleChunks_HasCorrectPositions()
        {
            // Arrange
            var windowSize = 10;
            var stride = 5;
            var strategy = new SlidingWindowChunkingStrategy(windowSize: windowSize, stride: stride);
            var text = "0123456789ABCDEFGHIJ"; // 20 characters

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            Assert.Equal(3, chunks.Count);

            // First chunk: 0-10
            Assert.Equal(0, chunks[0].StartPosition);
            Assert.Equal(10, chunks[0].EndPosition);

            // Second chunk: 5-15
            Assert.Equal(5, chunks[1].StartPosition);
            Assert.Equal(15, chunks[1].EndPosition);

            // Third chunk: 10-20
            Assert.Equal(10, chunks[2].StartPosition);
            Assert.Equal(20, chunks[2].EndPosition);
        }

        [Fact]
        public void ChunkWithPositions_ChunkContentMatchesSubstring()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 5);
            var text = "The quick brown fox jumps over the lazy dog.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            foreach (var (chunk, startPos, endPos) in chunks)
            {
                var expected = text.Substring(startPos, endPos - startPos);
                Assert.Equal(expected, chunk);
            }
        }

        [Fact]
        public void ChunkWithPositions_LastChunk_EndsAtTextLength()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 5);
            var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            var lastChunk = chunks.Last();
            Assert.Equal(text.Length, lastChunk.EndPosition);
        }

        [Fact]
        public void ChunkWithPositions_ConsecutiveChunks_StartPositionsDifferByStride()
        {
            // Arrange
            var windowSize = 10;
            var stride = 5;
            var strategy = new SlidingWindowChunkingStrategy(windowSize: windowSize, stride: stride);
            var text = new string('X', 100);

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            for (int i = 1; i < chunks.Count; i++)
            {
                var diff = chunks[i].StartPosition - chunks[i - 1].StartPosition;
                Assert.Equal(stride, diff);
            }
        }

        #endregion

        #region Edge Cases

        [Fact]
        public void Chunk_SingleCharacter_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 100, stride: 50);
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
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 100, stride: 50);
            var text = "   ";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
            Assert.Equal("   ", chunks[0]); // Whitespace is preserved
        }

        [Fact]
        public void Chunk_TextSlightlyOverWindowSize_ReturnsTwoChunks()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 5);
            var text = "12345678901"; // 11 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Equal(2, chunks.Count);
        }

        [Fact]
        public void Chunk_VeryLargeWindowSize_ReturnsSingleChunk()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 10000, stride: 5000);
            var text = "Short text that doesn't need sliding.";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.Single(chunks);
        }

        [Fact]
        public void Chunk_SmallStride_CreatesManyOverlappingChunks()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 2);
            var text = "0123456789ABCDEFGHIJ"; // 20 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Count > 5, "Small stride should create many overlapping chunks");
        }

        [Fact]
        public void Chunk_RepeatedCalls_ProduceSameResults()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 15, stride: 5);
            var text = "The quick brown fox jumps over the lazy dog.";

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

        #region Overlap Verification Tests

        [Fact]
        public void Chunk_OverlapAmount_EqualsWindowMinusStride()
        {
            // Arrange
            var windowSize = 20;
            var stride = 8;
            var expectedOverlap = windowSize - stride; // 12
            var strategy = new SlidingWindowChunkingStrategy(windowSize: windowSize, stride: stride);
            var text = new string('X', 100);

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            for (int i = 1; i < chunks.Count - 1; i++) // Exclude last chunk which may be shorter
            {
                var overlap = chunks[i - 1].EndPosition - chunks[i].StartPosition;
                Assert.Equal(expectedOverlap, overlap);
            }
        }

        [Fact]
        public void Chunk_AllChunksExceptLast_HaveWindowSize()
        {
            // Arrange
            var windowSize = 10;
            var stride = 5;
            var strategy = new SlidingWindowChunkingStrategy(windowSize: windowSize, stride: stride);
            var text = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // 36 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            for (int i = 0; i < chunks.Count - 1; i++)
            {
                Assert.Equal(windowSize, chunks[i].Length);
            }
        }

        [Fact]
        public void Chunk_LastChunk_MayBeShorterThanWindowSize()
        {
            // Arrange
            var windowSize = 10;
            var stride = 7;
            var strategy = new SlidingWindowChunkingStrategy(windowSize: windowSize, stride: stride);
            var text = "0123456789ABCD"; // 14 characters

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert
            Assert.True(chunks.Last().Length <= windowSize);
        }

        #endregion

        #region Coverage Tests

        [Fact]
        public void Chunk_EnsuresFullTextCoverage()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 10);
            var text = "0123456789ABCDEFGHIJ";

            // Act
            var chunks = strategy.Chunk(text).ToList();

            // Assert - with stride == windowSize, concatenation should equal original
            var reconstructed = string.Join("", chunks);
            Assert.Equal(text, reconstructed);
        }

        [Fact]
        public void ChunkWithPositions_StartPositionsAreAscending()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 15, stride: 5);
            var text = "The quick brown fox jumps over the lazy dog runs fast.";

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
        public void ChunkWithPositions_EndPositionMinusStartEquals_ChunkLength()
        {
            // Arrange
            var strategy = new SlidingWindowChunkingStrategy(windowSize: 12, stride: 4);
            var text = "The quick brown fox jumps over.";

            // Act
            var chunks = strategy.ChunkWithPositions(text).ToList();

            // Assert
            foreach (var (chunk, startPos, endPos) in chunks)
            {
                Assert.Equal(chunk.Length, endPos - startPos);
            }
        }

        #endregion
    }
}
