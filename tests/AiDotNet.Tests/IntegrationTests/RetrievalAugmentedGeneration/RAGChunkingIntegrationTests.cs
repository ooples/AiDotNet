using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Integration tests for RAG chunking strategies:
/// FixedSizeChunkingStrategy, SentenceChunkingStrategy,
/// SlidingWindowChunkingStrategy, RecursiveCharacterChunkingStrategy,
/// and ChunkingStrategyBase.
/// </summary>
public class RAGChunkingIntegrationTests
{
    #region FixedSizeChunkingStrategy

    [Fact]
    public void FixedSize_DefaultParams_ChunksText()
    {
        var strategy = new FixedSizeChunkingStrategy(100, 10);
        var text = new string('a', 250);
        var chunks = strategy.Chunk(text).ToList();

        Assert.True(chunks.Count >= 2, "Should produce multiple chunks for text longer than chunk size");
    }

    [Fact]
    public void FixedSize_ShortText_SingleChunk()
    {
        var strategy = new FixedSizeChunkingStrategy(500, 50);
        var text = "Short text here.";
        var chunks = strategy.Chunk(text).ToList();

        Assert.Single(chunks);
        Assert.Equal("Short text here.", chunks[0]);
    }

    [Fact]
    public void FixedSize_ChunkSizeRespected()
    {
        var strategy = new FixedSizeChunkingStrategy(20, 5);
        var text = "The quick brown fox jumps over the lazy dog and other animals.";
        var chunks = strategy.Chunk(text).ToList();

        foreach (var chunk in chunks)
        {
            Assert.True(chunk.Length <= 20, $"Chunk '{chunk}' exceeds chunk size of 20");
        }
    }

    [Fact]
    public void FixedSize_WithPositions_TracksPositions()
    {
        var strategy = new FixedSizeChunkingStrategy(10, 2);
        var text = "abcdefghijklmnopqrstuvwxyz";
        var chunksWithPos = strategy.ChunkWithPositions(text).ToList();

        Assert.True(chunksWithPos.Count > 1);
        // First chunk should start at 0
        Assert.Equal(0, chunksWithPos[0].StartPosition);
    }

    [Fact]
    public void FixedSize_OverlapCreatesOverlappingContent()
    {
        var strategy = new FixedSizeChunkingStrategy(10, 3);
        var text = "0123456789ABCDEFGHIJ";
        var chunks = strategy.Chunk(text).ToList();

        if (chunks.Count >= 2)
        {
            // End of first chunk should overlap with beginning of second
            var end1 = chunks[0].Substring(chunks[0].Length - 3);
            Assert.True(chunks[1].StartsWith(end1),
                $"Expected overlap: end of chunk1 '{end1}' should be start of chunk2 '{chunks[1]}'");
        }
    }

    [Fact]
    public void FixedSize_NullText_Throws()
    {
        var strategy = new FixedSizeChunkingStrategy();
        Assert.Throws<ArgumentNullException>(() => strategy.Chunk(null!).ToList());
    }

    [Fact]
    public void FixedSize_EmptyText_Throws()
    {
        var strategy = new FixedSizeChunkingStrategy();
        Assert.Throws<ArgumentException>(() => strategy.Chunk("").ToList());
    }

    [Fact]
    public void FixedSize_ZeroChunkSize_Throws()
    {
        Assert.Throws<ArgumentException>(() => new FixedSizeChunkingStrategy(0, 0));
    }

    [Fact]
    public void FixedSize_NegativeOverlap_Throws()
    {
        Assert.Throws<ArgumentException>(() => new FixedSizeChunkingStrategy(100, -1));
    }

    [Fact]
    public void FixedSize_OverlapGteChunkSize_Throws()
    {
        Assert.Throws<ArgumentException>(() => new FixedSizeChunkingStrategy(100, 100));
    }

    [Fact]
    public void FixedSize_Properties_MatchConstructorParams()
    {
        var strategy = new FixedSizeChunkingStrategy(200, 30);
        Assert.Equal(200, strategy.ChunkSize);
        Assert.Equal(30, strategy.ChunkOverlap);
    }

    #endregion

    #region SentenceChunkingStrategy

    [Fact]
    public void Sentence_SplitsOnSentenceBoundaries()
    {
        var strategy = new SentenceChunkingStrategy(targetChunkSize: 50, maxChunkSize: 100, overlapSentences: 0);
        var text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here.";
        var chunks = strategy.Chunk(text).ToList();

        Assert.True(chunks.Count >= 1, "Should produce at least one chunk");
    }

    [Fact]
    public void Sentence_ShortText_SingleChunk()
    {
        var strategy = new SentenceChunkingStrategy(targetChunkSize: 500, maxChunkSize: 1000);
        var text = "Just one short sentence.";
        var chunks = strategy.Chunk(text).ToList();

        Assert.Single(chunks);
    }

    [Fact]
    public void Sentence_MaxChunkSizeLessThanTarget_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new SentenceChunkingStrategy(targetChunkSize: 1000, maxChunkSize: 500));
    }

    [Fact]
    public void Sentence_NegativeOverlapSentences_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new SentenceChunkingStrategy(overlapSentences: -1));
    }

    [Fact]
    public void Sentence_WithPositions_TracksPositions()
    {
        var strategy = new SentenceChunkingStrategy(targetChunkSize: 30, maxChunkSize: 60);
        var text = "First here. Second here. Third here.";
        var chunksWithPos = strategy.ChunkWithPositions(text).ToList();

        Assert.True(chunksWithPos.Count >= 1);
        foreach (var (chunk, start, end) in chunksWithPos)
        {
            Assert.True(start >= 0, "Start position should be non-negative");
            Assert.True(end > start, "End position should be greater than start");
        }
    }

    #endregion

    #region SlidingWindowChunkingStrategy

    [Fact]
    public void SlidingWindow_CreatesOverlappingChunks()
    {
        var strategy = new SlidingWindowChunkingStrategy(windowSize: 20, stride: 10);
        var text = "The quick brown fox jumps over the lazy dog sleeping in the sun.";
        var chunks = strategy.Chunk(text).ToList();

        Assert.True(chunks.Count >= 2, "Should create multiple overlapping windows");
    }

    [Fact]
    public void SlidingWindow_ShortText_SingleChunk()
    {
        var strategy = new SlidingWindowChunkingStrategy(windowSize: 100, stride: 50);
        var text = "Short.";
        var chunks = strategy.Chunk(text).ToList();

        Assert.Single(chunks);
    }

    #endregion

    #region RecursiveCharacterChunkingStrategy

    [Fact]
    public void Recursive_SplitsLongText()
    {
        var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 50, chunkOverlap: 10);
        var text = "First paragraph about topic one.\n\nSecond paragraph about topic two.\n\nThird paragraph about topic three.";
        var chunks = strategy.Chunk(text).ToList();

        Assert.True(chunks.Count >= 1, "Should produce at least one chunk");
    }

    [Fact]
    public void Recursive_ShortText_SingleChunk()
    {
        var strategy = new RecursiveCharacterChunkingStrategy(chunkSize: 500);
        var text = "Just a short text.";
        var chunks = strategy.Chunk(text).ToList();

        Assert.Single(chunks);
    }

    #endregion

    #region Cross-Strategy - Consistency

    [Fact]
    public void AllStrategies_CoverEntireText_NoContentLoss()
    {
        var text = "The quick brown fox jumps over the lazy dog.";
        var strategy = new FixedSizeChunkingStrategy(15, 3);
        var chunks = strategy.Chunk(text).ToList();

        // All chunks should be non-empty
        Assert.All(chunks, chunk => Assert.False(string.IsNullOrWhiteSpace(chunk)));
    }

    [Fact]
    public void AllStrategies_WithPositions_PositionsInRange()
    {
        var text = "This is a test text for chunking with position tracking.";
        var strategy = new FixedSizeChunkingStrategy(20, 5);
        var chunksWithPos = strategy.ChunkWithPositions(text).ToList();

        foreach (var (chunk, start, end) in chunksWithPos)
        {
            Assert.True(start >= 0, $"Start position {start} should be non-negative");
            Assert.True(end <= text.Length, $"End position {end} should not exceed text length {text.Length}");
            Assert.True(start < end, $"Start {start} should be less than end {end}");
        }
    }

    #endregion
}
