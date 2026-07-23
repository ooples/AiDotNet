using AiDotNet.Interfaces;
using Xunit;

namespace AiDotNet.Tests.ComponentTests.Base;

/// <summary>
/// Base test class for IChunkingStrategy implementations.
/// Tests chunking invariants: non-empty text produces chunks, empty text produces no chunks,
/// chunk size is positive, all original content is preserved, and each chunk is non-empty.
/// </summary>
public abstract class ChunkerTestBase
{
    /// <summary>
    /// Creates the chunking strategy under test.
    /// </summary>
    protected abstract IChunkingStrategy CreateChunker();

    /// <summary>
    /// Sample text used for chunking tests. Override for domain-specific content.
    /// </summary>
    protected virtual string SampleText =>
        "Artificial intelligence is transforming how we process information. " +
        "Machine learning models can identify patterns in large datasets. " +
        "Natural language processing enables computers to understand human text. " +
        "Deep learning architectures have achieved remarkable results in image recognition. " +
        "Retrieval-augmented generation combines search with language models for better answers. " +
        "Vector databases store document embeddings for efficient similarity search. " +
        "Chunking strategies help divide documents into manageable pieces for processing. " +
        "The quality of retrieval depends on both the embedding model and the chunking approach.";

    // =====================================================
    // INVARIANT: Non-empty text should produce chunks
    // Chunking a non-trivial text must produce at least one chunk.
    // =====================================================

    [Fact]
    public void Chunk_WithText_ReturnsNonEmptyChunks()
    {
        var chunker = CreateChunker();

        var chunks = chunker.Chunk(SampleText);

        Assert.NotNull(chunks);
        var chunkList = chunks.ToList();
        Assert.True(chunkList.Count > 0,
            "Chunker should produce at least one chunk from non-empty text.");
    }

    // =====================================================
    // INVARIANT: Empty text should produce no chunks
    // Chunking an empty string must not crash and should
    // return an empty collection.
    // =====================================================

    [Fact]
    public void Chunk_WithEmptyText_ReturnsEmpty()
    {
        var chunker = CreateChunker();

        var chunks = chunker.Chunk(string.Empty);

        Assert.NotNull(chunks);
        var chunkList = chunks.ToList();
        Assert.Empty(chunkList);
    }

    // =====================================================
    // INVARIANT: ChunkSize must be positive
    // A non-positive chunk size is nonsensical.
    // =====================================================

    [Fact]
    public void ChunkSize_ShouldBePositive()
    {
        var chunker = CreateChunker();

        Assert.True(chunker.ChunkSize > 0,
            $"ChunkSize should be > 0 but was {chunker.ChunkSize}.");
    }

    // =====================================================
    // INVARIANT: All original content must be preserved
    // When chunks are concatenated (accounting for overlap),
    // every character from the original text should appear
    // in at least one chunk.
    // =====================================================

    [Fact]
    public void Chunk_AllTextPreserved()
    {
        var chunker = CreateChunker();

        var chunks = chunker.Chunk(SampleText).ToList();

        if (chunks.Count == 0)
        {
            return; // Empty text produces no chunks; nothing to verify
        }

        // Every substring of the original text that doesn't span a chunk boundary
        // should be findable in at least one chunk. We verify that each word from
        // the original text appears in at least one chunk.
        var words = SampleText.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var word in words)
        {
            bool found = chunks.Any(chunk => chunk.IndexOf(word, StringComparison.Ordinal) >= 0);
            Assert.True(found,
                $"Word '{word}' from the original text was not found in any chunk. " +
                "Chunker may be losing content at boundaries.");
        }
    }

    // =====================================================
    // INVARIANT: Each chunk must be non-empty
    // No chunk should be an empty or whitespace-only string.
    // =====================================================

    [Fact]
    public void Chunk_EachChunkNonEmpty()
    {
        var chunker = CreateChunker();

        var chunks = chunker.Chunk(SampleText).ToList();

        for (int i = 0; i < chunks.Count; i++)
        {
            Assert.False(string.IsNullOrWhiteSpace(chunks[i]),
                $"Chunk at index {i} is empty or whitespace-only. " +
                "Chunker should not produce empty chunks.");
        }
    }
}
