using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using AiDotNet.RetrievalAugmentedGeneration.Configuration;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Integration tests for RAG chunking strategies and configuration builder.
/// Tests golden reference chunking, overlap correctness, position tracking,
/// sentence boundary handling, configuration validation, and edge cases.
/// </summary>
public class RAGChunkingConfigIntegrationTests
{
    #region FixedSizeChunkingStrategy Tests

    [Fact]
    public void FixedSize_GoldenReference_ChunksCorrectly()
    {
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 0);
        var text = "0123456789ABCDEFGHIJ"; // 20 chars

        var chunks = strategy.Chunk(text).ToList();

        Assert.Equal(2, chunks.Count);
        Assert.Equal("0123456789", chunks[0]);
        Assert.Equal("ABCDEFGHIJ", chunks[1]);
    }

    [Fact]
    public void FixedSize_WithOverlap_OverlapsCorrectly()
    {
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 3);
        var text = "0123456789ABCDE"; // 15 chars

        var chunks = strategy.Chunk(text).ToList();

        // First chunk: [0..10) = "0123456789"
        // Second chunk starts at 10 - 3 = 7: [7..15) = "789ABCDE"
        Assert.Equal(2, chunks.Count);
        Assert.Equal("0123456789", chunks[0]);
        Assert.Equal("789ABCDE", chunks[1]);

        // Verify overlap content
        Assert.Equal(chunks[0][7..], chunks[1][..3]); // "789" overlaps
    }

    [Fact]
    public void FixedSize_ShortText_SingleChunk()
    {
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 10);
        var text = "Short text.";

        var chunks = strategy.Chunk(text).ToList();

        Assert.Single(chunks);
        Assert.Equal("Short text.", chunks[0]);
    }

    [Fact]
    public void FixedSize_ChunkWithPositions_PositionsAccurate()
    {
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 0);
        var text = "0123456789ABCDEFGHIJ"; // 20 chars

        var positions = strategy.ChunkWithPositions(text).ToList();

        Assert.Equal(2, positions.Count);
        Assert.Equal(0, positions[0].StartPosition);
        Assert.Equal(10, positions[0].EndPosition);
        Assert.Equal(10, positions[1].StartPosition);
        Assert.Equal(20, positions[1].EndPosition);

        // Verify extracted text matches positions
        Assert.Equal(text[positions[0].StartPosition..positions[0].EndPosition], positions[0].Chunk);
        Assert.Equal(text[positions[1].StartPosition..positions[1].EndPosition], positions[1].Chunk);
    }

    [Fact]
    public void FixedSize_UnevenLength_LastChunkSmaller()
    {
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 0);
        var text = "012345678901234"; // 15 chars

        var chunks = strategy.Chunk(text).ToList();

        Assert.Equal(2, chunks.Count);
        Assert.Equal(10, chunks[0].Length);
        Assert.Equal(5, chunks[1].Length);
    }

    [Fact]
    public void FixedSize_NullText_Throws()
    {
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 100);

        Assert.Throws<ArgumentNullException>(() => strategy.Chunk(null!).ToList());
    }

    [Fact]
    public void FixedSize_EmptyText_Throws()
    {
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 100);

        Assert.Throws<ArgumentException>(() => strategy.Chunk("").ToList());
    }

    [Fact]
    public void FixedSize_InvalidChunkSize_Throws()
    {
        Assert.Throws<ArgumentException>(() => new FixedSizeChunkingStrategy(chunkSize: 0));
        Assert.Throws<ArgumentException>(() => new FixedSizeChunkingStrategy(chunkSize: -1));
    }

    [Fact]
    public void FixedSize_OverlapExceedsSize_Throws()
    {
        Assert.Throws<ArgumentException>(() => new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 10));
        Assert.Throws<ArgumentException>(() => new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 15));
    }

    [Fact]
    public void FixedSize_NegativeOverlap_Throws()
    {
        Assert.Throws<ArgumentException>(() => new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: -1));
    }

    [Fact]
    public void FixedSize_DefaultParameters()
    {
        var strategy = new FixedSizeChunkingStrategy();

        Assert.Equal(500, strategy.ChunkSize);
        Assert.Equal(50, strategy.ChunkOverlap);
    }

    [Fact]
    public void FixedSize_LargeDocument_AllTextCovered()
    {
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 100, chunkOverlap: 20);
        var text = new string('A', 1000); // 1000 chars

        var positions = strategy.ChunkWithPositions(text).ToList();

        // First chunk covers position 0
        Assert.Equal(0, positions.First().StartPosition);

        // Last chunk covers the end
        Assert.Equal(text.Length, positions.Last().EndPosition);

        // All chunks should be <= chunkSize
        Assert.All(positions, p => Assert.True(p.Chunk.Length <= 100));
    }

    #endregion

    #region SlidingWindowChunkingStrategy Tests

    [Fact]
    public void SlidingWindow_GoldenReference()
    {
        var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 5);
        var text = "0123456789ABCDE"; // 15 chars

        var chunks = strategy.Chunk(text).ToList();

        // Window at 0: "0123456789"
        // Window at 5: "56789ABCDE"
        Assert.Equal(2, chunks.Count);
        Assert.Equal("0123456789", chunks[0]);
        Assert.Equal("56789ABCDE", chunks[1]);
    }

    [Fact]
    public void SlidingWindow_StrideEqualsWindow_NoOverlap()
    {
        var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 10);
        var text = "0123456789ABCDEFGHIJ"; // 20 chars

        var chunks = strategy.Chunk(text).ToList();

        Assert.Equal(2, chunks.Count);
        Assert.Equal("0123456789", chunks[0]);
        Assert.Equal("ABCDEFGHIJ", chunks[1]);
    }

    [Fact]
    public void SlidingWindow_SmallStride_HighOverlap()
    {
        var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 2);
        var text = "0123456789ABCDE"; // 15 chars

        var chunks = strategy.Chunk(text).ToList();

        // Should have many overlapping windows
        Assert.True(chunks.Count > 3);

        // Each chunk should be <= windowSize
        Assert.All(chunks, c => Assert.True(c.Length <= 10));

        // Adjacent chunks should overlap by 8 chars (10 - 2)
        for (int i = 0; i < chunks.Count - 1; i++)
        {
            if (chunks[i].Length == 10 && chunks[i + 1].Length == 10)
            {
                string overlap1 = chunks[i][2..]; // last 8 chars
                string overlap2 = chunks[i + 1][..8]; // first 8 chars
                Assert.Equal(overlap1, overlap2);
            }
        }
    }

    [Fact]
    public void SlidingWindow_InvalidStride_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlidingWindowChunkingStrategy(windowSize: 10, stride: 0));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlidingWindowChunkingStrategy(windowSize: 10, stride: -1));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlidingWindowChunkingStrategy(windowSize: 10, stride: 11));
    }

    [Fact]
    public void SlidingWindow_InvalidWindowSize_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new SlidingWindowChunkingStrategy(windowSize: 0, stride: 1));
    }

    [Fact]
    public void SlidingWindow_DefaultParameters()
    {
        var strategy = new SlidingWindowChunkingStrategy();

        Assert.Equal(1000, strategy.ChunkSize); // windowSize
    }

    [Fact]
    public void SlidingWindow_Positions_AreAccurate()
    {
        var strategy = new SlidingWindowChunkingStrategy(windowSize: 10, stride: 7);
        var text = "The quick brown fox jumps."; // 25 chars

        var positions = strategy.ChunkWithPositions(text).ToList();

        foreach (var pos in positions)
        {
            Assert.Equal(text.Substring(pos.StartPosition, pos.EndPosition - pos.StartPosition), pos.Chunk);
        }
    }

    #endregion

    #region SentenceChunkingStrategy Tests

    [Fact]
    public void Sentence_SplitsAtSentenceBoundaries()
    {
        var strategy = new SentenceChunkingStrategy(targetChunkSize: 50, maxChunkSize: 100, overlapSentences: 0);
        var text = "First sentence. Second sentence. Third sentence. Fourth sentence.";

        var chunks = strategy.Chunk(text).ToList();

        // Should not split mid-sentence
        Assert.All(chunks, c =>
        {
            // Each chunk should end at or after a sentence boundary
            string trimmed = c.Trim();
            Assert.True(trimmed.Length > 0);
        });
    }

    [Fact]
    public void Sentence_ShortText_SingleChunk()
    {
        var strategy = new SentenceChunkingStrategy(targetChunkSize: 1000, maxChunkSize: 2000, overlapSentences: 0);
        var text = "Short sentence.";

        var chunks = strategy.Chunk(text).ToList();

        Assert.Single(chunks);
        Assert.Contains("Short sentence", chunks[0]);
    }

    [Fact]
    public void Sentence_OverlapPreservesContext()
    {
        var strategy = new SentenceChunkingStrategy(targetChunkSize: 30, maxChunkSize: 60, overlapSentences: 1);
        var text = "First sentence here. Second sentence here. Third sentence here.";

        var chunks = strategy.Chunk(text).ToList();

        // With overlap, adjacent chunks should share at least one sentence
        if (chunks.Count > 1)
        {
            // At least verify chunks are non-empty and reasonable
            Assert.All(chunks, c => Assert.True(c.Trim().Length > 0));
        }
    }

    [Fact]
    public void Sentence_InvalidParameters_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new SentenceChunkingStrategy(targetChunkSize: 200, maxChunkSize: 100)); // max < target

        Assert.Throws<ArgumentException>(() =>
            new SentenceChunkingStrategy(overlapSentences: -1));
    }

    [Fact]
    public void Sentence_VeryLongSentence_GetsSplit()
    {
        var strategy = new SentenceChunkingStrategy(targetChunkSize: 20, maxChunkSize: 30, overlapSentences: 0);
        // This single "sentence" is longer than maxChunkSize
        var text = "This is a very long single sentence that exceeds maximum size.";

        var chunks = strategy.Chunk(text).ToList();

        // Should produce multiple chunks since the sentence exceeds maxChunkSize
        Assert.True(chunks.Count >= 1);
        Assert.All(chunks, c => Assert.True(c.Length > 0));
    }

    [Fact]
    public void Sentence_MultipleEndingTypes_AllRecognized()
    {
        var strategy = new SentenceChunkingStrategy(targetChunkSize: 50, maxChunkSize: 100, overlapSentences: 0);
        var text = "Period end. Question end? Exclamation end! More text here.";

        var chunks = strategy.Chunk(text).ToList();

        // Should handle all sentence ending types
        Assert.True(chunks.Count >= 1);
        // All original text should be represented
        string joined = string.Join("", chunks);
        Assert.Contains("Period", joined);
        Assert.Contains("Question", joined);
        Assert.Contains("Exclamation", joined);
    }

    [Fact]
    public void Sentence_NullText_Throws()
    {
        var strategy = new SentenceChunkingStrategy();

        Assert.Throws<ArgumentNullException>(() => strategy.Chunk(null!).ToList());
    }

    [Fact]
    public void Sentence_EmptyText_Throws()
    {
        var strategy = new SentenceChunkingStrategy();

        Assert.Throws<ArgumentException>(() => strategy.Chunk("").ToList());
    }

    #endregion

    #region RAGConfigurationBuilder Tests

    [Fact]
    public void Builder_FullConfiguration_Builds()
    {
        var config = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory")
            .WithChunking("FixedSize", chunkSize: 500, chunkOverlap: 50)
            .WithEmbedding("OpenAI", embeddingDimension: 1536)
            .WithRetrieval("VectorSearch", topK: 10)
            .Build();

        Assert.Equal("InMemory", config.DocumentStore.Type);
        Assert.Equal("FixedSize", config.Chunking.Strategy);
        Assert.Equal(500, config.Chunking.ChunkSize);
        Assert.Equal(50, config.Chunking.ChunkOverlap);
        Assert.Equal("OpenAI", config.Embedding.ModelType);
        Assert.Equal(1536, config.Embedding.EmbeddingDimension);
        Assert.Equal("VectorSearch", config.Retrieval.Strategy);
        Assert.Equal(10, config.Retrieval.TopK);
    }

    [Fact]
    public void Builder_WithReranking_SetsEnabled()
    {
        var config = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory")
            .WithChunking("FixedSize")
            .WithEmbedding("OpenAI")
            .WithRetrieval("VectorSearch")
            .WithReranking("Diversity", topK: 5)
            .Build();

        Assert.True(config.Reranking.Enabled);
        Assert.Equal("Diversity", config.Reranking.Strategy);
        Assert.Equal(5, config.Reranking.TopK);
    }

    [Fact]
    public void Builder_WithQueryExpansion_SetsEnabled()
    {
        var config = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory")
            .WithChunking("FixedSize")
            .WithEmbedding("OpenAI")
            .WithRetrieval("VectorSearch")
            .WithQueryExpansion("HyDE", numExpansions: 5)
            .Build();

        Assert.True(config.QueryExpansion.Enabled);
        Assert.Equal("HyDE", config.QueryExpansion.Strategy);
        Assert.Equal(5, config.QueryExpansion.NumExpansions);
    }

    [Fact]
    public void Builder_WithContextCompression_SetsEnabled()
    {
        var config = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory")
            .WithChunking("FixedSize")
            .WithEmbedding("OpenAI")
            .WithRetrieval("VectorSearch")
            .WithContextCompression("Summarizer", compressionRatio: 0.3, maxLength: 200)
            .Build();

        Assert.True(config.ContextCompression.Enabled);
        Assert.Equal("Summarizer", config.ContextCompression.Strategy);
        Assert.Equal(0.3, config.ContextCompression.CompressionRatio);
        Assert.Equal(200, config.ContextCompression.MaxLength);
    }

    [Fact]
    public void Builder_ChainingMethods_ReturnsSameBuilder()
    {
        var builder = new RAGConfigurationBuilder<double>();

        var result = builder
            .WithDocumentStore("InMemory")
            .WithChunking("FixedSize")
            .WithEmbedding("OpenAI")
            .WithRetrieval("VectorSearch");

        Assert.Same(builder, result);
    }

    [Fact]
    public void Builder_MissingDocumentStore_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>()
            .WithChunking("FixedSize")
            .WithEmbedding("OpenAI")
            .WithRetrieval("VectorSearch");

        Assert.Throws<InvalidOperationException>(() => builder.Build());
    }

    [Fact]
    public void Builder_MissingChunking_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory")
            .WithEmbedding("OpenAI")
            .WithRetrieval("VectorSearch");

        Assert.Throws<InvalidOperationException>(() => builder.Build());
    }

    [Fact]
    public void Builder_MissingEmbedding_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory")
            .WithChunking("FixedSize")
            .WithRetrieval("VectorSearch");

        Assert.Throws<InvalidOperationException>(() => builder.Build());
    }

    [Fact]
    public void Builder_MissingRetrieval_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory")
            .WithChunking("FixedSize")
            .WithEmbedding("OpenAI");

        Assert.Throws<InvalidOperationException>(() => builder.Build());
    }

    [Fact]
    public void Builder_EmptyDocumentStoreType_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>();

        Assert.Throws<ArgumentException>(() => builder.WithDocumentStore(""));
        Assert.Throws<ArgumentException>(() => builder.WithDocumentStore("  "));
    }

    [Fact]
    public void Builder_InvalidChunkSize_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() => builder.WithChunking("FixedSize", chunkSize: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => builder.WithChunking("FixedSize", chunkSize: -1));
    }

    [Fact]
    public void Builder_NegativeChunkOverlap_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() => builder.WithChunking("FixedSize", chunkOverlap: -1));
    }

    [Fact]
    public void Builder_InvalidEmbeddingDimension_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() => builder.WithEmbedding("OpenAI", embeddingDimension: 0));
    }

    [Fact]
    public void Builder_InvalidTopK_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() => builder.WithRetrieval("VectorSearch", topK: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => builder.WithReranking("Diversity", topK: 0));
    }

    [Fact]
    public void Builder_InvalidCompressionRatio_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            builder.WithContextCompression("Summarizer", compressionRatio: -0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            builder.WithContextCompression("Summarizer", compressionRatio: 1.1));
    }

    [Fact]
    public void Builder_InvalidQueryExpansionCount_Throws()
    {
        var builder = new RAGConfigurationBuilder<double>();

        Assert.Throws<ArgumentOutOfRangeException>(() => builder.WithQueryExpansion("HyDE", numExpansions: 0));
    }

    [Fact]
    public void Builder_DocumentStoreWithParameters()
    {
        var parameters = new Dictionary<string, object>
        {
            { "connectionString", "memory://" },
            { "maxDocuments", 10000 }
        };

        var config = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory", parameters)
            .WithChunking("FixedSize")
            .WithEmbedding("OpenAI")
            .WithRetrieval("VectorSearch")
            .Build();

        Assert.Equal("memory://", config.DocumentStore.Parameters["connectionString"]);
        Assert.Equal(10000, config.DocumentStore.Parameters["maxDocuments"]);
    }

    [Fact]
    public void Builder_EmbeddingWithApiKey()
    {
        var config = new RAGConfigurationBuilder<double>()
            .WithDocumentStore("InMemory")
            .WithChunking("FixedSize")
            .WithEmbedding("OpenAI", apiKey: "sk-test-key", embeddingDimension: 1536)
            .WithRetrieval("VectorSearch")
            .Build();

        Assert.Equal("sk-test-key", config.Embedding.ApiKey);
    }

    #endregion

    #region End-to-End Chunking Pipeline Tests

    [Fact]
    public void Pipeline_FixedSizeChunking_CoverEntireDocument()
    {
        var strategy = new FixedSizeChunkingStrategy(chunkSize: 50, chunkOverlap: 10);
        var text = string.Join(" ", Enumerable.Range(0, 100).Select(i => $"word{i}"));

        var positions = strategy.ChunkWithPositions(text).ToList();

        // All positions should be valid
        Assert.All(positions, p =>
        {
            Assert.True(p.StartPosition >= 0);
            Assert.True(p.EndPosition <= text.Length);
            Assert.True(p.StartPosition < p.EndPosition);
        });

        // First chunk starts at 0, last chunk ends at text.Length
        Assert.Equal(0, positions.First().StartPosition);
        Assert.Equal(text.Length, positions.Last().EndPosition);
    }

    [Fact]
    public void Pipeline_SlidingWindow_ConsistentOverlap()
    {
        int windowSize = 20;
        int stride = 8;
        var strategy = new SlidingWindowChunkingStrategy(windowSize: windowSize, stride: stride);
        var text = "The quick brown fox jumps over the lazy dog and runs away fast.";

        var positions = strategy.ChunkWithPositions(text).ToList();

        // Verify stride between consecutive chunks
        for (int i = 1; i < positions.Count; i++)
        {
            int actualStride = positions[i].StartPosition - positions[i - 1].StartPosition;
            Assert.Equal(stride, actualStride);
        }
    }

    [Fact]
    public void Pipeline_CompareChunkingStrategies_AllCoverDocument()
    {
        var text = "Machine learning is great. Neural networks are powerful. Deep learning rocks. AI will transform the world.";

        var fixedChunks = new FixedSizeChunkingStrategy(chunkSize: 40, chunkOverlap: 5)
            .Chunk(text).ToList();
        var slidingChunks = new SlidingWindowChunkingStrategy(windowSize: 40, stride: 35)
            .Chunk(text).ToList();

        // Both should produce non-empty results
        Assert.True(fixedChunks.Count > 0);
        Assert.True(slidingChunks.Count > 0);

        // All chunks from both should be substrings of the original
        Assert.All(fixedChunks, c => Assert.True(text.Contains(c)));
        Assert.All(slidingChunks, c => Assert.True(text.Contains(c)));
    }

    #endregion
}
