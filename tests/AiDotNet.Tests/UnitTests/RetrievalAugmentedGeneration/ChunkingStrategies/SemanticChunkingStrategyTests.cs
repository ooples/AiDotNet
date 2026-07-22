using System.Collections.Generic;
using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using Xunit;

namespace AiDotNet.Tests.UnitTests.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Tests for the embedding-based <see cref="SemanticChunkingStrategy"/>: it should place a chunk boundary
/// at the semantic topic shift when given an embedder, and fall back to size packing without one.
/// </summary>
public class SemanticChunkingStrategyTests
{
    // Two orthogonal "topics": cat sentences → [1,0], dog sentences → [0,1].
    private static IReadOnlyList<double[]> TopicEmbed(IReadOnlyList<string> sentences)
        => sentences.Select(s => s.ToLowerInvariant().Contains("cat")
            ? new[] { 1.0, 0.0 }
            : new[] { 0.0, 1.0 }).ToList();

    [Fact]
    public void SplitsAtSemanticBreakpoint()
    {
        var strat = new SemanticChunkingStrategy(
            maxChunkSize: 10000, chunkOverlap: 0, embedBatch: TopicEmbed, breakpointPercentile: 90);
        var text = "Cats purr softly. Cats sleep all day. Dogs bark loudly. Dogs run fast.";

        var chunks = strat.Chunk(text).ToList();

        Assert.Equal(2, chunks.Count);
        Assert.Contains("Cats", chunks[0]);
        Assert.DoesNotContain("Dogs", chunks[0]);
        Assert.Contains("Dogs", chunks[1]);
    }

    [Fact]
    public void FallsBackToSizePackingWithoutEmbedder()
    {
        var strat = new SemanticChunkingStrategy(maxChunkSize: 10000, chunkOverlap: 0);
        var text = "Cats purr softly. Cats sleep all day. Dogs bark loudly. Dogs run fast.";

        var chunks = strat.Chunk(text).ToList();

        // No embedder → one size-bounded chunk (no semantic split).
        Assert.Single(chunks);
    }

    [Fact]
    public void PositionsMapBackToOriginalText()
    {
        var strat = new SemanticChunkingStrategy(
            maxChunkSize: 10000, chunkOverlap: 0, embedBatch: TopicEmbed, breakpointPercentile: 90);
        var text = "Cats purr softly. Cats sleep all day. Dogs bark loudly. Dogs run fast.";

        foreach (var (chunk, start, end) in strat.ChunkWithPositions(text))
        {
            Assert.Equal(chunk, text.Substring(start, end - start));
        }
    }
}
