using System.Linq;
using AiDotNet.RetrievalAugmentedGeneration.ChunkingStrategies;
using Xunit;

namespace AiDotNet.Tests.UnitTests.RetrievalAugmentedGeneration.ChunkingStrategies;

/// <summary>
/// Tests for <see cref="TokenBasedChunkingStrategy"/> — token-budgeted chunking with overlap and an
/// injectable token counter.
/// </summary>
public class TokenBasedChunkingStrategyTests
{
    private static int WordCount(string s) => s.Split((char[]?)null, System.StringSplitOptions.RemoveEmptyEntries).Length;

    [Fact]
    public void RespectsMaxTokenBudget_WithDefaultWordCounter()
    {
        var strat = new TokenBasedChunkingStrategy(maxTokens: 5, overlapTokens: 2);
        var text = "one two three four five six seven eight nine ten eleven twelve";

        var chunks = strat.Chunk(text).ToList();

        Assert.True(chunks.Count > 1);
        Assert.All(chunks, c => Assert.True(WordCount(c) <= 5, $"chunk exceeded budget: '{c}'"));
    }

    [Fact]
    public void ConsecutiveChunksOverlap()
    {
        var strat = new TokenBasedChunkingStrategy(maxTokens: 5, overlapTokens: 2);
        var chunks = strat.Chunk("one two three four five six seven eight").ToList();

        // Chunk1 = one..five, Chunk2 = four..eight → overlap "four five".
        Assert.Contains("four", chunks[0]);
        Assert.Contains("five", chunks[0]);
        Assert.Contains("four", chunks[1]);
        Assert.Contains("five", chunks[1]);
    }

    [Fact]
    public void PositionsMapBackToOriginalText()
    {
        var strat = new TokenBasedChunkingStrategy(maxTokens: 4, overlapTokens: 1);
        var text = "alpha beta gamma delta epsilon zeta";

        foreach (var (chunk, start, end) in strat.ChunkWithPositions(text))
        {
            Assert.Equal(chunk, text.Substring(start, end - start));
        }
    }

    [Fact]
    public void HonorsCustomTokenCounter()
    {
        // Counter that treats every word as 3 tokens → budget of 6 tokens allows ~2 words per chunk.
        var strat = new TokenBasedChunkingStrategy(maxTokens: 6, overlapTokens: 0, tokenCounter: s => WordCount(s) * 3);
        var chunks = strat.Chunk("a b c d e f").ToList();

        Assert.All(chunks, c => Assert.True(WordCount(c) <= 2, $"expected <=2 words/chunk, got '{c}'"));
        Assert.True(chunks.Count >= 3);
    }
}
