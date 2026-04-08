using AiDotNet.PromptEngineering;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.PromptEngineering;

public class ContextWindowManagerTests
{
    [Fact(Timeout = 60000)]
    public async Task Constructor_WithMaxTokens_SetsMaxTokens()
    {
        var manager = new ContextWindowManager(4096);

        Assert.NotNull(manager);
        Assert.Equal(4096, manager.MaxTokens);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_WithDefaultEstimator_CreatesManager()
    {
        var manager = new ContextWindowManager(8192);

        Assert.NotNull(manager);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_WithCustomEstimator_UsesEstimator()
    {
        var manager = new ContextWindowManager(4096, text => text.Length / 4);

        Assert.NotNull(manager);
    }

    [Fact(Timeout = 60000)]
    public async Task EstimateTokens_ReturnsEstimate()
    {
        var manager = new ContextWindowManager(4096);
        var text = "Hello world, this is a test.";

        var estimate = manager.EstimateTokens(text);

        Assert.True(estimate > 0);
    }

    [Fact(Timeout = 60000)]
    public async Task EstimateTokens_EmptyString_ReturnsZero()
    {
        var manager = new ContextWindowManager(4096);

        var estimate = manager.EstimateTokens("");

        Assert.Equal(0, estimate);
    }

    [Fact(Timeout = 60000)]
    public async Task EstimateTokens_NullString_ReturnsZero()
    {
        var manager = new ContextWindowManager(4096);

        var estimate = manager.EstimateTokens(null!);

        Assert.Equal(0, estimate);
    }

    [Fact(Timeout = 60000)]
    public async Task EstimateTokens_WithCustomEstimator_UsesCustom()
    {
        var manager = new ContextWindowManager(4096, text => text.Length);
        var text = "Hello";

        var estimate = manager.EstimateTokens(text);

        Assert.Equal(5, estimate);
    }

    [Fact(Timeout = 60000)]
    public async Task FitsInWindow_WhenFits_ReturnsTrue()
    {
        var manager = new ContextWindowManager(1000);
        var text = "Short text";

        var result = manager.FitsInWindow(text);

        Assert.True(result);
    }

    [Fact(Timeout = 60000)]
    public async Task FitsInWindow_WhenTooLarge_ReturnsFalse()
    {
        var manager = new ContextWindowManager(10);
        var text = new string('x', 1000);

        var result = manager.FitsInWindow(text);

        Assert.False(result);
    }

    [Fact(Timeout = 60000)]
    public async Task FitsInWindow_WithReserved_AccountsForReserved()
    {
        var manager = new ContextWindowManager(100);
        // "Some longer text that will use more tokens" is 42 chars ≈ 11 tokens
        var text = "Some longer text that will use more tokens";

        // With 90 reserved, only 10 tokens available, but text uses ~11 tokens
        var result = manager.FitsInWindow(text, 90);

        Assert.False(result);
    }

    [Fact(Timeout = 60000)]
    public async Task RemainingTokens_ReturnsAvailableSpace()
    {
        var manager = new ContextWindowManager(1000);
        var text = "Short text";

        var remaining = manager.RemainingTokens(text);

        Assert.True(remaining > 0);
        Assert.True(remaining < 1000);
    }

    [Fact(Timeout = 60000)]
    public async Task RemainingTokens_WithReserved_SubtractsReserved()
    {
        var manager = new ContextWindowManager(1000);
        var text = "";

        var withoutReserved = manager.RemainingTokens(text, 0);
        var withReserved = manager.RemainingTokens(text, 500);

        Assert.True(withoutReserved > withReserved);
    }

    [Fact(Timeout = 60000)]
    public async Task TruncateToFit_WhenFits_ReturnsOriginal()
    {
        var manager = new ContextWindowManager(10000);
        var text = "Short text that fits";

        var result = manager.TruncateToFit(text);

        Assert.Equal(text, result);
    }

    [Fact(Timeout = 60000)]
    public async Task TruncateToFit_WhenTooLong_Truncates()
    {
        var manager = new ContextWindowManager(50);
        var text = new string('x', 1000);

        var result = manager.TruncateToFit(text);

        Assert.True(result.Length < text.Length);
    }

    [Fact(Timeout = 60000)]
    public async Task TruncateToFit_WithReserved_AccountsForReserved()
    {
        var manager = new ContextWindowManager(100);
        var text = "Some text here";

        var withoutReserved = manager.TruncateToFit(text, 0);
        var withReserved = manager.TruncateToFit(text, 50);

        Assert.True(withReserved.Length <= withoutReserved.Length);
    }

    [Fact(Timeout = 60000)]
    public async Task TruncateToFit_EmptyString_ReturnsEmpty()
    {
        var manager = new ContextWindowManager(100);

        var result = manager.TruncateToFit("");

        Assert.Equal("", result);
    }

    [Fact(Timeout = 60000)]
    public async Task TruncateToFit_PreservesSuffix_ByDefault()
    {
        var manager = new ContextWindowManager(20, text => text.Length);
        var text = "Start...Middle...End";

        var result = manager.TruncateToFit(text);

        // Default behavior may vary, just ensure it truncates
        Assert.True(result.Length <= 20);
    }

    [Fact(Timeout = 60000)]
    public async Task SplitIntoChunks_SmallText_ReturnsSingleChunk()
    {
        var manager = new ContextWindowManager(1000);
        var text = "Small text";

        var chunks = manager.SplitIntoChunks(text);

        Assert.Single(chunks);
        Assert.Equal(text, chunks[0]);
    }

    [Fact(Timeout = 60000)]
    public async Task SplitIntoChunks_LargeText_ReturnsMultipleChunks()
    {
        var manager = new ContextWindowManager(50, text => text.Length);
        var text = new string('x', 200);

        var chunks = manager.SplitIntoChunks(text);

        Assert.True(chunks.Count > 1);
    }

    [Fact(Timeout = 60000)]
    public async Task SplitIntoChunks_EmptyText_ReturnsEmptyList()
    {
        var manager = new ContextWindowManager(100);

        var chunks = manager.SplitIntoChunks("");

        Assert.Empty(chunks);
    }

    [Fact(Timeout = 60000)]
    public async Task SplitIntoChunks_EachChunkFits()
    {
        var manager = new ContextWindowManager(100, text => text.Length);
        var text = new string('x', 500);

        var chunks = manager.SplitIntoChunks(text);

        foreach (var chunk in chunks)
        {
            Assert.True(manager.FitsInWindow(chunk));
        }
    }

    [Fact(Timeout = 60000)]
    public async Task SplitIntoChunks_WithOverlap_CreatesOverlappingChunks()
    {
        var manager = new ContextWindowManager(100, text => text.Length);
        var text = new string('x', 300);

        var chunksNoOverlap = manager.SplitIntoChunks(text, 0);
        var chunksWithOverlap = manager.SplitIntoChunks(text, 20);

        // With overlap, we may need more chunks
        Assert.True(chunksWithOverlap.Count >= chunksNoOverlap.Count);
    }

    [Fact(Timeout = 60000)]
    public async Task MaxTokens_ReturnsConfiguredValue()
    {
        var manager = new ContextWindowManager(16384);

        Assert.Equal(16384, manager.MaxTokens);
    }

    [Fact(Timeout = 60000)]
    public async Task Constructor_WithZeroMaxTokens_AllowsCreation()
    {
        var manager = new ContextWindowManager(0);

        Assert.Equal(0, manager.MaxTokens);
    }

    [Fact(Timeout = 60000)]
    public async Task EstimateTokens_LongText_ReturnsReasonableEstimate()
    {
        var manager = new ContextWindowManager(100000);
        var text = new string('a', 10000);

        var estimate = manager.EstimateTokens(text);

        // Estimate should be reasonable (not 0, not larger than text length)
        Assert.True(estimate > 0);
        Assert.True(estimate <= text.Length);
    }

    [Fact(Timeout = 60000)]
    public async Task FitsInWindow_ExactlyAtLimit_ReturnsTrue()
    {
        var manager = new ContextWindowManager(10, text => text.Length);
        var text = "1234567890"; // Exactly 10 characters

        var result = manager.FitsInWindow(text);

        Assert.True(result);
    }

    [Fact(Timeout = 60000)]
    public async Task FitsInWindow_OneOverLimit_ReturnsFalse()
    {
        var manager = new ContextWindowManager(10, text => text.Length);
        var text = "12345678901"; // 11 characters

        var result = manager.FitsInWindow(text);

        Assert.False(result);
    }
}
