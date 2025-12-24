using AiDotNet.JitCompiler;
using Xunit;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

public sealed class CompilationStatsTests
{
    [Fact]
    public void ToString_IncludesKeyFields()
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = 10,
            OptimizedOperationCount = 7,
            OptimizationsApplied = ["Constant Folding"],
            CompilationTime = TimeSpan.FromMilliseconds(12.34),
            CacheHit = false
        };

        var text = stats.ToString();

        Assert.Contains("Compilation Stats:", text);
        Assert.Contains("Original operations: 10", text);
        Assert.Contains("Optimized operations: 7", text);
        Assert.Contains("Optimizations applied: Constant Folding", text);
        Assert.Contains("Compilation time:", text);
        Assert.Contains("Cache hit: False", text);
    }
}

