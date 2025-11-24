using Xunit;
using AiDotNet.Autodiff;
using AiDotNet.JitCompiler;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

/// <summary>
/// Tests for the main JitCompiler class.
/// </summary>
public class JitCompilerTests
{
    [Fact]
    public void Compile_SimpleGraph_Succeeds()
    {
        // Arrange
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = "Input"
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = "ReLU"
        };

        var jit = new JitCompiler();

        // Act
        var compiled = jit.Compile(result, new List<ComputationNode<float>> { input });

        // Assert
        Assert.NotNull(compiled);
    }

    [Fact]
    public void Compile_WithStats_ReturnsStatistics()
    {
        // Arrange
        var input1 = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = "Input"
        };
        var input2 = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = "Input"
        };

        var add = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input1, input2 })
        {
            OperationType = "Add"
        };

        var jit = new JitCompiler();

        // Act
        var (compiled, stats) = jit.CompileWithStats(add, new List<ComputationNode<float>> { input1, input2 });

        // Assert
        Assert.NotNull(compiled);
        Assert.NotNull(stats);
        Assert.True(stats.OriginalOperationCount >= 0);
        Assert.True(stats.OptimizedOperationCount >= 0);
        Assert.NotNull(stats.OptimizationsApplied);
        Assert.False(stats.CacheHit); // First compilation
    }

    [Fact]
    public void Compile_SecondTime_HitsCacheOptimized()
    {
        // Arrange
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = "Input"
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = "Exp"
        };

        var jit = new JitCompiler();

        // Act - First compilation
        var (compiled1, stats1) = jit.CompileWithStats(result, new List<ComputationNode<float>> { input });

        // Create new nodes with same structure
        var input2 = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = "Input"
        };

        var result2 = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input2 })
        {
            OperationType = "Exp"
        };

        // Act - Second compilation
        var (compiled2, stats2) = jit.CompileWithStats(result2, new List<ComputationNode<float>> { input2 });

        // Assert
        Assert.NotNull(compiled1);
        Assert.NotNull(compiled2);
        Assert.False(stats1.CacheHit);
        Assert.True(stats2.CacheHit);  // Should hit cache
        Assert.Equal(TimeSpan.Zero, stats2.CompilationTime);  // Cached, no compilation time
    }

    [Fact]
    public void JitCompiler_WithCustomOptions_RespectsConfiguration()
    {
        // Arrange
        var options = new JitCompilerOptions
        {
            EnableConstantFolding = false,
            EnableDeadCodeElimination = true,
            EnableOperationFusion = false,
            EnableCaching = false
        };

        var jit = new JitCompiler(options);

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = "Input"
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = "Log"
        };

        // Act
        var (compiled, stats) = jit.CompileWithStats(result, new List<ComputationNode<float>> { input });

        // Assert
        Assert.NotNull(compiled);
        Assert.DoesNotContain("Constant Folding", stats.OptimizationsApplied);
        Assert.Contains("Dead Code Elimination", stats.OptimizationsApplied);
        Assert.DoesNotContain("Operation Fusion", stats.OptimizationsApplied);
    }

    [Fact]
    public void ClearCache_RemovesAllCachedGraphs()
    {
        // Arrange
        var jit = new JitCompiler();

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = "Input"
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = "Sqrt"
        };

        // Compile once
        jit.Compile(result, new List<ComputationNode<float>> { input });

        var statsBefore = jit.GetCacheStats();
        Assert.True(statsBefore.CachedGraphCount > 0);

        // Act
        jit.ClearCache();

        // Assert
        var statsAfter = jit.GetCacheStats();
        Assert.Equal(0, statsAfter.CachedGraphCount);
    }

    [Fact]
    public void GetCacheStats_ReturnsCorrectCounts()
    {
        // Arrange
        var jit = new JitCompiler();

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = "Input"
        };

        // Act & Assert - Initially empty
        var stats1 = jit.GetCacheStats();
        Assert.Equal(0, stats1.CachedGraphCount);

        // Compile a graph
        var result1 = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = "ReLU"
        };
        jit.Compile(result1, new List<ComputationNode<float>> { input });

        var stats2 = jit.GetCacheStats();
        Assert.Equal(1, stats2.CachedGraphCount);

        // Compile another unique graph
        var result2 = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = "Sigmoid"
        };
        jit.Compile(result2, new List<ComputationNode<float>> { input });

        var stats3 = jit.GetCacheStats();
        Assert.Equal(2, stats3.CachedGraphCount);
    }

    [Fact]
    public void Compile_NullOutputNode_ThrowsException()
    {
        // Arrange
        var jit = new JitCompiler();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            jit.Compile<float>(null!, new List<ComputationNode<float>>()));
    }

    [Fact]
    public void Compile_NullInputList_ThrowsException()
    {
        // Arrange
        var jit = new JitCompiler();
        var output = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }));

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            jit.Compile(output, null!));
    }

    [Fact]
    public void CompilationStats_ToString_ContainsRelevantInfo()
    {
        // Arrange
        var stats = new CompilationStats
        {
            OriginalOperationCount = 10,
            OptimizedOperationCount = 6,
            OptimizationsApplied = new List<string> { "Constant Folding", "Dead Code Elimination" },
            CompilationTime = TimeSpan.FromMilliseconds(15.5),
            CacheHit = false
        };

        // Act
        var str = stats.ToString();

        // Assert
        Assert.Contains("10", str);
        Assert.Contains("6", str);
        Assert.Contains("Constant Folding", str);
        Assert.Contains("15.5", str);
        Assert.Contains("false", str);
    }

    [Fact]
    public void CompilationStats_OptimizationPercentage_CalculatesCorrectly()
    {
        // Arrange
        var stats = new CompilationStats
        {
            OriginalOperationCount = 100,
            OptimizedOperationCount = 60
        };

        // Act
        var percentage = stats.OptimizationPercentage;

        // Assert
        Assert.Equal(40.0, percentage);  // 40% reduction
    }

    [Fact]
    public void CacheStats_ToString_ContainsRelevantInfo()
    {
        // Arrange
        var stats = new CacheStats
        {
            CachedGraphCount = 5,
            EstimatedMemoryBytes = 10240
        };

        // Act
        var str = stats.ToString();

        // Assert
        Assert.Contains("5", str);
        Assert.Contains("10.00", str);  // KB
    }
}
