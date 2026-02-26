using AiDotNet.JitCompiler;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.JitCompiler;

/// <summary>
/// Deep integration tests for JitCompiler:
/// CacheStats (defaults, ToString),
/// CompilationStats (defaults, computed properties, ToString),
/// JitCompilerOptions (defaults),
/// JitCompatibilityResult (defaults, computed properties, ToString),
/// UnsupportedOperationInfo (defaults, ToString),
/// HybridCompilationResult (defaults, ToString).
/// </summary>
public class JitCompilerDeepMathIntegrationTests
{
    // ============================
    // CacheStats: Defaults
    // ============================

    [Fact]
    public void CacheStats_Defaults()
    {
        var stats = new CacheStats();
        Assert.Equal(0, stats.CachedGraphCount);
        Assert.Equal(0L, stats.EstimatedMemoryBytes);
    }

    [Fact]
    public void CacheStats_SetProperties()
    {
        var stats = new CacheStats
        {
            CachedGraphCount = 10,
            EstimatedMemoryBytes = 1024 * 1024 // 1 MB
        };

        Assert.Equal(10, stats.CachedGraphCount);
        Assert.Equal(1024 * 1024, stats.EstimatedMemoryBytes);
    }

    [Fact]
    public void CacheStats_ToString_ContainsCachedGraphs()
    {
        var stats = new CacheStats { CachedGraphCount = 5, EstimatedMemoryBytes = 2048 };
        string str = stats.ToString();
        Assert.Contains("5", str);
        Assert.Contains("2.00", str); // 2048 / 1024 = 2.00 KB
    }

    [Theory]
    [InlineData(0, "0.00")]
    [InlineData(1024, "1.00")]
    [InlineData(512, "0.50")]
    [InlineData(2560, "2.50")]
    public void CacheStats_ToString_MemoryInKb(long bytes, string expectedKb)
    {
        var stats = new CacheStats { EstimatedMemoryBytes = bytes };
        string str = stats.ToString();
        Assert.Contains(expectedKb, str);
    }

    // ============================
    // CompilationStats: Defaults
    // ============================

    [Fact]
    public void CompilationStats_Defaults()
    {
        var stats = new CompilationStats();
        Assert.Equal(0, stats.OriginalOperationCount);
        Assert.Equal(0, stats.OptimizedOperationCount);
        Assert.NotNull(stats.OptimizationsApplied);
        Assert.Empty(stats.OptimizationsApplied);
        Assert.Equal(TimeSpan.Zero, stats.CompilationTime);
        Assert.False(stats.CacheHit);
    }

    // ============================
    // CompilationStats: Computed Properties
    // ============================

    [Theory]
    [InlineData(100, 50, 50)]    // 50% reduction
    [InlineData(100, 100, 0)]    // No reduction
    [InlineData(100, 0, 100)]    // Complete elimination
    [InlineData(10, 3, 7)]       // 70% reduction
    public void CompilationStats_OperationsEliminated(int original, int optimized, int expectedEliminated)
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = original,
            OptimizedOperationCount = optimized
        };

        Assert.Equal(expectedEliminated, stats.OperationsEliminated);
    }

    [Theory]
    [InlineData(100, 50, 50.0)]   // 50%
    [InlineData(100, 100, 0.0)]   // 0%
    [InlineData(100, 0, 100.0)]   // 100%
    [InlineData(10, 3, 70.0)]     // 70%
    [InlineData(0, 0, 0.0)]       // Edge: no operations
    public void CompilationStats_OptimizationPercentage(int original, int optimized, double expectedPercent)
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = original,
            OptimizedOperationCount = optimized
        };

        Assert.Equal(expectedPercent, stats.OptimizationPercentage, 1e-10);
    }

    [Fact]
    public void CompilationStats_ToString_ContainsAllInfo()
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = 100,
            OptimizedOperationCount = 60,
            OptimizationsApplied = new List<string> { "ConstantFolding", "DeadCodeElimination" },
            CompilationTime = TimeSpan.FromMilliseconds(15.5),
            CacheHit = false
        };

        string str = stats.ToString();
        Assert.Contains("100", str);
        Assert.Contains("60", str);
        Assert.Contains("40", str); // 100 - 60
        Assert.Contains("40.0%", str);
        Assert.Contains("ConstantFolding", str);
        Assert.Contains("DeadCodeElimination", str);
        Assert.Contains("False", str); // CacheHit
    }

    // ============================
    // JitCompilerOptions: Defaults
    // ============================

    [Fact]
    public void JitCompilerOptions_Defaults_EnableConstantFolding()
    {
        var options = new JitCompilerOptions();
        Assert.True(options.EnableConstantFolding);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_EnableDeadCodeElimination()
    {
        var options = new JitCompilerOptions();
        Assert.True(options.EnableDeadCodeElimination);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_EnableOperationFusion()
    {
        var options = new JitCompilerOptions();
        Assert.True(options.EnableOperationFusion);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_EnableCaching()
    {
        var options = new JitCompilerOptions();
        Assert.True(options.EnableCaching);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_EnableLoopUnrolling()
    {
        var options = new JitCompilerOptions();
        Assert.True(options.EnableLoopUnrolling);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_AdaptiveFusionDisabled()
    {
        var options = new JitCompilerOptions();
        Assert.False(options.EnableAdaptiveFusion);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_AutoTuningEnabled()
    {
        var options = new JitCompilerOptions();
        Assert.True(options.EnableAutoTuning);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_SIMDHintsDisabled()
    {
        var options = new JitCompilerOptions();
        Assert.False(options.EnableSIMDHints);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_MemoryPoolingEnabled()
    {
        var options = new JitCompilerOptions();
        Assert.True(options.EnableMemoryPooling);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_MaxPoolSizePerShape10()
    {
        var options = new JitCompilerOptions();
        Assert.Equal(10, options.MaxPoolSizePerShape);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_MaxElementsToPool10M()
    {
        var options = new JitCompilerOptions();
        Assert.Equal(10_000_000, options.MaxElementsToPool);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_FallbackHandling()
    {
        var options = new JitCompilerOptions();
        Assert.Equal(UnsupportedLayerHandling.Fallback, options.UnsupportedLayerHandling);
    }

    [Fact]
    public void JitCompilerOptions_Defaults_LogUnsupportedTrue()
    {
        var options = new JitCompilerOptions();
        Assert.True(options.LogUnsupportedOperations);
    }

    [Fact]
    public void JitCompilerOptions_SetAllOptions()
    {
        var options = new JitCompilerOptions
        {
            EnableConstantFolding = false,
            EnableDeadCodeElimination = false,
            EnableOperationFusion = false,
            EnableCaching = false,
            EnableLoopUnrolling = false,
            EnableAdaptiveFusion = true,
            EnableAutoTuning = false,
            EnableSIMDHints = true,
            EnableMemoryPooling = false,
            MaxPoolSizePerShape = 20,
            MaxElementsToPool = 5_000_000,
            UnsupportedLayerHandling = UnsupportedLayerHandling.Hybrid,
            LogUnsupportedOperations = false
        };

        Assert.False(options.EnableConstantFolding);
        Assert.False(options.EnableDeadCodeElimination);
        Assert.True(options.EnableAdaptiveFusion);
        Assert.Equal(UnsupportedLayerHandling.Hybrid, options.UnsupportedLayerHandling);
        Assert.Equal(20, options.MaxPoolSizePerShape);
    }

    // ============================
    // JitCompilerOptions: Memory Math
    // ============================

    [Theory]
    [InlineData(10_000_000, 4, 40_000_000)]   // float32: 40MB
    [InlineData(10_000_000, 8, 80_000_000)]   // float64: 80MB
    public void JitCompilerOptions_MaxPoolMemory_Calculation(int maxElements, int bytesPerElement, long expectedBytes)
    {
        var options = new JitCompilerOptions { MaxElementsToPool = maxElements };
        long approxMemory = (long)options.MaxElementsToPool * bytesPerElement;
        Assert.Equal(expectedBytes, approxMemory);
    }

    // ============================
    // JitCompatibilityResult: Defaults
    // ============================

    [Fact]
    public void JitCompatibilityResult_Defaults()
    {
        var result = new JitCompatibilityResult();
        Assert.False(result.IsFullySupported);
        Assert.NotNull(result.SupportedOperations);
        Assert.Empty(result.SupportedOperations);
        Assert.NotNull(result.UnsupportedOperations);
        Assert.Empty(result.UnsupportedOperations);
        Assert.False(result.CanUseHybridMode);
    }

    // ============================
    // JitCompatibilityResult: SupportedPercentage
    // ============================

    [Fact]
    public void JitCompatibilityResult_SupportedPercentage_AllSupported()
    {
        var result = new JitCompatibilityResult
        {
            SupportedOperations = new List<string> { "add", "mul", "relu" }
        };

        Assert.Equal(100.0, result.SupportedPercentage, 1e-10);
    }

    [Fact]
    public void JitCompatibilityResult_SupportedPercentage_HalfSupported()
    {
        var result = new JitCompatibilityResult
        {
            SupportedOperations = new List<string> { "add", "mul" },
            UnsupportedOperations = new List<UnsupportedOperationInfo>
            {
                new() { OperationType = "custom1" },
                new() { OperationType = "custom2" }
            }
        };

        Assert.Equal(50.0, result.SupportedPercentage, 1e-10);
    }

    [Fact]
    public void JitCompatibilityResult_SupportedPercentage_NoneSupported()
    {
        var result = new JitCompatibilityResult
        {
            UnsupportedOperations = new List<UnsupportedOperationInfo>
            {
                new() { OperationType = "custom1" }
            }
        };

        Assert.Equal(0.0, result.SupportedPercentage, 1e-10);
    }

    [Fact]
    public void JitCompatibilityResult_SupportedPercentage_EmptyIs100()
    {
        var result = new JitCompatibilityResult();
        Assert.Equal(100.0, result.SupportedPercentage, 1e-10);
    }

    [Fact]
    public void JitCompatibilityResult_ToString_FullySupported()
    {
        var result = new JitCompatibilityResult
        {
            IsFullySupported = true,
            SupportedOperations = new List<string> { "add", "mul", "relu" }
        };

        string str = result.ToString();
        Assert.Contains("Fully JIT compatible", str);
        Assert.Contains("3", str);
    }

    [Fact]
    public void JitCompatibilityResult_ToString_PartialSupport()
    {
        var result = new JitCompatibilityResult
        {
            IsFullySupported = false,
            SupportedOperations = new List<string> { "add" },
            UnsupportedOperations = new List<UnsupportedOperationInfo>
            {
                new() { OperationType = "custom" }
            },
            CanUseHybridMode = true
        };

        string str = result.ToString();
        Assert.Contains("Partial", str);
        Assert.Contains("50.0%", str);
        Assert.Contains("available", str);
    }

    // ============================
    // UnsupportedOperationInfo: Defaults
    // ============================

    [Fact]
    public void UnsupportedOperationInfo_Defaults()
    {
        var info = new UnsupportedOperationInfo();
        Assert.Equal("", info.OperationType);
        Assert.Null(info.NodeName);
        Assert.Equal(0, info.TensorId);
        Assert.Equal("Operation type not implemented in JIT compiler", info.Reason);
        Assert.True(info.CanFallback);
    }

    [Fact]
    public void UnsupportedOperationInfo_ToString_WithNodeName()
    {
        var info = new UnsupportedOperationInfo
        {
            OperationType = "CustomActivation",
            NodeName = "layer_3",
            TensorId = 42,
            Reason = "Not implemented"
        };

        string str = info.ToString();
        Assert.Contains("CustomActivation", str);
        Assert.Contains("layer_3", str);
        Assert.Contains("42", str);
        Assert.Contains("Not implemented", str);
    }

    [Fact]
    public void UnsupportedOperationInfo_ToString_WithoutNodeName()
    {
        var info = new UnsupportedOperationInfo
        {
            OperationType = "Gather",
            TensorId = 10
        };

        string str = info.ToString();
        Assert.Contains("Gather", str);
        Assert.Contains("10", str);
        Assert.DoesNotContain("(", str); // No node name parentheses
    }

    // ============================
    // HybridCompilationResult: Defaults
    // ============================

    [Fact]
    public void HybridCompilationResult_Defaults()
    {
        var result = new HybridCompilationResult<double>();
        Assert.False(result.IsFullyJitCompiled);
        Assert.Equal("Unknown", result.ExecutionMode);
        Assert.NotNull(result.Compatibility);
        Assert.NotNull(result.Warnings);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void HybridCompilationResult_ToString_FullJit()
    {
        var result = new HybridCompilationResult<double>
        {
            IsFullyJitCompiled = true,
            ExecutionMode = "JIT"
        };

        string str = result.ToString();
        Assert.Contains("JIT", str);
        Assert.Contains("100%", str);
    }

    [Fact]
    public void HybridCompilationResult_ToString_WithWarnings()
    {
        var result = new HybridCompilationResult<double>
        {
            IsFullyJitCompiled = false,
            ExecutionMode = "Hybrid",
            Warnings = new List<string> { "warning1", "warning2" },
            Compatibility = new JitCompatibilityResult
            {
                SupportedOperations = new List<string> { "add", "mul" },
                UnsupportedOperations = new List<UnsupportedOperationInfo>
                {
                    new() { OperationType = "custom" }
                }
            }
        };

        string str = result.ToString();
        Assert.Contains("Hybrid", str);
        Assert.Contains("2 warnings", str);
    }

    // ============================
    // UnsupportedLayerHandling Enum
    // ============================

    [Theory]
    [InlineData(UnsupportedLayerHandling.Throw)]
    [InlineData(UnsupportedLayerHandling.Fallback)]
    [InlineData(UnsupportedLayerHandling.Hybrid)]
    [InlineData(UnsupportedLayerHandling.Skip)]
    public void UnsupportedLayerHandling_AllValuesValid(UnsupportedLayerHandling handling)
    {
        Assert.True(Enum.IsDefined(handling));
    }

    [Fact]
    public void UnsupportedLayerHandling_HasFourValues()
    {
        var values = Enum.GetValues<UnsupportedLayerHandling>();
        Assert.Equal(4, values.Length);
    }
}
