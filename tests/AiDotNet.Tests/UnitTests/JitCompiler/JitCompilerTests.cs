using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.JitCompiler;
using Xunit;
using JitCompilerClass = AiDotNet.JitCompiler.JitCompiler;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

/// <summary>
/// Tests for the main JitCompiler class.
/// </summary>
/// <remarks>
/// These tests are quarantined because they trigger GPU initialization which can fail
/// on machines without proper GPU support or drivers.
/// </remarks>
[Trait("Category", "GPU")]
public class JitCompilerTests
{
    [Fact]
    public void Compile_SimpleGraph_Succeeds()
    {
        // Arrange
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.ReLU
        };

        var jit = new JitCompilerClass();

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
            OperationType = OperationType.Input
        };
        var input2 = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        var add = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input1, input2 })
        {
            OperationType = OperationType.Add
        };

        var jit = new JitCompilerClass();

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
            OperationType = OperationType.Input
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Exp
        };

        var jit = new JitCompilerClass();

        // Act - First compilation
        var (compiled1, stats1) = jit.CompileWithStats(result, new List<ComputationNode<float>> { input });

        // Create new nodes with same structure
        var input2 = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        var result2 = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input2 })
        {
            OperationType = OperationType.Exp
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

        var jit = new JitCompilerClass(options);

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Log
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
        var jit = new JitCompilerClass();

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Sqrt
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
        var jit = new JitCompilerClass();

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        // Act & Assert - Initially empty
        var stats1 = jit.GetCacheStats();
        Assert.Equal(0, stats1.CachedGraphCount);

        // Compile a graph
        var result1 = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.ReLU
        };
        jit.Compile(result1, new List<ComputationNode<float>> { input });

        var stats2 = jit.GetCacheStats();
        Assert.Equal(1, stats2.CachedGraphCount);

        // Compile another unique graph
        var result2 = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Sigmoid
        };
        jit.Compile(result2, new List<ComputationNode<float>> { input });

        var stats3 = jit.GetCacheStats();
        Assert.Equal(2, stats3.CachedGraphCount);
    }

    [Fact]
    public void Compile_NullOutputNode_ThrowsException()
    {
        // Arrange
        var jit = new JitCompilerClass();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            jit.Compile<float>(null!, new List<ComputationNode<float>>()));
    }

    [Fact]
    public void Compile_NullInputList_ThrowsException()
    {
        // Arrange
        var jit = new JitCompilerClass();
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
            CompilationTime = TimeSpan.FromTicks(155_000), // 15.5ms (use ticks to avoid .NET Framework FromMilliseconds rounding)
            CacheHit = false
        };

        // Act
        var str = stats.ToString();

        // Assert
        Assert.Contains("10", str);
        Assert.Contains("6", str);
        Assert.Contains("Constant Folding", str);
        Assert.Contains("15.5", str);
        Assert.Contains("False", str);
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

    #region Unsupported Layer Handling Tests

    [Fact]
    public void GetSupportedOperationTypes_ReturnsExpectedOperations()
    {
        // Act
        var supportedOps = JitCompilerClass.GetSupportedOperationTypes();

        // Assert
        Assert.Contains(OperationType.Add, supportedOps);
        Assert.Contains(OperationType.Subtract, supportedOps);
        Assert.Contains(OperationType.Multiply, supportedOps);
        Assert.Contains(OperationType.ReLU, supportedOps);
        Assert.Contains(OperationType.Sigmoid, supportedOps);
        Assert.Contains(OperationType.MatMul, supportedOps);
        Assert.Contains(OperationType.Conv2D, supportedOps);
        Assert.Contains(OperationType.MaxPool2D, supportedOps);
        Assert.Contains(OperationType.BatchNorm, supportedOps);
        Assert.Contains(OperationType.LSTMCell, supportedOps);
        Assert.Contains(OperationType.GRUCell, supportedOps);
    }

    [Fact]
    public void AnalyzeCompatibility_FullySupportedGraph_ReturnsFullySupported()
    {
        // Arrange
        var jit = new JitCompilerClass();
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.MatMul  // Just for testing, Input doesn't need OperationType
        };

        var relu = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.ReLU
        };

        // Act
        var result = jit.AnalyzeCompatibility(relu, new List<ComputationNode<float>> { input });

        // Assert
        Assert.True(result.IsFullySupported);
        Assert.Empty(result.UnsupportedOperations);
        Assert.Single(result.SupportedOperations);
        Assert.Equal(100.0, result.SupportedPercentage);
        Assert.True(result.CanUseHybridMode);
    }

    [Fact]
    public void AnalyzeCompatibility_GraphWithUnsupportedOp_ReturnsPartialSupport()
    {
        // Arrange
        var jit = new JitCompilerClass();
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            Name = "input"
        };

        // Create a node with an unsupported operation type (no OperationType set)
        var unsupportedNode = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            Name = "unsupported_op",
            OperationType = null  // Unsupported - no operation type
        };

        // Act
        var result = jit.AnalyzeCompatibility(unsupportedNode, new List<ComputationNode<float>> { input });

        // Assert
        Assert.False(result.IsFullySupported);
        Assert.Single(result.UnsupportedOperations);
        Assert.Contains("Unknown", result.UnsupportedOperations[0].OperationType);
        Assert.True(result.CanUseHybridMode);  // Can still fallback
    }

    [Fact]
    public void CompileWithUnsupportedHandling_ThrowMode_FullySupported_Succeeds()
    {
        // Arrange
        var options = new JitCompilerOptions
        {
            UnsupportedLayerHandling = UnsupportedLayerHandling.Throw
        };
        var jit = new JitCompilerClass(options);

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }));
        var relu = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.ReLU
        };

        // Act
        var result = jit.CompileWithUnsupportedHandling(relu, new List<ComputationNode<float>> { input });

        // Assert
        Assert.NotNull(result.CompiledFunc);
        Assert.True(result.IsFullyJitCompiled);
        Assert.Equal("JIT", result.ExecutionMode);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void CompileWithUnsupportedHandling_ThrowMode_UnsupportedOp_ThrowsException()
    {
        // Arrange
        var options = new JitCompilerOptions
        {
            UnsupportedLayerHandling = UnsupportedLayerHandling.Throw
        };
        var jit = new JitCompilerClass(options);

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }));
        var unsupportedNode = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = null  // No operation type = unsupported
        };

        // Act & Assert
        Assert.Throws<NotSupportedException>(() =>
            jit.CompileWithUnsupportedHandling(unsupportedNode, new List<ComputationNode<float>> { input }));
    }

    [Fact]
    public void CompileWithUnsupportedHandling_FallbackMode_UnsupportedOp_FallsBackToInterpreted()
    {
        // Arrange
        var options = new JitCompilerOptions
        {
            UnsupportedLayerHandling = UnsupportedLayerHandling.Fallback,
            LogUnsupportedOperations = true
        };
        var jit = new JitCompilerClass(options);

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }));
        var unsupportedNode = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = null  // No operation type = unsupported
        };

        // Act
        var result = jit.CompileWithUnsupportedHandling(unsupportedNode, new List<ComputationNode<float>> { input });

        // Assert
        Assert.NotNull(result.CompiledFunc);
        Assert.False(result.IsFullyJitCompiled);
        Assert.Equal("Interpreted", result.ExecutionMode);
        Assert.NotEmpty(result.Warnings);
        Assert.Contains(result.Warnings, w => w.Contains("interpreted"));
    }

    [Fact]
    public void CompileWithUnsupportedHandling_HybridMode_UnsupportedOp_UsesHybridExecution()
    {
        // Arrange
        var options = new JitCompilerOptions
        {
            UnsupportedLayerHandling = UnsupportedLayerHandling.Hybrid,
            LogUnsupportedOperations = true
        };
        var jit = new JitCompilerClass(options);

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }));
        var unsupportedNode = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = null  // No operation type = unsupported
        };

        // Act
        var result = jit.CompileWithUnsupportedHandling(unsupportedNode, new List<ComputationNode<float>> { input });

        // Assert
        Assert.NotNull(result.CompiledFunc);
        Assert.False(result.IsFullyJitCompiled);
        Assert.Equal("Hybrid", result.ExecutionMode);
        Assert.True(result.Compatibility.CanUseHybridMode);
    }

    [Fact]
    public void CompileWithUnsupportedHandling_SkipMode_UnsupportedOp_SkipsWithWarning()
    {
        // Arrange
        var options = new JitCompilerOptions
        {
            UnsupportedLayerHandling = UnsupportedLayerHandling.Skip,
            LogUnsupportedOperations = true
        };
        var jit = new JitCompilerClass(options);

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }));
        var unsupportedNode = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = null  // No operation type = unsupported
        };

        // Act
        var result = jit.CompileWithUnsupportedHandling(unsupportedNode, new List<ComputationNode<float>> { input });

        // Assert
        Assert.NotNull(result.CompiledFunc);
        Assert.NotEmpty(result.Warnings);
        Assert.Contains(result.Warnings, w => w.Contains("WARNING") || w.Contains("skipped"));
    }

    [Fact]
    public void UnsupportedOperationInfo_ToString_FormatsCorrectly()
    {
        // Arrange
        var info = new UnsupportedOperationInfo
        {
            OperationType = "CustomOp",
            NodeName = "my_layer",
            TensorId = 42,
            Reason = "Not implemented"
        };

        // Act
        var str = info.ToString();

        // Assert
        Assert.Contains("CustomOp", str);
        Assert.Contains("my_layer", str);
        Assert.Contains("42", str);
        Assert.Contains("Not implemented", str);
    }

    [Fact]
    public void JitCompatibilityResult_SupportedPercentage_CalculatesCorrectly()
    {
        // Arrange
        var result = new JitCompatibilityResult
        {
            SupportedOperations = new List<string> { "Add", "ReLU", "MatMul" },
            UnsupportedOperations = new List<UnsupportedOperationInfo>
            {
                new() { OperationType = "CustomOp1" },
                new() { OperationType = "CustomOp2" }
            }
        };

        // Act
        var percentage = result.SupportedPercentage;

        // Assert
        Assert.Equal(60.0, percentage);  // 3 out of 5 = 60%
    }

    [Fact]
    public void JitCompatibilityResult_ToString_FullySupported()
    {
        // Arrange
        var result = new JitCompatibilityResult
        {
            IsFullySupported = true,
            SupportedOperations = new List<string> { "Add", "ReLU" }
        };

        // Act
        var str = result.ToString();

        // Assert
        Assert.Contains("Fully JIT compatible", str);
        Assert.Contains("2 operations", str);
    }

    [Fact]
    public void JitCompatibilityResult_ToString_PartialSupport()
    {
        // Arrange
        var result = new JitCompatibilityResult
        {
            IsFullySupported = false,
            CanUseHybridMode = true,
            SupportedOperations = new List<string> { "Add" },
            UnsupportedOperations = new List<UnsupportedOperationInfo>
            {
                new() { OperationType = "CustomOp" }
            }
        };

        // Act
        var str = result.ToString();

        // Assert
        Assert.Contains("Partial JIT support", str);
        Assert.Contains("50.0%", str);
        Assert.Contains("Hybrid mode: available", str);
    }

    [Fact]
    public void HybridCompilationResult_ToString_FormatsCorrectly()
    {
        // Arrange
        var result = new HybridCompilationResult<float>
        {
            IsFullyJitCompiled = false,
            ExecutionMode = "Hybrid",
            Compatibility = new JitCompatibilityResult
            {
                SupportedOperations = new List<string> { "Add", "ReLU", "MatMul" },
                UnsupportedOperations = new List<UnsupportedOperationInfo>
                {
                    new() { OperationType = "CustomOp" }
                }
            },
            Warnings = new List<string> { "Some operations use fallback" }
        };

        // Act
        var str = result.ToString();

        // Assert
        Assert.Contains("Hybrid", str);
        Assert.Contains("75.0%", str);
        Assert.Contains("1 warnings", str);
    }

    [Fact]
    public void JitCompilerOptions_UnsupportedLayerHandling_DefaultIsFallback()
    {
        // Arrange
        var options = new JitCompilerOptions();

        // Assert
        Assert.Equal(UnsupportedLayerHandling.Fallback, options.UnsupportedLayerHandling);
        Assert.True(options.LogUnsupportedOperations);
    }

    #endregion

    #region Extended Operation Support Tests

    [Fact]
    public void GetSupportedOperationTypes_IncludesExtendedActivations()
    {
        // Act
        var supportedOps = JitCompilerClass.GetSupportedOperationTypes();

        // Assert - Extended activation functions
        Assert.Contains(OperationType.ELU, supportedOps);
        Assert.Contains(OperationType.LeakyReLU, supportedOps);
        Assert.Contains(OperationType.GELU, supportedOps);
        Assert.Contains(OperationType.Swish, supportedOps);
        Assert.Contains(OperationType.Mish, supportedOps);
        Assert.Contains(OperationType.SoftPlus, supportedOps);
        Assert.Contains(OperationType.SELU, supportedOps);
        Assert.Contains(OperationType.HardSigmoid, supportedOps);
        Assert.Contains(OperationType.HardTanh, supportedOps);
        Assert.Contains(OperationType.SoftSign, supportedOps);
        Assert.Contains(OperationType.CELU, supportedOps);
        Assert.Contains(OperationType.LogSoftmax, supportedOps);
        Assert.Contains(OperationType.PReLU, supportedOps);
        Assert.Contains(OperationType.ThresholdedReLU, supportedOps);
    }

    [Fact]
    public void GetSupportedOperationTypes_IncludesExtendedShapeOps()
    {
        // Act
        var supportedOps = JitCompilerClass.GetSupportedOperationTypes();

        // Assert - Shape operations
        Assert.Contains(OperationType.Split, supportedOps);
        Assert.Contains(OperationType.Slice, supportedOps);
        Assert.Contains(OperationType.Square, supportedOps);
        Assert.Contains(OperationType.Norm, supportedOps);
    }

    [Fact]
    public void GetSupportedOperationTypes_IncludesEmbeddingAndAttentionOps()
    {
        // Act
        var supportedOps = JitCompilerClass.GetSupportedOperationTypes();

        // Assert - Embedding and attention operations
        Assert.Contains(OperationType.Embedding, supportedOps);
        Assert.Contains(OperationType.ScaledDotProductAttention, supportedOps);
        Assert.Contains(OperationType.MultiHeadAttention, supportedOps);
    }

    [Fact]
    public void GetSupportedOperationTypes_IncludesFusedOps()
    {
        // Act
        var supportedOps = JitCompilerClass.GetSupportedOperationTypes();

        // Assert - Fused operations
        Assert.Contains(OperationType.FusedMatMulAdd, supportedOps);
        Assert.Contains(OperationType.FusedLinearReLU, supportedOps);
        Assert.Contains(OperationType.FusedConvBatchNorm, supportedOps);
        Assert.Contains(OperationType.FusedAddReLU, supportedOps);
    }

    [Fact]
    public void GetSupportedOperationTypes_IncludesComplexNumberOps()
    {
        // Act
        var supportedOps = JitCompilerClass.GetSupportedOperationTypes();

        // Assert - Complex number operations
        Assert.Contains(OperationType.ComplexMatMul, supportedOps);
        Assert.Contains(OperationType.ComplexMultiply, supportedOps);
    }

    [Fact]
    public void AnalyzeCompatibility_ExtendedActivation_IsSupported()
    {
        // Arrange
        var jit = new JitCompilerClass();
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }));

        var gelu = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.GELU
        };

        // Act
        var result = jit.AnalyzeCompatibility(gelu, new List<ComputationNode<float>> { input });

        // Assert
        Assert.True(result.IsFullySupported);
        Assert.Empty(result.UnsupportedOperations);
    }

    [Fact]
    public void AnalyzeCompatibility_AttentionOp_IsSupported()
    {
        // Arrange
        var jit = new JitCompilerClass();
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }));

        var attention = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.ScaledDotProductAttention
        };

        // Act
        var result = jit.AnalyzeCompatibility(attention, new List<ComputationNode<float>> { input });

        // Assert
        Assert.True(result.IsFullySupported);
        Assert.Empty(result.UnsupportedOperations);
    }

    [Fact]
    public void AnalyzeCompatibility_EmbeddingOp_IsSupported()
    {
        // Arrange
        var jit = new JitCompilerClass();
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }));

        var embedding = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Embedding
        };

        // Act
        var result = jit.AnalyzeCompatibility(embedding, new List<ComputationNode<float>> { input });

        // Assert
        Assert.True(result.IsFullySupported);
        Assert.Empty(result.UnsupportedOperations);
    }

    [Fact]
    public void AnalyzeCompatibility_FusedOp_IsSupported()
    {
        // Arrange
        var jit = new JitCompilerClass();
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }));

        var fusedOp = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.FusedLinearReLU
        };

        // Act
        var result = jit.AnalyzeCompatibility(fusedOp, new List<ComputationNode<float>> { input });

        // Assert
        Assert.True(result.IsFullySupported);
        Assert.Empty(result.UnsupportedOperations);
    }

    [Fact]
    public void GetSupportedOperationTypes_CountIsSignificantlyHigher()
    {
        // Act
        var supportedOps = JitCompilerClass.GetSupportedOperationTypes();

        // Assert - We should now support many more operations
        // Originally ~45, now should be ~65+ with the new additions
        Assert.True(supportedOps.Count >= 60,
            $"Expected at least 60 supported operations, but got {supportedOps.Count}");
    }

    #endregion
}
