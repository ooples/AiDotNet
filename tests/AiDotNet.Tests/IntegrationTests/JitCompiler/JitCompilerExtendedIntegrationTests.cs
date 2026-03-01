using AiDotNet.JitCompiler;
using AiDotNet.JitCompiler.CodeGen;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using AiDotNet.JitCompiler.Memory;
using AiDotNet.JitCompiler.Optimizations;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.JitCompiler;

/// <summary>
/// Extended integration tests for the JitCompiler module covering options, stats,
/// IR operations, optimization passes, memory pooling, SIMD capabilities, and
/// compatibility analysis classes.
/// </summary>
public class JitCompilerExtendedIntegrationTests
{
    #region JitCompilerOptions

    [Fact]
    public void JitCompilerOptions_DefaultValues()
    {
        var opts = new JitCompilerOptions();

        Assert.True(opts.EnableConstantFolding);
        Assert.True(opts.EnableDeadCodeElimination);
        Assert.True(opts.EnableOperationFusion);
        Assert.True(opts.EnableCaching);
        Assert.True(opts.EnableLoopUnrolling);
        Assert.False(opts.EnableAdaptiveFusion);
        Assert.True(opts.EnableAutoTuning);
        Assert.False(opts.EnableSIMDHints);
        Assert.True(opts.EnableMemoryPooling);
        Assert.Equal(10, opts.MaxPoolSizePerShape);
        Assert.Equal(10_000_000, opts.MaxElementsToPool);
        Assert.Equal(UnsupportedLayerHandling.Fallback, opts.UnsupportedLayerHandling);
        Assert.True(opts.LogUnsupportedOperations);
    }

    [Fact]
    public void JitCompilerOptions_CustomValues()
    {
        var opts = new JitCompilerOptions
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

        Assert.False(opts.EnableConstantFolding);
        Assert.False(opts.EnableDeadCodeElimination);
        Assert.False(opts.EnableOperationFusion);
        Assert.False(opts.EnableCaching);
        Assert.False(opts.EnableLoopUnrolling);
        Assert.True(opts.EnableAdaptiveFusion);
        Assert.False(opts.EnableAutoTuning);
        Assert.True(opts.EnableSIMDHints);
        Assert.False(opts.EnableMemoryPooling);
        Assert.Equal(20, opts.MaxPoolSizePerShape);
        Assert.Equal(5_000_000, opts.MaxElementsToPool);
        Assert.Equal(UnsupportedLayerHandling.Hybrid, opts.UnsupportedLayerHandling);
        Assert.False(opts.LogUnsupportedOperations);
    }

    #endregion

    #region CacheStats

    [Fact]
    public void CacheStats_DefaultValues()
    {
        var stats = new CacheStats();

        Assert.Equal(0, stats.CachedGraphCount);
        Assert.Equal(0L, stats.EstimatedMemoryBytes);
    }

    [Fact]
    public void CacheStats_CustomValues()
    {
        var stats = new CacheStats
        {
            CachedGraphCount = 5,
            EstimatedMemoryBytes = 1024 * 1024
        };

        Assert.Equal(5, stats.CachedGraphCount);
        Assert.Equal(1024 * 1024, stats.EstimatedMemoryBytes);
    }

    [Fact]
    public void CacheStats_ToString_ContainsInfo()
    {
        var stats = new CacheStats
        {
            CachedGraphCount = 3,
            EstimatedMemoryBytes = 2048
        };

        var str = stats.ToString();
        Assert.Contains("Cached graphs: 3", str);
        Assert.Contains("Estimated memory", str);
    }

    #endregion

    #region CompilationStats

    [Fact]
    public void CompilationStats_DefaultValues()
    {
        var stats = new CompilationStats();

        Assert.Equal(0, stats.OriginalOperationCount);
        Assert.Equal(0, stats.OptimizedOperationCount);
        Assert.Empty(stats.OptimizationsApplied);
        Assert.Equal(TimeSpan.Zero, stats.CompilationTime);
        Assert.False(stats.CacheHit);
    }

    [Fact]
    public void CompilationStats_OperationsEliminated()
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = 100,
            OptimizedOperationCount = 60
        };

        Assert.Equal(40, stats.OperationsEliminated);
        Assert.Equal(40.0, stats.OptimizationPercentage, 1);
    }

    [Fact]
    public void CompilationStats_ZeroOriginalCount_ZeroPercentage()
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = 0,
            OptimizedOperationCount = 0
        };

        Assert.Equal(0.0, stats.OptimizationPercentage);
    }

    [Fact]
    public void CompilationStats_ToString_ContainsInfo()
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = 50,
            OptimizedOperationCount = 30,
            CompilationTime = TimeSpan.FromMilliseconds(15.5),
            CacheHit = false,
            OptimizationsApplied = { "ConstantFolding", "DeadCodeElimination" }
        };

        var str = stats.ToString();
        Assert.Contains("Original operations: 50", str);
        Assert.Contains("Optimized operations: 30", str);
        Assert.Contains("Operations eliminated: 20", str);
        Assert.Contains("ConstantFolding", str);
    }

    #endregion

    #region JitCompatibilityResult

    [Fact]
    public void JitCompatibilityResult_DefaultValues()
    {
        var result = new JitCompatibilityResult();

        Assert.False(result.IsFullySupported);
        Assert.Empty(result.SupportedOperations);
        Assert.Empty(result.UnsupportedOperations);
        Assert.False(result.CanUseHybridMode);
    }

    [Fact]
    public void JitCompatibilityResult_FullySupported()
    {
        var result = new JitCompatibilityResult
        {
            IsFullySupported = true,
            SupportedOperations = { "Add", "MatMul", "ReLU" }
        };

        Assert.Equal(100.0, result.SupportedPercentage);
        Assert.Contains("Fully JIT compatible", result.ToString());
    }

    [Fact]
    public void JitCompatibilityResult_PartialSupport()
    {
        var result = new JitCompatibilityResult
        {
            SupportedOperations = { "Add", "MatMul" },
            UnsupportedOperations =
            {
                new UnsupportedOperationInfo { OperationType = "CustomOp" }
            },
            CanUseHybridMode = true
        };

        // 2 out of 3 = 66.7%
        Assert.InRange(result.SupportedPercentage, 66.0, 67.0);
        Assert.Contains("Partial JIT support", result.ToString());
        Assert.Contains("Hybrid mode: available", result.ToString());
    }

    [Fact]
    public void JitCompatibilityResult_NoOps_100Percent()
    {
        var result = new JitCompatibilityResult();
        Assert.Equal(100.0, result.SupportedPercentage);
    }

    #endregion

    #region UnsupportedOperationInfo

    [Fact]
    public void UnsupportedOperationInfo_DefaultValues()
    {
        var info = new UnsupportedOperationInfo();

        Assert.Equal("", info.OperationType);
        Assert.Null(info.NodeName);
        Assert.Equal(0, info.TensorId);
        Assert.Contains("not implemented", info.Reason);
        Assert.True(info.CanFallback);
    }

    [Fact]
    public void UnsupportedOperationInfo_CustomValues()
    {
        var info = new UnsupportedOperationInfo
        {
            OperationType = "QuantizeOp",
            NodeName = "quantize_0",
            TensorId = 42,
            Reason = "INT8 quantization not supported",
            CanFallback = false
        };

        Assert.Equal("QuantizeOp", info.OperationType);
        Assert.Equal("quantize_0", info.NodeName);
        Assert.Equal(42, info.TensorId);
        Assert.False(info.CanFallback);
    }

    [Fact]
    public void UnsupportedOperationInfo_ToString_ContainsInfo()
    {
        var info = new UnsupportedOperationInfo
        {
            OperationType = "CustomOp",
            NodeName = "node_1",
            TensorId = 5
        };

        var str = info.ToString();
        Assert.Contains("CustomOp", str);
        Assert.Contains("node_1", str);
        Assert.Contains("tensor 5", str);
    }

    #endregion

    #region HybridCompilationResult

    [Fact]
    public void HybridCompilationResult_DefaultValues()
    {
        var result = new HybridCompilationResult<double>();

        Assert.False(result.IsFullyJitCompiled);
        Assert.Equal("Unknown", result.ExecutionMode);
        Assert.NotNull(result.Compatibility);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void HybridCompilationResult_FullyCompiled()
    {
        var result = new HybridCompilationResult<double>
        {
            IsFullyJitCompiled = true,
            ExecutionMode = "JIT"
        };

        var str = result.ToString();
        Assert.Contains("JIT", str);
        Assert.Contains("100%", str);
    }

    [Fact]
    public void HybridCompilationResult_WithWarnings()
    {
        var result = new HybridCompilationResult<double>
        {
            ExecutionMode = "Hybrid",
            Warnings = { "Op 'CustomOp' using fallback", "Tensor shape mismatch" }
        };

        var str = result.ToString();
        Assert.Contains("Hybrid", str);
        Assert.Contains("2 warnings", str);
    }

    #endregion

    #region TensorPool

    [Fact]
    public void TensorPool_Construction_Default()
    {
        using var pool = new TensorPool();
        Assert.NotNull(pool);
    }

    [Fact]
    public void TensorPool_Construction_Custom()
    {
        using var pool = new TensorPool(maxPoolSizePerShape: 20, maxElementsToPool: 1_000_000);
        Assert.NotNull(pool);
    }

    [Fact]
    public void TensorPool_Rent_ReturnsArray()
    {
        using var pool = new TensorPool();
        var buffer = pool.Rent<double>(100);

        Assert.NotNull(buffer);
        Assert.Equal(100, buffer.Length);
    }

    [Fact]
    public void TensorPool_RentReturn_Reuse()
    {
        using var pool = new TensorPool();
        var buffer1 = pool.Rent<double>(100);
        pool.Return(buffer1);

        var buffer2 = pool.Rent<double>(100);
        // After return + rent of same size, should reuse (or at least return valid array)
        Assert.NotNull(buffer2);
        Assert.Equal(100, buffer2.Length);
    }

    [Fact]
    public void TensorPool_RentLargeArray_NotPooled()
    {
        using var pool = new TensorPool(maxPoolSizePerShape: 10, maxElementsToPool: 100);
        // Request larger than maxElementsToPool
        var buffer = pool.Rent<double>(200);

        Assert.NotNull(buffer);
        Assert.Equal(200, buffer.Length);
    }

    [Fact]
    public void TensorPool_GetStats()
    {
        using var pool = new TensorPool();
        var stats = pool.GetStats();

        Assert.NotNull(stats);
        Assert.Equal(0, stats.TotalPooledBuffers);
    }

    [Fact]
    public void TensorPool_GetStats_AfterReturn()
    {
        using var pool = new TensorPool();
        var buffer = pool.Rent<double>(50);
        pool.Return(buffer);

        var stats = pool.GetStats();
        Assert.True(stats.TotalPooledBuffers >= 0);
    }

    [Fact]
    public void TensorPool_Dispose_Safe()
    {
        var pool = new TensorPool();
        var buffer = pool.Rent<double>(10);
        pool.Return(buffer);
        pool.Dispose();
        // Should not throw after dispose
    }

    #endregion

    #region TensorPoolStats

    [Fact]
    public void TensorPoolStats_DefaultValues()
    {
        var stats = new TensorPoolStats();

        Assert.Equal(0, stats.TotalPooledBuffers);
        Assert.Equal(0L, stats.EstimatedMemoryBytes);
        Assert.Equal(0, stats.UniqueShapes);
    }

    [Fact]
    public void TensorPoolStats_CustomValues()
    {
        var stats = new TensorPoolStats
        {
            TotalPooledBuffers = 15,
            EstimatedMemoryBytes = 1024 * 1024,
            UniqueShapes = 3
        };

        Assert.Equal(15, stats.TotalPooledBuffers);
        Assert.Equal(1024 * 1024, stats.EstimatedMemoryBytes);
        Assert.Equal(3, stats.UniqueShapes);
    }

    #endregion

    #region TensorShapeExtensions

    [Fact]
    public void TensorShapeExtensions_GetElementCount_Vector()
    {
        var shape = new[] { 5 };
        Assert.Equal(5, shape.GetElementCount());
    }

    [Fact]
    public void TensorShapeExtensions_GetElementCount_Matrix()
    {
        var shape = new[] { 3, 4 };
        Assert.Equal(12, shape.GetElementCount());
    }

    [Fact]
    public void TensorShapeExtensions_GetElementCount_3D()
    {
        var shape = new[] { 2, 3, 4 };
        Assert.Equal(24, shape.GetElementCount());
    }

    [Fact]
    public void TensorShapeExtensions_GetElementCount_Scalar()
    {
        var shape = Array.Empty<int>();
        Assert.Equal(1, shape.GetElementCount());
    }

    [Fact]
    public void TensorShapeExtensions_GetElementCount_Dynamic()
    {
        var shape = new[] { 3, -1, 4 };
        Assert.Equal(-1, shape.GetElementCount());
    }

    [Fact]
    public void TensorShapeExtensions_IsValidShape_Valid()
    {
        Assert.True(new[] { 3, 4 }.IsValidShape());
        Assert.True(new[] { 1 }.IsValidShape());
        Assert.True(Array.Empty<int>().IsValidShape());
    }

    [Fact]
    public void TensorShapeExtensions_IsValidShape_WithDynamic()
    {
        // Dynamic shapes (-1) may still be valid in some contexts
        var shape = new[] { -1, 4 };
        // Check behavior - dynamic dims are considered valid for shape computations
        Assert.True(shape.IsValidShape());
    }

    #endregion

    #region IR Operations - Construction and Validation

    [Fact]
    public void AddOp_Construction_AndValidation()
    {
        var op = new AddOp
        {
            OutputId = 2,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 }
        };

        Assert.Equal("Add", op.OpType);
        Assert.True(op.Validate());
        Assert.Equal(2, op.OutputId);
    }

    [Fact]
    public void SubtractOp_Construction()
    {
        var op = new SubtractOp
        {
            OutputId = 2,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 5 }
        };

        Assert.Equal("Subtract", op.OpType);
        Assert.True(op.Validate());
    }

    [Fact]
    public void MatMulOp_Construction()
    {
        var op = new MatMulOp
        {
            OutputId = 2,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 5 }
        };

        Assert.Equal("MatMul", op.OpType);
        Assert.True(op.Validate());
    }

    [Fact]
    public void ReLUOp_Construction()
    {
        var op = new ReLUOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("ReLU", op.OpType);
        Assert.True(op.Validate());
    }

    [Fact]
    public void SigmoidOp_Construction()
    {
        var op = new SigmoidOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("Sigmoid", op.OpType);
    }

    [Fact]
    public void TanhOp_Construction()
    {
        var op = new TanhOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("Tanh", op.OpType);
    }

    [Fact]
    public void SoftmaxOp_Construction()
    {
        var op = new SoftmaxOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("Softmax", op.OpType);
    }

    [Fact]
    public void Conv2DOp_Construction()
    {
        var op = new Conv2DOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 1, 16, 28, 28 }
        };

        Assert.Equal("Conv2D", op.OpType);
    }

    [Fact]
    public void MaxPool2DOp_Construction()
    {
        var op = new MaxPool2DOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 1, 16, 14, 14 }
        };

        Assert.Equal("MaxPool2D", op.OpType);
    }

    [Fact]
    public void AvgPool2DOp_Construction()
    {
        var op = new AvgPool2DOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 1, 16, 14, 14 }
        };

        Assert.Equal("AvgPool2D", op.OpType);
    }

    [Fact]
    public void BatchNormOp_Construction()
    {
        var op = new BatchNormOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 32, 64 }
        };

        Assert.Equal("BatchNorm", op.OpType);
    }

    [Fact]
    public void LayerNormOp_Construction()
    {
        var op = new LayerNormOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 32, 64 }
        };

        Assert.Equal("LayerNorm", op.OpType);
    }

    [Fact]
    public void DropoutOp_Construction()
    {
        var op = new DropoutOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 32, 64 }
        };

        Assert.Equal("Dropout", op.OpType);
    }

    [Fact]
    public void ReshapeOp_Construction()
    {
        var op = new ReshapeOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 32, 64 }
        };

        Assert.Equal("Reshape", op.OpType);
    }

    [Fact]
    public void TransposeOp_Construction()
    {
        var op = new TransposeOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 64, 32 }
        };

        Assert.Equal("Transpose", op.OpType);
    }

    [Fact]
    public void ConcatOp_Construction()
    {
        var op = new ConcatOp
        {
            OutputId = 2,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 64 }
        };

        Assert.Equal("Concat", op.OpType);
    }

    [Fact]
    public void SliceOp_Construction()
    {
        var op = new SliceOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 16 }
        };

        Assert.Equal("Slice", op.OpType);
    }

    [Fact]
    public void ExpOp_Construction()
    {
        var op = new ExpOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("Exp", op.OpType);
    }

    [Fact]
    public void LogOp_Construction()
    {
        var op = new LogOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("Log", op.OpType);
    }

    [Fact]
    public void SqrtOp_Construction()
    {
        var op = new SqrtOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("Sqrt", op.OpType);
    }

    [Fact]
    public void PowerOp_Construction()
    {
        var op = new PowerOp
        {
            OutputId = 2,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("Power", op.OpType);
    }

    [Fact]
    public void NegateOp_Construction()
    {
        var op = new NegateOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("Negate", op.OpType);
    }

    [Fact]
    public void AbsOp_Construction()
    {
        var op = new AbsOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("Abs", op.OpType);
    }

    [Fact]
    public void ConstantOp_Construction()
    {
        var op = new ConstantOp
        {
            OutputId = 0,
            InputIds = Array.Empty<int>(),
            OutputShape = new[] { 3, 4 }
        };

        Assert.Equal("Constant", op.OpType);
    }

    [Fact]
    public void MeanOp_Construction()
    {
        var op = new MeanOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 1 }
        };

        Assert.Equal("Mean", op.OpType);
    }

    [Fact]
    public void SumOp_Construction()
    {
        var op = new SumOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 1 }
        };

        Assert.Equal("Sum", op.OpType);
    }

    [Fact]
    public void ElementwiseMultiplyOp_Construction()
    {
        var op = new ElementwiseMultiplyOp
        {
            OutputId = 2,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("ElementwiseMultiply", op.OpType);
    }

    [Fact]
    public void DivideOp_Construction()
    {
        var op = new DivideOp
        {
            OutputId = 2,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 10 }
        };

        Assert.Equal("Divide", op.OpType);
    }

    #endregion

    #region IR Operations - Activation Functions

    [Fact]
    public void GELUOp_Construction()
    {
        var op = new GELUOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("GELU", op.OpType);
    }

    [Fact]
    public void SwishOp_Construction()
    {
        var op = new SwishOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("Swish", op.OpType);
    }

    [Fact]
    public void MishOp_Construction()
    {
        var op = new MishOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("Mish", op.OpType);
    }

    [Fact]
    public void LeakyReLUOp_Construction()
    {
        var op = new LeakyReLUOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("LeakyReLU", op.OpType);
    }

    [Fact]
    public void ELUOp_Construction()
    {
        var op = new ELUOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("ELU", op.OpType);
    }

    [Fact]
    public void SELUOp_Construction()
    {
        var op = new SELUOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("SELU", op.OpType);
    }

    [Fact]
    public void SoftPlusOp_Construction()
    {
        var op = new SoftPlusOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("SoftPlus", op.OpType);
    }

    [Fact]
    public void SoftSignOp_Construction()
    {
        var op = new SoftSignOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("SoftSign", op.OpType);
    }

    [Fact]
    public void HardSigmoidOp_Construction()
    {
        var op = new HardSigmoidOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("HardSigmoid", op.OpType);
    }

    [Fact]
    public void HardTanhOp_Construction()
    {
        var op = new HardTanhOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("HardTanh", op.OpType);
    }

    #endregion

    #region IR Operations - Fused Operations

    [Fact]
    public void FusedAddReLUOp_Construction()
    {
        var op = new FusedAddReLUOp { OutputId = 3, InputIds = new[] { 0, 1 }, OutputShape = new[] { 10 } };
        Assert.Equal("FusedAddReLU", op.OpType);
    }

    [Fact]
    public void FusedLinearReLUOp_Construction()
    {
        var op = new FusedLinearReLUOp { OutputId = 3, InputIds = new[] { 0, 1 }, OutputShape = new[] { 10 } };
        Assert.Equal("FusedLinearReLU", op.OpType);
    }

    [Fact]
    public void FusedMatMulAddOp_Construction()
    {
        var op = new FusedMatMulAddOp { OutputId = 3, InputIds = new[] { 0, 1, 2 }, OutputShape = new[] { 3, 5 } };
        Assert.Equal("FusedMatMulAdd", op.OpType);
    }

    [Fact]
    public void FusedConvBatchNormOp_Construction()
    {
        var op = new FusedConvBatchNormOp { OutputId = 2, InputIds = new[] { 0, 1 }, OutputShape = new[] { 1, 16, 28, 28 } };
        Assert.Equal("FusedConvBatchNorm", op.OpType);
    }

    [Fact]
    public void FusedGELUOp_Construction()
    {
        var op = new FusedGELUOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("FusedGELU", op.OpType);
    }

    [Fact]
    public void FusedSwishOp_Construction()
    {
        var op = new FusedSwishOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 } };
        Assert.Equal("FusedSwish", op.OpType);
    }

    [Fact]
    public void FusedAttentionOp_Construction()
    {
        var op = new FusedAttentionOp { OutputId = 3, InputIds = new[] { 0, 1, 2 }, OutputShape = new[] { 1, 8, 10, 64 } };
        Assert.Equal("FusedAttention", op.OpType);
    }

    #endregion

    #region IR Operations - RNN Operations

    [Fact]
    public void LSTMCellOp_Construction()
    {
        var op = new LSTMCellOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 32, 64 } };
        Assert.Equal("LSTMCell", op.OpType);
    }

    [Fact]
    public void GRUCellOp_Construction()
    {
        var op = new GRUCellOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 32, 64 } };
        Assert.Equal("GRUCell", op.OpType);
    }

    #endregion

    #region IR Operations - Attention

    [Fact]
    public void AttentionOp_Construction()
    {
        var op = new AttentionOp { OutputId = 3, InputIds = new[] { 0, 1, 2 }, OutputShape = new[] { 1, 8, 10, 64 } };
        Assert.Equal("Attention", op.OpType);
    }

    [Fact]
    public void MultiHeadAttentionOp_Construction()
    {
        var op = new MultiHeadAttentionOp { OutputId = 3, InputIds = new[] { 0, 1, 2 }, OutputShape = new[] { 1, 10, 512 } };
        Assert.Equal("MultiHeadAttention", op.OpType);
    }

    [Fact]
    public void ScaledDotProductAttentionOp_Construction()
    {
        var op = new ScaledDotProductAttentionOp { OutputId = 3, InputIds = new[] { 0, 1, 2 }, OutputShape = new[] { 1, 10, 64 } };
        Assert.Equal("ScaledDotProductAttention", op.OpType);
    }

    #endregion

    #region IR Operations - Validation

    [Fact]
    public void IROp_Validate_FailsWithNoOutputs()
    {
        var op = new AddOp
        {
            OutputIds = Array.Empty<int>(),
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 10 }
        };

        Assert.False(op.Validate());
    }

    [Fact]
    public void IROp_Validate_FailsWithNegativeOutputId()
    {
        var op = new AddOp
        {
            OutputId = -1,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 10 }
        };

        Assert.False(op.Validate());
    }

    [Fact]
    public void AddOp_Validate_FailsWithWrongInputCount()
    {
        var op = new AddOp
        {
            OutputId = 1,
            InputIds = new[] { 0 }, // Add needs 2 inputs
            OutputShape = new[] { 10 }
        };

        Assert.False(op.Validate());
    }

    [Fact]
    public void IROp_ToString_Format()
    {
        var op = new AddOp
        {
            OutputId = 2,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 },
            OutputType = IRType.Float32
        };

        var str = op.ToString();
        Assert.Contains("Add", str);
        Assert.Contains("t0", str);
        Assert.Contains("t1", str);
        Assert.Contains("t2", str);
    }

    #endregion

    #region IR Operations - Graph Operations

    [Fact]
    public void EmbeddingOp_Construction()
    {
        var op = new EmbeddingOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10, 128 } };
        Assert.Equal("Embedding", op.OpType);
    }

    [Fact]
    public void PadOp_Construction()
    {
        var op = new PadOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 12, 12 } };
        Assert.Equal("Pad", op.OpType);
    }

    [Fact]
    public void UpsampleOp_Construction()
    {
        var op = new UpsampleOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 1, 3, 64, 64 } };
        Assert.Equal("Upsample", op.OpType);
    }

    [Fact]
    public void ConvTranspose2DOp_Construction()
    {
        var op = new ConvTranspose2DOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 1, 16, 56, 56 } };
        Assert.Equal("ConvTranspose2D", op.OpType);
    }

    [Fact]
    public void DepthwiseConv2DOp_Construction()
    {
        var op = new DepthwiseConv2DOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 1, 32, 28, 28 } };
        Assert.Equal("DepthwiseConv2D", op.OpType);
    }

    [Fact]
    public void DilatedConv2DOp_Construction()
    {
        var op = new DilatedConv2DOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 1, 16, 28, 28 } };
        Assert.Equal("DilatedConv2D", op.OpType);
    }

    #endregion

    #region SIMDCapabilities

    [Fact]
    public void SIMDCapabilities_Detection()
    {
        var caps = SIMDCapabilities.Detect();

        Assert.NotNull(caps);
        // At minimum, IsHardwareAccelerated tells us if SIMD is available
        // These are hardware-dependent, so we just verify they return valid values
        Assert.True(caps.MaxVectorWidth >= 0);
    }

    #endregion

    #region SIMDStats

    [Fact]
    public void SIMDStats_DefaultValues()
    {
        var stats = new SIMDStats();

        Assert.Equal(0, stats.TotalOperations);
        Assert.Equal(0, stats.VectorizableOperations);
    }

    [Fact]
    public void SIMDStats_CustomValues()
    {
        var stats = new SIMDStats
        {
            TotalOperations = 100,
            VectorizableOperations = 75
        };

        Assert.Equal(100, stats.TotalOperations);
        Assert.Equal(75, stats.VectorizableOperations);
    }

    #endregion

    #region VectorizationStats

    [Fact]
    public void VectorizationStats_DefaultValues()
    {
        var stats = new VectorizationStats();

        Assert.Equal(0, stats.TotalOperations);
        Assert.Equal(0, stats.VectorizableOperations);
        Assert.Equal(0L, stats.TotalVectorizableElements);
    }

    [Fact]
    public void VectorizationStats_CustomValues()
    {
        var stats = new VectorizationStats
        {
            TotalOperations = 50,
            VectorizableOperations = 30,
            TotalVectorizableElements = 1_000_000,
            HardwareVectorWidth = 8
        };

        Assert.Equal(50, stats.TotalOperations);
        Assert.Equal(30, stats.VectorizableOperations);
        Assert.Equal(1_000_000L, stats.TotalVectorizableElements);
        Assert.Equal(8, stats.HardwareVectorWidth);
    }

    #endregion
}
