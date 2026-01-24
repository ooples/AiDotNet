using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Xunit;
using AiDotNet.JitCompiler;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using AiDotNet.JitCompiler.CodeGen;

namespace AiDotNet.Tests.IntegrationTests.JitCompiler;

/// <summary>
/// Comprehensive integration tests for the JitCompiler module.
/// Tests IR types, graph structures, operations, SIMD capabilities, and compilation stats.
/// </summary>
public class JitCompilerIntegrationTests
{
    #region IRType Enum Tests

    [Fact]
    public void IRType_ContainsExpectedValues()
    {
        var values = (IRType[])Enum.GetValues(typeof(IRType));

        Assert.Contains(IRType.Float32, values);
        Assert.Contains(IRType.Float64, values);
        Assert.Contains(IRType.Int32, values);
        Assert.Contains(IRType.Int64, values);
        Assert.Contains(IRType.Byte, values);
        Assert.Contains(IRType.SByte, values);
        Assert.Contains(IRType.Int16, values);
        Assert.Contains(IRType.UInt16, values);
        Assert.Contains(IRType.UInt32, values);
        Assert.Contains(IRType.UInt64, values);
        Assert.Contains(IRType.Decimal, values);
        Assert.Contains(IRType.Half, values);
        Assert.Contains(IRType.Complex, values);
    }

    [Fact]
    public void IRType_HasExpectedCount()
    {
        var values = (IRType[])Enum.GetValues(typeof(IRType));
        Assert.Equal(13, values.Length);
    }

    [Theory]
    [InlineData(IRType.Float32)]
    [InlineData(IRType.Float64)]
    [InlineData(IRType.Int32)]
    [InlineData(IRType.Int64)]
    [InlineData(IRType.Byte)]
    [InlineData(IRType.Decimal)]
    [InlineData(IRType.Half)]
    [InlineData(IRType.Complex)]
    public void IRType_IsDefined(IRType irType)
    {
        Assert.True(Enum.IsDefined(typeof(IRType), irType));
    }

    #endregion

    #region IRTypeExtensions Tests

    [Theory]
    [InlineData(typeof(float), IRType.Float32)]
    [InlineData(typeof(double), IRType.Float64)]
    [InlineData(typeof(int), IRType.Int32)]
    [InlineData(typeof(long), IRType.Int64)]
    [InlineData(typeof(byte), IRType.Byte)]
    [InlineData(typeof(sbyte), IRType.SByte)]
    [InlineData(typeof(short), IRType.Int16)]
    [InlineData(typeof(ushort), IRType.UInt16)]
    [InlineData(typeof(uint), IRType.UInt32)]
    [InlineData(typeof(ulong), IRType.UInt64)]
    [InlineData(typeof(decimal), IRType.Decimal)]
    public void IRTypeExtensions_FromSystemType_ReturnsCorrectIRType(Type systemType, IRType expected)
    {
        var result = IRTypeExtensions.FromSystemType(systemType);
        Assert.Equal(expected, result);
    }

    [Theory]
    [InlineData(IRType.Float32, typeof(float))]
    [InlineData(IRType.Float64, typeof(double))]
    [InlineData(IRType.Int32, typeof(int))]
    [InlineData(IRType.Int64, typeof(long))]
    [InlineData(IRType.Byte, typeof(byte))]
    [InlineData(IRType.SByte, typeof(sbyte))]
    [InlineData(IRType.Int16, typeof(short))]
    [InlineData(IRType.UInt16, typeof(ushort))]
    [InlineData(IRType.UInt32, typeof(uint))]
    [InlineData(IRType.UInt64, typeof(ulong))]
    [InlineData(IRType.Decimal, typeof(decimal))]
    public void IRTypeExtensions_ToSystemType_ReturnsCorrectType(IRType irType, Type expected)
    {
        var result = irType.ToSystemType();
        Assert.Equal(expected, result);
    }

    [Fact]
    public void IRTypeExtensions_FromSystemType_UnsupportedType_ThrowsException()
    {
        Assert.Throws<NotSupportedException>(() => IRTypeExtensions.FromSystemType(typeof(string)));
    }

    [Fact]
    public void IRTypeExtensions_RoundTrip_PreservesType()
    {
        var irTypes = (IRType[])Enum.GetValues(typeof(IRType));
        foreach (var irType in irTypes)
        {
            try
            {
                var systemType = irType.ToSystemType();
                var backToIRType = IRTypeExtensions.FromSystemType(systemType);
                Assert.Equal(irType, backToIRType);
            }
            catch (NotSupportedException)
            {
                // Some types may not be fully supported for round-trip
            }
        }
    }

    #endregion

    #region CacheStats Tests

    [Fact]
    public void CacheStats_DefaultConstructor_InitializesProperties()
    {
        var stats = new CacheStats();

        Assert.Equal(0, stats.CachedGraphCount);
        Assert.Equal(0, stats.EstimatedMemoryBytes);
    }

    [Fact]
    public void CacheStats_CanSetCachedGraphCount()
    {
        var stats = new CacheStats
        {
            CachedGraphCount = 42
        };

        Assert.Equal(42, stats.CachedGraphCount);
    }

    [Fact]
    public void CacheStats_CanSetEstimatedMemoryBytes()
    {
        var stats = new CacheStats
        {
            EstimatedMemoryBytes = 1024 * 1024 // 1 MB
        };

        Assert.Equal(1024 * 1024, stats.EstimatedMemoryBytes);
    }

    [Fact]
    public void CacheStats_ToString_ReturnsFormattedString()
    {
        var stats = new CacheStats
        {
            CachedGraphCount = 5,
            EstimatedMemoryBytes = 2048
        };

        var result = stats.ToString();

        Assert.Contains("Cache Stats", result);
        Assert.Contains("Cached graphs: 5", result);
        Assert.Contains("KB", result);
    }

    #endregion

    #region CompilationStats Tests

    [Fact]
    public void CompilationStats_DefaultConstructor_InitializesProperties()
    {
        var stats = new CompilationStats();

        Assert.Equal(0, stats.OriginalOperationCount);
        Assert.Equal(0, stats.OptimizedOperationCount);
        Assert.NotNull(stats.OptimizationsApplied);
        Assert.Empty(stats.OptimizationsApplied);
        Assert.Equal(TimeSpan.Zero, stats.CompilationTime);
        Assert.False(stats.CacheHit);
    }

    [Fact]
    public void CompilationStats_OperationsEliminated_ReturnsCorrectValue()
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = 100,
            OptimizedOperationCount = 75
        };

        Assert.Equal(25, stats.OperationsEliminated);
    }

    [Fact]
    public void CompilationStats_OptimizationPercentage_ReturnsCorrectValue()
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = 100,
            OptimizedOperationCount = 80
        };

        Assert.Equal(20.0, stats.OptimizationPercentage);
    }

    [Fact]
    public void CompilationStats_OptimizationPercentage_ZeroOriginal_ReturnsZero()
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = 0,
            OptimizedOperationCount = 0
        };

        Assert.Equal(0.0, stats.OptimizationPercentage);
    }

    [Fact]
    public void CompilationStats_CanSetCompilationTime()
    {
        var stats = new CompilationStats
        {
            CompilationTime = TimeSpan.FromMilliseconds(125)
        };

        Assert.Equal(125.0, stats.CompilationTime.TotalMilliseconds);
    }

    [Fact]
    public void CompilationStats_CanAddOptimizations()
    {
        var stats = new CompilationStats();
        stats.OptimizationsApplied.Add("Constant Folding");
        stats.OptimizationsApplied.Add("Dead Code Elimination");

        Assert.Equal(2, stats.OptimizationsApplied.Count);
        Assert.Contains("Constant Folding", stats.OptimizationsApplied);
    }

    [Fact]
    public void CompilationStats_ToString_ReturnsFormattedString()
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = 100,
            OptimizedOperationCount = 75,
            CompilationTime = TimeSpan.FromMilliseconds(50),
            CacheHit = false
        };
        stats.OptimizationsApplied.Add("Constant Folding");

        var result = stats.ToString();

        Assert.Contains("Compilation Stats", result);
        Assert.Contains("Original operations: 100", result);
        Assert.Contains("Optimized operations: 75", result);
        Assert.Contains("Operations eliminated: 25", result);
        Assert.Contains("Constant Folding", result);
        Assert.Contains("Cache hit: False", result);
    }

    #endregion

    #region JitCompatibilityResult Tests

    [Fact]
    public void JitCompatibilityResult_DefaultConstructor_InitializesProperties()
    {
        var result = new JitCompatibilityResult();

        Assert.False(result.IsFullySupported);
        Assert.NotNull(result.SupportedOperations);
        Assert.NotNull(result.UnsupportedOperations);
        Assert.Empty(result.SupportedOperations);
        Assert.Empty(result.UnsupportedOperations);
        Assert.False(result.CanUseHybridMode);
    }

    [Fact]
    public void JitCompatibilityResult_SupportedPercentage_AllSupported_Returns100()
    {
        var result = new JitCompatibilityResult();
        result.SupportedOperations.Add("Add");
        result.SupportedOperations.Add("MatMul");
        result.SupportedOperations.Add("ReLU");

        Assert.Equal(100.0, result.SupportedPercentage);
    }

    [Fact]
    public void JitCompatibilityResult_SupportedPercentage_Mixed_ReturnsCorrectValue()
    {
        var result = new JitCompatibilityResult();
        result.SupportedOperations.Add("Add");
        result.SupportedOperations.Add("MatMul");
        result.UnsupportedOperations.Add(new UnsupportedOperationInfo { OperationType = "CustomOp" });

        // 2 supported, 1 unsupported = 66.67%
        Assert.True(result.SupportedPercentage > 66.0 && result.SupportedPercentage < 67.0);
    }

    [Fact]
    public void JitCompatibilityResult_SupportedPercentage_Empty_Returns100()
    {
        var result = new JitCompatibilityResult();

        Assert.Equal(100.0, result.SupportedPercentage);
    }

    [Fact]
    public void JitCompatibilityResult_ToString_FullySupported_ReturnsCorrectMessage()
    {
        var result = new JitCompatibilityResult
        {
            IsFullySupported = true
        };
        result.SupportedOperations.Add("Add");
        result.SupportedOperations.Add("MatMul");

        var str = result.ToString();

        Assert.Contains("Fully JIT compatible", str);
        Assert.Contains("2 operations", str);
    }

    [Fact]
    public void JitCompatibilityResult_ToString_PartialSupport_ReturnsCorrectMessage()
    {
        var result = new JitCompatibilityResult
        {
            IsFullySupported = false,
            CanUseHybridMode = true
        };
        result.SupportedOperations.Add("Add");
        result.UnsupportedOperations.Add(new UnsupportedOperationInfo { OperationType = "CustomOp" });

        var str = result.ToString();

        Assert.Contains("Partial JIT support", str);
        Assert.Contains("Hybrid mode: available", str);
    }

    #endregion

    #region UnsupportedOperationInfo Tests

    [Fact]
    public void UnsupportedOperationInfo_DefaultConstructor_InitializesProperties()
    {
        var info = new UnsupportedOperationInfo();

        Assert.Equal("", info.OperationType);
        Assert.Null(info.NodeName);
        Assert.Equal(0, info.TensorId);
        Assert.Contains("not implemented", info.Reason);
        Assert.True(info.CanFallback);
    }

    [Fact]
    public void UnsupportedOperationInfo_CanSetOperationType()
    {
        var info = new UnsupportedOperationInfo
        {
            OperationType = "CustomAttentionOp"
        };

        Assert.Equal("CustomAttentionOp", info.OperationType);
    }

    [Fact]
    public void UnsupportedOperationInfo_CanSetNodeName()
    {
        var info = new UnsupportedOperationInfo
        {
            NodeName = "attention_layer_1"
        };

        Assert.Equal("attention_layer_1", info.NodeName);
    }

    [Fact]
    public void UnsupportedOperationInfo_CanSetTensorId()
    {
        var info = new UnsupportedOperationInfo
        {
            TensorId = 42
        };

        Assert.Equal(42, info.TensorId);
    }

    [Fact]
    public void UnsupportedOperationInfo_CanSetReason()
    {
        var info = new UnsupportedOperationInfo
        {
            Reason = "Complex number operations not supported"
        };

        Assert.Equal("Complex number operations not supported", info.Reason);
    }

    [Fact]
    public void UnsupportedOperationInfo_CanSetCanFallback()
    {
        var info = new UnsupportedOperationInfo
        {
            CanFallback = false
        };

        Assert.False(info.CanFallback);
    }

    [Fact]
    public void UnsupportedOperationInfo_ToString_WithoutNodeName_ReturnsCorrectFormat()
    {
        var info = new UnsupportedOperationInfo
        {
            OperationType = "CustomOp",
            TensorId = 5,
            Reason = "Not implemented"
        };

        var str = info.ToString();

        Assert.Contains("Unsupported: CustomOp", str);
        Assert.Contains("tensor 5", str);
        Assert.Contains("Not implemented", str);
    }

    [Fact]
    public void UnsupportedOperationInfo_ToString_WithNodeName_IncludesNodeName()
    {
        var info = new UnsupportedOperationInfo
        {
            OperationType = "CustomOp",
            NodeName = "my_layer",
            TensorId = 5,
            Reason = "Not implemented"
        };

        var str = info.ToString();

        Assert.Contains("(my_layer)", str);
    }

    #endregion

    #region IROp Base Class Tests

    [Fact]
    public void IROp_DefaultProperties_HaveCorrectDefaults()
    {
        var op = new AddOp();

        Assert.NotNull(op.OutputIds);
        Assert.Empty(op.OutputIds);
        Assert.NotNull(op.InputIds);
        Assert.Empty(op.InputIds);
        Assert.NotNull(op.OutputShape);
        Assert.Empty(op.OutputShape);
    }

    [Fact]
    public void IROp_OutputId_SingleValue_ReturnsFirstElement()
    {
        var op = new AddOp
        {
            OutputIds = new[] { 5 }
        };

        Assert.Equal(5, op.OutputId);
    }

    [Fact]
    public void IROp_OutputId_Empty_ReturnsNegativeOne()
    {
        var op = new AddOp
        {
            OutputIds = Array.Empty<int>()
        };

        Assert.Equal(-1, op.OutputId);
    }

    [Fact]
    public void IROp_OutputId_Set_UpdatesOutputIds()
    {
        var op = new AddOp();
        op.OutputId = 10;

        Assert.Single(op.OutputIds);
        Assert.Equal(10, op.OutputIds[0]);
    }

    [Fact]
    public void IROp_OpType_ReturnsClassNameWithoutOpSuffix()
    {
        var addOp = new AddOp();
        var constantOp = new ConstantOp();

        Assert.Equal("Add", addOp.OpType);
        Assert.Equal("Constant", constantOp.OpType);
    }

    [Fact]
    public void IROp_Validate_NoOutputIds_ReturnsFalse()
    {
        var op = new AddOp
        {
            OutputIds = Array.Empty<int>(),
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 }
        };

        Assert.False(op.Validate());
    }

    [Fact]
    public void IROp_Validate_NegativeOutputId_ReturnsFalse()
    {
        var op = new AddOp
        {
            OutputIds = new[] { -1 },
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 }
        };

        Assert.False(op.Validate());
    }

    #endregion

    #region AddOp Tests

    [Fact]
    public void AddOp_Validate_ValidConfiguration_ReturnsTrue()
    {
        var op = new AddOp
        {
            OutputIds = new[] { 2 },
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 },
            OutputType = IRType.Float32
        };

        Assert.True(op.Validate());
    }

    [Fact]
    public void AddOp_Validate_WrongInputCount_ReturnsFalse()
    {
        var op = new AddOp
        {
            OutputIds = new[] { 2 },
            InputIds = new[] { 0 }, // Should have 2 inputs
            OutputShape = new[] { 3, 4 },
            OutputType = IRType.Float32
        };

        Assert.False(op.Validate());
    }

    [Fact]
    public void AddOp_OpType_ReturnsAdd()
    {
        var op = new AddOp();
        Assert.Equal("Add", op.OpType);
    }

    [Fact]
    public void AddOp_ToString_ReturnsCorrectFormat()
    {
        var op = new AddOp
        {
            OutputIds = new[] { 2 },
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 },
            OutputType = IRType.Float32
        };

        var str = op.ToString();

        Assert.Contains("t2", str);
        Assert.Contains("Add", str);
        Assert.Contains("t0", str);
        Assert.Contains("t1", str);
        Assert.Contains("Float32", str);
    }

    #endregion

    #region ConstantOp Tests

    [Fact]
    public void ConstantOp_DefaultConstructor_InitializesProperties()
    {
        var op = new ConstantOp();

        Assert.NotNull(op.Values);
        Assert.Empty(op.Values);
    }

    [Fact]
    public void ConstantOp_CanSetValues()
    {
        var op = new ConstantOp
        {
            Values = new[] { 1.0, 2.0, 3.0 }
        };

        Assert.Equal(3, op.Values.Length);
        Assert.Equal(1.0, op.Values[0]);
        Assert.Equal(2.0, op.Values[1]);
        Assert.Equal(3.0, op.Values[2]);
    }

    [Fact]
    public void ConstantOp_IsScalar_ScalarShape_ReturnsTrue()
    {
        var op = new ConstantOp
        {
            OutputShape = Array.Empty<int>(), // Scalar
            Values = new[] { 5.0 }
        };

        Assert.True(op.IsScalar);
    }

    [Fact]
    public void ConstantOp_IsScalar_SingleElementShape_ReturnsTrue()
    {
        var op = new ConstantOp
        {
            OutputShape = new[] { 1 },
            Values = new[] { 5.0 }
        };

        Assert.True(op.IsScalar);
    }

    [Fact]
    public void ConstantOp_IsScalar_VectorShape_ReturnsFalse()
    {
        var op = new ConstantOp
        {
            OutputShape = new[] { 3 },
            Values = new[] { 1.0, 2.0, 3.0 }
        };

        Assert.False(op.IsScalar);
    }

    [Fact]
    public void ConstantOp_Validate_ValidScalar_ReturnsTrue()
    {
        var op = new ConstantOp
        {
            OutputIds = new[] { 0 },
            InputIds = Array.Empty<int>(),
            OutputShape = new[] { 1 },
            OutputType = IRType.Float64,
            Values = new[] { 5.0 }
        };

        Assert.True(op.Validate());
    }

    [Fact]
    public void ConstantOp_Validate_ValidVector_ReturnsTrue()
    {
        var op = new ConstantOp
        {
            OutputIds = new[] { 0 },
            InputIds = Array.Empty<int>(),
            OutputShape = new[] { 3 },
            OutputType = IRType.Float64,
            Values = new[] { 1.0, 2.0, 3.0 }
        };

        Assert.True(op.Validate());
    }

    [Fact]
    public void ConstantOp_Validate_WrongValueCount_ReturnsFalse()
    {
        var op = new ConstantOp
        {
            OutputIds = new[] { 0 },
            InputIds = Array.Empty<int>(),
            OutputShape = new[] { 3 },
            OutputType = IRType.Float64,
            Values = new[] { 1.0, 2.0 } // Should have 3 values
        };

        Assert.False(op.Validate());
    }

    [Fact]
    public void ConstantOp_Validate_HasInputs_ReturnsFalse()
    {
        var op = new ConstantOp
        {
            OutputIds = new[] { 0 },
            InputIds = new[] { 1 }, // Constants shouldn't have inputs
            OutputShape = new[] { 1 },
            OutputType = IRType.Float64,
            Values = new[] { 5.0 }
        };

        Assert.False(op.Validate());
    }

    [Fact]
    public void ConstantOp_ToString_SmallValues_ShowsAllValues()
    {
        var op = new ConstantOp
        {
            OutputIds = new[] { 0 },
            OutputShape = new[] { 3 },
            OutputType = IRType.Float64,
            Values = new[] { 1.0, 2.0, 3.0 }
        };

        var str = op.ToString();

        Assert.Contains("Constant", str);
        Assert.Contains("1", str);
        Assert.Contains("2", str);
        Assert.Contains("3", str);
    }

    [Fact]
    public void ConstantOp_ToString_LargeValues_ShowsTruncated()
    {
        var values = Enumerable.Range(0, 100).Select(i => (double)i).ToArray();
        var op = new ConstantOp
        {
            OutputIds = new[] { 0 },
            OutputShape = new[] { 100 },
            OutputType = IRType.Float64,
            Values = values
        };

        var str = op.ToString();

        Assert.Contains("100 elements", str);
    }

    #endregion

    #region IRGraph Tests

    [Fact]
    public void IRGraph_DefaultConstructor_InitializesProperties()
    {
        var graph = new IRGraph();

        Assert.NotNull(graph.Operations);
        Assert.NotNull(graph.TensorShapes);
        Assert.NotNull(graph.InputIds);
        Assert.NotNull(graph.OutputIds);
        Assert.NotNull(graph.Metadata);
        Assert.Empty(graph.Operations);
        Assert.Empty(graph.TensorShapes);
        Assert.Empty(graph.InputIds);
        Assert.Empty(graph.OutputIds);
        Assert.Empty(graph.Metadata);
    }

    [Fact]
    public void IRGraph_CanAddOperations()
    {
        var graph = new IRGraph();
        graph.Operations.Add(new AddOp
        {
            OutputIds = new[] { 2 },
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 },
            OutputType = IRType.Float32
        });

        Assert.Single(graph.Operations);
    }

    [Fact]
    public void IRGraph_CanSetInputIds()
    {
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1 }
        };

        Assert.Equal(2, graph.InputIds.Count);
        Assert.Contains(0, graph.InputIds);
        Assert.Contains(1, graph.InputIds);
    }

    [Fact]
    public void IRGraph_CanSetOutputIds()
    {
        var graph = new IRGraph
        {
            OutputIds = new List<int> { 5 }
        };

        Assert.Single(graph.OutputIds);
        Assert.Contains(5, graph.OutputIds);
    }

    [Fact]
    public void IRGraph_CanSetTensorShapes()
    {
        var graph = new IRGraph();
        graph.TensorShapes[0] = new[] { 32, 784 };
        graph.TensorShapes[1] = new[] { 784, 128 };

        Assert.Equal(2, graph.TensorShapes.Count);
        Assert.Equal(new[] { 32, 784 }, graph.TensorShapes[0]);
    }

    [Fact]
    public void IRGraph_CanAddMetadata()
    {
        var graph = new IRGraph();
        graph.Metadata["name"] = "my_model";
        graph.Metadata["version"] = 1;

        Assert.Equal("my_model", graph.Metadata["name"]);
        Assert.Equal(1, graph.Metadata["version"]);
    }

    [Fact]
    public void IRGraph_Validate_EmptyGraph_ReturnsTrue()
    {
        var graph = new IRGraph();

        Assert.True(graph.Validate());
    }

    [Fact]
    public void IRGraph_Validate_ValidSimpleGraph_ReturnsTrue()
    {
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1 },
            OutputIds = new List<int> { 2 }
        };
        graph.TensorShapes[0] = new[] { 3, 4 };
        graph.TensorShapes[1] = new[] { 3, 4 };
        graph.Operations.Add(new AddOp
        {
            OutputIds = new[] { 2 },
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 },
            OutputType = IRType.Float32
        });

        Assert.True(graph.Validate());
    }

    [Fact]
    public void IRGraph_Validate_InputMissingShape_ReturnsFalse()
    {
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1 }, // No shapes defined
            OutputIds = new List<int> { 2 }
        };
        graph.Operations.Add(new AddOp
        {
            OutputIds = new[] { 2 },
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 },
            OutputType = IRType.Float32
        });

        Assert.False(graph.Validate());
    }

    [Fact]
    public void IRGraph_Validate_OutputNotProduced_ReturnsFalse()
    {
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0 },
            OutputIds = new List<int> { 5 } // Output 5 is never produced
        };
        graph.TensorShapes[0] = new[] { 3, 4 };

        Assert.False(graph.Validate());
    }

    [Fact]
    public void IRGraph_ComputeStructureHash_SameStructure_ReturnsSameHash()
    {
        var graph1 = CreateSimpleGraph();
        var graph2 = CreateSimpleGraph();

        Assert.Equal(graph1.ComputeStructureHash(), graph2.ComputeStructureHash());
    }

    [Fact]
    public void IRGraph_ComputeStructureHash_DifferentStructure_ReturnsDifferentHash()
    {
        var graph1 = CreateSimpleGraph();

        var graph2 = new IRGraph
        {
            InputIds = new List<int> { 0, 1 },
            OutputIds = new List<int> { 2 }
        };
        graph2.TensorShapes[0] = new[] { 5, 5 }; // Different shape
        graph2.TensorShapes[1] = new[] { 5, 5 };
        graph2.Operations.Add(new AddOp
        {
            OutputIds = new[] { 2 },
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 5, 5 },
            OutputType = IRType.Float32
        });

        Assert.NotEqual(graph1.ComputeStructureHash(), graph2.ComputeStructureHash());
    }

    [Fact]
    public void IRGraph_ToString_ReturnsFormattedString()
    {
        var graph = CreateSimpleGraph();

        var str = graph.ToString();

        Assert.Contains("IR Graph", str);
        Assert.Contains("Inputs", str);
        Assert.Contains("Operations", str);
        Assert.Contains("Outputs", str);
    }

    private static IRGraph CreateSimpleGraph()
    {
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1 },
            OutputIds = new List<int> { 2 }
        };
        graph.TensorShapes[0] = new[] { 3, 4 };
        graph.TensorShapes[1] = new[] { 3, 4 };
        graph.Operations.Add(new AddOp
        {
            OutputIds = new[] { 2 },
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 },
            OutputType = IRType.Float32
        });
        return graph;
    }

    #endregion

    #region SIMDCapabilities Tests

    [Fact]
    public void SIMDCapabilities_Detect_ReturnsNonNull()
    {
        var caps = SIMDCapabilities.Detect();

        Assert.NotNull(caps);
    }

    [Fact]
    public void SIMDCapabilities_DefaultProperties_AreInitialized()
    {
        var caps = new SIMDCapabilities();

        // Just verify the properties can be read without exception
        _ = caps.HasSSE;
        _ = caps.HasAVX;
        _ = caps.HasAVX2;
        _ = caps.HasAVX512;
        _ = caps.HasFMA;
        _ = caps.HasNEON;
        _ = caps.MaxVectorWidth;
    }

    [Fact]
    public void SIMDCapabilities_IsHardwareAccelerated_ReflectsCapabilities()
    {
        var caps = new SIMDCapabilities
        {
            HasSSE = true,
            HasNEON = false
        };

        Assert.True(caps.IsHardwareAccelerated);

        caps = new SIMDCapabilities
        {
            HasSSE = false,
            HasNEON = true
        };

        Assert.True(caps.IsHardwareAccelerated);

        caps = new SIMDCapabilities
        {
            HasSSE = false,
            HasNEON = false
        };

        Assert.False(caps.IsHardwareAccelerated);
    }

    [Theory]
    [InlineData(4, 4)]  // float size, 16/4 = 4 elements
    [InlineData(8, 2)]  // double size, 16/8 = 2 elements
    [InlineData(1, 16)] // byte size, 16/1 = 16 elements
    public void SIMDCapabilities_GetVectorCount_ReturnsCorrectValue(int typeSize, int expected)
    {
        var caps = new SIMDCapabilities
        {
            MaxVectorWidth = 16 // 128-bit SIMD
        };

        Assert.Equal(expected, caps.GetVectorCount(typeSize));
    }

    [Fact]
    public void SIMDCapabilities_GetVectorCount_ZeroTypeSize_ReturnsOne()
    {
        var caps = new SIMDCapabilities
        {
            MaxVectorWidth = 16
        };

        Assert.Equal(1, caps.GetVectorCount(0));
    }

    [Fact]
    public void SIMDCapabilities_GetVectorCount_ZeroVectorWidth_ReturnsOne()
    {
        var caps = new SIMDCapabilities
        {
            MaxVectorWidth = 0
        };

        Assert.Equal(1, caps.GetVectorCount(4));
    }

    [Fact]
    public void SIMDCapabilities_ToString_NoFeatures_ReturnsNotAvailable()
    {
        var caps = new SIMDCapabilities
        {
            HasSSE = false,
            HasAVX = false,
            HasNEON = false
        };

        Assert.Contains("Not available", caps.ToString());
    }

    [Fact]
    public void SIMDCapabilities_ToString_WithFeatures_ListsFeatures()
    {
        var caps = new SIMDCapabilities
        {
            HasSSE = true,
            HasAVX = true,
            MaxVectorWidth = 32
        };

        var str = caps.ToString();

        Assert.Contains("SSE", str);
        Assert.Contains("AVX", str);
        Assert.Contains("32 bytes", str);
    }

    #endregion

    #region SIMDStats Tests

    [Fact]
    public void SIMDStats_DefaultConstructor_InitializesProperties()
    {
        var stats = new SIMDStats();

        Assert.Equal(0, stats.TotalOperations);
        Assert.Equal(0, stats.VectorizableOperations);
        // VectorSize and HardwareAccelerated depend on runtime hardware
    }

    [Fact]
    public void SIMDStats_VectorizableRatio_ReturnsCorrectValue()
    {
        var stats = new SIMDStats
        {
            TotalOperations = 100,
            VectorizableOperations = 75
        };

        Assert.Equal(0.75, stats.VectorizableRatio);
    }

    [Fact]
    public void SIMDStats_VectorizableRatio_ZeroTotal_ReturnsZero()
    {
        var stats = new SIMDStats
        {
            TotalOperations = 0,
            VectorizableOperations = 0
        };

        Assert.Equal(0.0, stats.VectorizableRatio);
    }

    [Fact]
    public void SIMDStats_EstimatedSpeedup_NoHardwareAcceleration_ReturnsOne()
    {
        var stats = new SIMDStats
        {
            TotalOperations = 100,
            VectorizableOperations = 80,
            HardwareAccelerated = false,
            VectorSize = 4
        };

        Assert.Equal(1.0, stats.EstimatedSpeedup);
    }

    [Fact]
    public void SIMDStats_EstimatedSpeedup_ZeroOperations_ReturnsOne()
    {
        var stats = new SIMDStats
        {
            TotalOperations = 0,
            VectorizableOperations = 0,
            HardwareAccelerated = true,
            VectorSize = 4
        };

        Assert.Equal(1.0, stats.EstimatedSpeedup);
    }

    [Fact]
    public void SIMDStats_EstimatedSpeedup_FullyVectorizable_ReturnsHighValue()
    {
        var stats = new SIMDStats
        {
            TotalOperations = 100,
            VectorizableOperations = 100, // 100% vectorizable
            HardwareAccelerated = true,
            VectorSize = 8 // AVX
        };

        // Should be greater than 1 with full vectorization
        Assert.True(stats.EstimatedSpeedup > 1.0);
    }

    [Fact]
    public void SIMDStats_ToString_ReturnsFormattedString()
    {
        var stats = new SIMDStats
        {
            TotalOperations = 100,
            VectorizableOperations = 75,
            VectorSize = 8
        };

        var str = stats.ToString();

        Assert.Contains("SIMD Stats", str);
        Assert.Contains("75/100", str);
        Assert.Contains("vectorizable", str);
        Assert.Contains("Vector size: 8", str);
        Assert.Contains("speedup", str);
    }

    #endregion

    #region HybridCompilationResult Tests

    [Fact]
    public void HybridCompilationResult_DefaultConstructor_InitializesProperties()
    {
        var result = new HybridCompilationResult<double>();

        Assert.False(result.IsFullyJitCompiled);
        Assert.Equal("Unknown", result.ExecutionMode);
        Assert.NotNull(result.Compatibility);
        Assert.NotNull(result.Warnings);
        Assert.Empty(result.Warnings);
    }

    [Fact]
    public void HybridCompilationResult_CanSetIsFullyJitCompiled()
    {
        var result = new HybridCompilationResult<double>
        {
            IsFullyJitCompiled = true
        };

        Assert.True(result.IsFullyJitCompiled);
    }

    [Fact]
    public void HybridCompilationResult_CanSetExecutionMode()
    {
        var result = new HybridCompilationResult<double>
        {
            ExecutionMode = "JIT"
        };

        Assert.Equal("JIT", result.ExecutionMode);
    }

    [Fact]
    public void HybridCompilationResult_CanAddWarnings()
    {
        var result = new HybridCompilationResult<double>();
        result.Warnings.Add("Operation X uses fallback");
        result.Warnings.Add("Performance may be degraded");

        Assert.Equal(2, result.Warnings.Count);
    }

    [Fact]
    public void HybridCompilationResult_ToString_NoWarnings_ReturnsCorrectFormat()
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
    public void HybridCompilationResult_ToString_WithWarnings_IncludesWarningCount()
    {
        var result = new HybridCompilationResult<double>
        {
            IsFullyJitCompiled = false,
            ExecutionMode = "Hybrid"
        };
        result.Warnings.Add("Warning 1");
        result.Warnings.Add("Warning 2");

        var str = result.ToString();

        Assert.Contains("2 warnings", str);
    }

    #endregion

    #region Integration Tests - Multi-Operation Graphs

    [Fact]
    public void IRGraph_MultipleOperations_ChainedCorrectly()
    {
        // Build: result = ReLU(MatMul(input, weights) + bias)
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1, 2 }, // input, weights, bias
            OutputIds = new List<int> { 5 }
        };

        // Define shapes
        graph.TensorShapes[0] = new[] { 32, 784 };  // input: batch x features
        graph.TensorShapes[1] = new[] { 784, 128 }; // weights: features x hidden
        graph.TensorShapes[2] = new[] { 128 };      // bias: hidden

        // MatMul: t3 = t0 * t1
        graph.Operations.Add(new AddOp // Using AddOp as placeholder for MatMul
        {
            OutputIds = new[] { 3 },
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 32, 128 },
            OutputType = IRType.Float32
        });

        // Add: t4 = t3 + t2
        graph.Operations.Add(new AddOp
        {
            OutputIds = new[] { 4 },
            InputIds = new[] { 3, 2 },
            OutputShape = new[] { 32, 128 },
            OutputType = IRType.Float32
        });

        // ReLU: t5 = ReLU(t4) - using AddOp as placeholder
        graph.Operations.Add(new AddOp
        {
            OutputIds = new[] { 5 },
            InputIds = new[] { 4, 4 }, // Self-add as placeholder
            OutputShape = new[] { 32, 128 },
            OutputType = IRType.Float32
        });

        Assert.True(graph.Validate());
        Assert.Equal(3, graph.Operations.Count);
    }

    [Fact]
    public void IRGraph_WithConstants_ValidatesCorrectly()
    {
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0 },
            OutputIds = new List<int> { 2 }
        };

        graph.TensorShapes[0] = new[] { 3, 4 };

        // Constant: t1 = [1, 1, 1, ...]
        graph.Operations.Add(new ConstantOp
        {
            OutputIds = new[] { 1 },
            InputIds = Array.Empty<int>(),
            OutputShape = new[] { 3, 4 },
            OutputType = IRType.Float32,
            Values = Enumerable.Repeat(1.0, 12).ToArray()
        });

        // Add: t2 = t0 + t1
        graph.Operations.Add(new AddOp
        {
            OutputIds = new[] { 2 },
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 3, 4 },
            OutputType = IRType.Float32
        });

        Assert.True(graph.Validate());
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void IRGraph_EmptyOperationsWithInputsAndOutputs_IsValid()
    {
        // A graph that just passes through an input
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0 },
            OutputIds = new List<int> { 0 }
        };
        graph.TensorShapes[0] = new[] { 3, 4 };

        Assert.True(graph.Validate());
    }

    [Fact]
    public void CompilationStats_HighReduction_CalculatesCorrectly()
    {
        var stats = new CompilationStats
        {
            OriginalOperationCount = 1000,
            OptimizedOperationCount = 100
        };

        Assert.Equal(900, stats.OperationsEliminated);
        Assert.Equal(90.0, stats.OptimizationPercentage);
    }

    [Fact]
    public void SIMDCapabilities_AllFeatures_ListsAllInToString()
    {
        var caps = new SIMDCapabilities
        {
            HasSSE = true,
            HasAVX = true,
            HasAVX2 = true,
            HasAVX512 = true,
            HasFMA = true,
            HasNEON = true,
            MaxVectorWidth = 64
        };

        var str = caps.ToString();

        Assert.Contains("SSE", str);
        Assert.Contains("AVX", str);
        Assert.Contains("AVX2", str);
        Assert.Contains("AVX-512", str);
        Assert.Contains("FMA", str);
        Assert.Contains("NEON", str);
    }

    #endregion
}
