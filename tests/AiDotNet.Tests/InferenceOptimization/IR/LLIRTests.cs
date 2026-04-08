using AiDotNet.InferenceOptimization.IR.Common;
using AiDotNet.InferenceOptimization.IR.LowLevel;
using Xunit;

namespace AiDotNet.Tests.InferenceOptimization.IR;

/// <summary>
/// Tests for Low-Level IR classes.
/// </summary>
public class LLIRTests
{
    #region LLIROp Tests

    [Fact]
    public void MatMulOp_EstimateCost_CalculatesCorrectFLOPs()
    {
        var matmul = new MatMulOp
        {
            M = 128,
            N = 256,
            K = 512,
            OutputShape = new[] { 128, 256 }
        };

        var cost = matmul.EstimateCost();

        // 2 * M * N * K for matmul
        Assert.Equal(2L * 128 * 256 * 512, cost.FLOPs);
    }

    [Fact]
    public void MatMulOp_Validate_ReturnsTrueForValidOp()
    {
        var matmul = new MatMulOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 128, 256 },
            M = 128,
            N = 256,
            K = 512
        };

        Assert.True(matmul.Validate());
    }

    [Fact]
    public void ElementwiseOp_EstimateCost_CalculatesCorrectly()
    {
        var elementwise = new ElementwiseOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 2, 3, 4 },
            ElementwiseType = ElementwiseOpType.ReLU
        };

        var cost = elementwise.EstimateCost();

        Assert.Equal(24, cost.FLOPs); // 2*3*4 = 24 elements
    }

    [Fact]
    public void ElementwiseOp_FusedMultiplyAdd_DoublesOperationCount()
    {
        var fma = new ElementwiseOp
        {
            OutputId = 1,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 10 },
            ElementwiseType = ElementwiseOpType.FusedMultiplyAdd
        };

        var cost = fma.EstimateCost();

        Assert.Equal(20, cost.FLOPs); // 10 * 2 for FMA
    }

    [Fact]
    public void Conv2DOp_EstimateCost_CalculatesCorrectFLOPs()
    {
        var conv = new Conv2DOp
        {
            BatchSize = 1,
            InputChannels = 64,
            OutputChannels = 128,
            InputHeight = 56,
            InputWidth = 56,
            KernelHeight = 3,
            KernelWidth = 3,
            StrideH = 1,
            StrideW = 1,
            PadH = 1,
            PadW = 1,
            OutputShape = new[] { 1, 128, 56, 56 }
        };

        var cost = conv.EstimateCost();

        // 2 * BatchSize * OutputChannels * OutH * OutW * (InputChannels/Groups) * KH * KW
        var expectedFlops = 2L * 1 * 128 * 56 * 56 * 64 * 3 * 3;
        Assert.Equal(expectedFlops, cost.FLOPs);
    }

    [Fact]
    public void ReduceOp_EstimateCost_CalculatesCorrectly()
    {
        var reduce = new ReduceOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 4 },
            ReduceType = ReduceType.Sum,
            Axes = new[] { 1, 2 }
        };

        var cost = reduce.EstimateCost();

        Assert.True(cost.FLOPs > 0);
        Assert.True(cost.MemoryRead > 0);
    }

    [Fact]
    public void MemoryOp_EstimateCost_HasZeroFLOPs()
    {
        var memOp = new MemoryOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 2, 3, 4 },
            MemoryOpType = MemoryOpType.Reshape
        };

        var cost = memOp.EstimateCost();

        Assert.Equal(0, cost.FLOPs);
        Assert.True(cost.MemoryRead > 0);
        Assert.True(cost.MemoryWrite > 0);
    }

    [Fact]
    public void FusedOp_EstimateCost_CombinesOperations()
    {
        var op1 = new ElementwiseOp { OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.ReLU };
        var op2 = new ElementwiseOp { OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.Add };

        var fused = new FusedOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 },
            FusionPattern = "ReLU_Add",
            FusedOps = new List<LLIROp> { op1, op2 }
        };

        var cost = fused.EstimateCost();

        Assert.Equal(20, cost.FLOPs); // 10 + 10
    }

    [Fact]
    public void ConstantOp_EstimateCost_CalculatesCorrectly()
    {
        var constant = new ConstantOp
        {
            OutputId = 0,
            OutputShape = new[] { 10, 20 },
            IsParameter = true,
            ParameterName = "weights"
        };

        var cost = constant.EstimateCost();

        Assert.Equal(0, cost.FLOPs);
        Assert.True(cost.MemoryRead > 0);
    }

    [Fact]
    public void LLIROp_ToString_ReturnsFormattedString()
    {
        var matmul = new MatMulOp
        {
            OutputId = 2,
            InputIds = new[] { 0, 1 },
            OutputShape = new[] { 128, 256 },
            OutputDataType = IRDataType.Float32,
            Device = DeviceType.GPU
        };

        var str = matmul.ToString();

        Assert.Contains("b2", str);
        Assert.Contains("MatMul", str);
        Assert.Contains("b0", str);
        Assert.Contains("GPU", str);
    }

    #endregion

    #region ScheduleInfo Tests

    [Fact]
    public void ScheduleInfo_DefaultValues_AreCorrect()
    {
        var schedule = new ScheduleInfo();

        Assert.Empty(schedule.TileSizes);
        Assert.Empty(schedule.LoopOrder);
        Assert.Empty(schedule.ParallelAxes);
        Assert.Equal(-1, schedule.VectorAxis);
        Assert.Equal(1, schedule.VectorWidth);
        Assert.Equal(1, schedule.UnrollFactor);
    }

    [Fact]
    public void ScheduleInfo_Clone_CreatesIndependentCopy()
    {
        var original = new ScheduleInfo
        {
            TileSizes = new[] { 32, 32 },
            VectorWidth = 8,
            ThreadBlockDims = new[] { 256, 1, 1 }
        };

        var clone = original.Clone();

        Assert.Equal(original.TileSizes, clone.TileSizes);
        Assert.Equal(original.VectorWidth, clone.VectorWidth);
        Assert.Equal(original.ThreadBlockDims, clone.ThreadBlockDims);

        // Modify clone
        clone.TileSizes[0] = 64;
        Assert.NotEqual(original.TileSizes[0], clone.TileSizes[0]);
    }

    #endregion

    #region BufferInfo Tests

    [Fact]
    public void BufferInfo_DefaultValues_AreCorrect()
    {
        var buffer = new BufferInfo();

        Assert.Equal(0, buffer.SizeBytes);
        Assert.Equal(64, buffer.Alignment);
        Assert.Equal(-1, buffer.MemoryPoolId);
        Assert.Equal(MemoryLevel.DRAM, buffer.MemoryLevel);
        Assert.False(buffer.CanInPlace);
        Assert.False(buffer.IsPersistent);
    }

    [Fact]
    public void BufferInfo_InPlaceConfiguration_CanBeSet()
    {
        var buffer = new BufferInfo
        {
            CanInPlace = true,
            InPlaceInputId = 5,
            SizeBytes = 1024
        };

        Assert.True(buffer.CanInPlace);
        Assert.Equal(5, buffer.InPlaceInputId);
    }

    #endregion

    #region LLIRGraph Tests

    [Fact]
    public void LLIRGraph_AddOperation_AssignsOutputId()
    {
        var graph = new LLIRGraph();
        var op = new ElementwiseOp
        {
            OutputId = -1,
            OutputShape = new[] { 10 },
            ElementwiseType = ElementwiseOpType.ReLU
        };

        graph.AddOperation(op);

        Assert.True(op.OutputId >= 0);
        Assert.Single(graph.Operations);
    }

    [Fact]
    public void LLIRGraph_AllocateBufferId_ReturnsUniqueIds()
    {
        var graph = new LLIRGraph();

        var id1 = graph.AllocateBufferId();
        var id2 = graph.AllocateBufferId();
        var id3 = graph.AllocateBufferId();

        Assert.NotEqual(id1, id2);
        Assert.NotEqual(id2, id3);
        Assert.NotEqual(id1, id3);
    }

    [Fact]
    public void LLIRGraph_GetOperationByOutputId_ReturnsCorrectOp()
    {
        var graph = new LLIRGraph();
        var op1 = new ElementwiseOp { OutputId = 5, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.ReLU };
        var op2 = new ElementwiseOp { OutputId = 10, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.Add };

        graph.AddOperation(op1);
        graph.AddOperation(op2);

        Assert.Same(op1, graph.GetOperationByOutputId(5));
        Assert.Same(op2, graph.GetOperationByOutputId(10));
        Assert.Null(graph.GetOperationByOutputId(999));
    }

    [Fact]
    public void LLIRGraph_GetConsumers_ReturnsCorrectOperations()
    {
        var graph = new LLIRGraph();
        graph.InputIds.Add(0);

        var op1 = new ElementwiseOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.ReLU };
        var op2 = new ElementwiseOp { OutputId = 2, InputIds = new[] { 0 }, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.Add };
        var op3 = new ElementwiseOp { OutputId = 3, InputIds = new[] { 1 }, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.Sigmoid };

        graph.AddOperation(op1);
        graph.AddOperation(op2);
        graph.AddOperation(op3);

        var consumers = graph.GetConsumers(0).ToList();

        Assert.Equal(2, consumers.Count);
        Assert.Contains(op1, consumers);
        Assert.Contains(op2, consumers);
    }

    [Fact]
    public void LLIRGraph_Validate_ReturnsTrueForValidGraph()
    {
        var graph = new LLIRGraph();
        graph.InputIds.Add(0);
        graph.BufferShapes[0] = new[] { 10 };
        graph.BufferTypes[0] = IRDataType.Float32;

        var op = new ElementwiseOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 10 },
            ElementwiseType = ElementwiseOpType.ReLU
        };
        graph.AddOperation(op);
        graph.OutputIds.Add(1);

        var result = graph.Validate();

        Assert.True(result.IsValid);
    }

    [Fact]
    public void LLIRGraph_Validate_DetectsUndefinedInput()
    {
        var graph = new LLIRGraph();

        var op = new ElementwiseOp
        {
            OutputId = 1,
            InputIds = new[] { 999 }, // Undefined input
            OutputShape = new[] { 10 },
            ElementwiseType = ElementwiseOpType.ReLU
        };
        graph.AddOperation(op);

        var result = graph.Validate();

        Assert.False(result.IsValid);
        Assert.Contains(result.Errors, e => e.Contains("undefined buffer"));
    }

    [Fact]
    public void LLIRGraph_ComputeMetrics_CalculatesCorrectly()
    {
        var graph = new LLIRGraph();
        graph.InputIds.Add(0);
        graph.BufferShapes[0] = new[] { 128, 256 };

        var matmul = new MatMulOp
        {
            OutputId = 1,
            InputIds = new[] { 0 },
            OutputShape = new[] { 128, 256 },
            M = 128,
            N = 256,
            K = 512
        };
        graph.AddOperation(matmul);

        var metrics = graph.ComputeMetrics();

        Assert.Equal(1, metrics.OperationCount);
        Assert.True(metrics.TotalFLOPs > 0);
        Assert.True(metrics.PeakMemoryBytes > 0);
    }

    [Fact]
    public void LLIRGraph_OptimizeMemory_CreatesMemoryPlan()
    {
        var graph = new LLIRGraph();
        graph.InputIds.Add(0);
        graph.BufferShapes[0] = new[] { 10 };
        graph.BufferTypes[0] = IRDataType.Float32;

        var op1 = new ElementwiseOp { InputIds = new[] { 0 }, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.ReLU };
        var op2 = new ElementwiseOp { InputIds = new[] { op1.OutputId }, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.Add };

        graph.AddOperation(op1);
        op2.InputIds = new[] { op1.OutputId };
        graph.AddOperation(op2);

        graph.OptimizeMemory();

        Assert.NotNull(graph.MemoryPlan);
        Assert.True(graph.MemoryPlan.PoolCount >= 0);
    }

    [Fact]
    public void LLIRGraph_AutoSchedule_SetsSchedulingInfo()
    {
        var graph = new LLIRGraph();
        graph.DeviceConfig = new DeviceConfiguration { CPUVectorWidth = 8, CPUCores = 4 };

        var op = new ElementwiseOp
        {
            OutputId = 0,
            OutputShape = new[] { 64, 64 },
            ElementwiseType = ElementwiseOpType.ReLU,
            Device = DeviceType.CPU
        };
        graph.AddOperation(op);

        graph.AutoSchedule();

        // AutoSchedule should set vector width for CPU ops
        Assert.True(op.Schedule.VectorWidth >= 1);
    }

    [Fact]
    public void LLIRGraph_ComputeCriticalPath_CalculatesCorrectly()
    {
        var graph = new LLIRGraph();
        graph.InputIds.Add(0);
        graph.BufferShapes[0] = new[] { 10 };

        var op1 = new ElementwiseOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.ReLU };
        var op2 = new ElementwiseOp { OutputId = 2, InputIds = new[] { 1 }, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.Add };

        graph.AddOperation(op1);
        graph.AddOperation(op2);
        graph.OutputIds.Add(2);

        var criticalPath = graph.ComputeCriticalPath();

        Assert.True(criticalPath > 0);
    }

    [Fact]
    public void LLIRGraph_Clone_CreatesIndependentCopy()
    {
        var graph = new LLIRGraph { Name = "original" };
        graph.InputIds.Add(0);
        graph.BufferShapes[0] = new[] { 10 };

        var op = new ElementwiseOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.ReLU };
        graph.AddOperation(op);

        var clone = graph.Clone();

        Assert.Contains("_clone", clone.Name);
        Assert.Equal(graph.Operations.Count, clone.Operations.Count);
        Assert.Equal(graph.InputIds, clone.InputIds);
    }

    [Fact]
    public void LLIRGraph_ComputeStructureHash_ReturnsSameHashForSameStructure()
    {
        var graph1 = new LLIRGraph();
        graph1.InputIds.Add(0);
        graph1.BufferShapes[0] = new[] { 10 };
        var op1 = new ElementwiseOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.ReLU };
        graph1.AddOperation(op1);
        graph1.OutputIds.Add(1);

        var graph2 = new LLIRGraph();
        graph2.InputIds.Add(0);
        graph2.BufferShapes[0] = new[] { 10 };
        var op2 = new ElementwiseOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 10 }, ElementwiseType = ElementwiseOpType.ReLU };
        graph2.AddOperation(op2);
        graph2.OutputIds.Add(1);

        Assert.Equal(graph1.ComputeStructureHash(), graph2.ComputeStructureHash());
    }

    #endregion

    #region DeviceConfiguration Tests

    [Fact]
    public void DeviceConfiguration_DefaultValues_AreReasonable()
    {
        var config = new DeviceConfiguration();

        Assert.True(config.CPUCores > 0);
        Assert.True(config.CPUVectorWidth > 0);
        Assert.True(config.L1CacheBytes > 0);
        Assert.True(config.L2CacheBytes > 0);
        Assert.True(config.L3CacheBytes > 0);
        Assert.True(config.CPUMemoryBandwidth > 0);
        Assert.True(config.CPUPeakGFLOPS > 0);
    }

    #endregion

    #region OperationMetrics Tests

    [Fact]
    public void OperationMetrics_ArithmeticIntensity_CalculatesCorrectly()
    {
        var metrics = new OperationMetrics
        {
            FLOPs = 1000,
            IntOps = 0,
            MemoryRead = 50,
            MemoryWrite = 50
        };

        Assert.Equal(10.0, metrics.ArithmeticIntensity);
    }

    [Fact]
    public void OperationMetrics_RooflineGFLOPS_CalculatesCorrectly()
    {
        var metrics = new OperationMetrics
        {
            FLOPs = 1000,
            MemoryRead = 50,
            MemoryWrite = 50
        };

        double peakGFLOPS = 100;
        double memBandwidth = 50;

        var roofline = metrics.RooflineGFLOPS(peakGFLOPS, memBandwidth);

        // min(100, 10 * 50) = min(100, 500) = 100
        Assert.Equal(100, roofline);
    }

    #endregion

    #region MemoryPlan Tests

    [Fact]
    public void MemoryPlan_Validate_ReturnsTrueForValidPlan()
    {
        var plan = new MemoryPlan
        {
            PoolCount = 2,
            PoolSizes = new long[] { 1024, 2048 },
            BufferAssignments = new Dictionary<int, (int, long)>
            {
                { 0, (0, 0) },
                { 1, (1, 0) }
            }
        };

        var result = plan.Validate();

        Assert.True(result.IsValid);
    }

    [Fact]
    public void MemoryPlan_Validate_DetectsInvalidPoolId()
    {
        var plan = new MemoryPlan
        {
            PoolCount = 1,
            BufferAssignments = new Dictionary<int, (int, long)>
            {
                { 0, (5, 0) } // Invalid pool ID
            }
        };

        var result = plan.Validate();

        Assert.False(result.IsValid);
    }

    #endregion
}
