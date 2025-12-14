using AiDotNet.InferenceOptimization.IR.Common;

namespace AiDotNet.InferenceOptimization.IR.LowLevel;

/// <summary>
/// Low-Level Intermediate Representation Graph.
/// Represents the computation graph optimized for hardware execution.
/// </summary>
/// <remarks>
/// <para><b>Design Philosophy:</b></para>
/// <para>
/// LLIRGraph is the final representation before code generation. It contains:
/// - Operations with scheduling information
/// - Memory allocation plan
/// - Device placement decisions
/// - Execution order
/// </para>
///
/// <para><b>Exceeds Standards By:</b></para>
/// <list type="bullet">
/// <item>Integrated memory planning with buffer reuse</item>
/// <item>Multi-device execution support</item>
/// <item>Streaming execution capability</item>
/// <item>Auto-tuning integration points</item>
/// </list>
/// </remarks>
public class LLIRGraph
{
    #region Properties

    /// <summary>
    /// Operations in execution order.
    /// </summary>
    public List<LLIROp> Operations { get; } = new();

    /// <summary>
    /// Mapping from buffer ID to shape.
    /// </summary>
    public Dictionary<int, int[]> BufferShapes { get; } = new();

    /// <summary>
    /// Mapping from buffer ID to data type.
    /// </summary>
    public Dictionary<int, IRDataType> BufferTypes { get; } = new();

    /// <summary>
    /// Input buffer IDs.
    /// </summary>
    public List<int> InputIds { get; } = new();

    /// <summary>
    /// Output buffer IDs.
    /// </summary>
    public List<int> OutputIds { get; } = new();

    /// <summary>
    /// Memory allocation plan.
    /// </summary>
    public MemoryPlan? MemoryPlan { get; set; }

    /// <summary>
    /// Graph name.
    /// </summary>
    public string Name { get; set; } = "LLIRGraph";

    /// <summary>
    /// Metadata.
    /// </summary>
    public Dictionary<string, object> Metadata { get; } = new();

    /// <summary>
    /// Target device configuration.
    /// </summary>
    public DeviceConfiguration DeviceConfig { get; set; } = new();

    /// <summary>
    /// Next buffer ID for allocation.
    /// </summary>
    private int _nextBufferId;

    #endregion

    #region Operations

    /// <summary>
    /// Adds an operation to the graph.
    /// </summary>
    public void AddOperation(LLIROp op)
    {
        if (op.OutputId < 0)
        {
            op.OutputId = _nextBufferId++;
        }
        else
        {
            _nextBufferId = Math.Max(_nextBufferId, op.OutputId + 1);
        }

        Operations.Add(op);
        BufferShapes[op.OutputId] = op.OutputShape;
        BufferTypes[op.OutputId] = op.OutputDataType;
    }

    /// <summary>
    /// Creates a new buffer ID.
    /// </summary>
    public int AllocateBufferId() => _nextBufferId++;

    /// <summary>
    /// Gets an operation by output buffer ID.
    /// </summary>
    public LLIROp? GetOperationByOutputId(int bufferId) =>
        Operations.FirstOrDefault(op => op.OutputId == bufferId);

    /// <summary>
    /// Gets all operations that use a buffer as input.
    /// </summary>
    public IEnumerable<LLIROp> GetConsumers(int bufferId) =>
        Operations.Where(op => op.InputIds.Contains(bufferId));

    #endregion

    #region Validation

    /// <summary>
    /// Validates the graph structure and scheduling.
    /// </summary>
    public LLIRValidationResult Validate()
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        // Check buffer definitions
        var definedBuffers = new HashSet<int>(InputIds);
        foreach (var op in Operations)
        {
            // Check inputs are defined
            foreach (var inputId in op.InputIds)
            {
                if (!definedBuffers.Contains(inputId))
                {
                    errors.Add($"Operation {op.Name} uses undefined buffer b{inputId}");
                }
            }

            // Check operation validity
            if (!op.Validate())
            {
                errors.Add($"Invalid operation: {op.Name}");
            }

            definedBuffers.Add(op.OutputId);
        }

        // Check outputs are defined
        foreach (var outputId in OutputIds)
        {
            if (!definedBuffers.Contains(outputId))
            {
                errors.Add($"Output buffer b{outputId} not defined");
            }
        }

        // Check scheduling
        foreach (var op in Operations)
        {
            if (op.Schedule.VectorWidth > 1 && op.Schedule.VectorAxis < 0)
            {
                warnings.Add($"Operation {op.Name} has vector width but no vector axis");
            }

            if (op.Device == DeviceType.GPU &&
                (op.Schedule.ThreadBlockDims.Length == 0 || op.Schedule.GridDims.Length == 0))
            {
                warnings.Add($"GPU operation {op.Name} missing thread/grid dimensions");
            }
        }

        // Check memory plan
        if (MemoryPlan != null)
        {
            var planResult = MemoryPlan.Validate();
            errors.AddRange(planResult.Errors);
            warnings.AddRange(planResult.Warnings);
        }

        return new LLIRValidationResult
        {
            IsValid = errors.Count == 0,
            Errors = errors,
            Warnings = warnings
        };
    }

    #endregion

    #region Analysis

    /// <summary>
    /// Computes total metrics for the graph.
    /// </summary>
    public LLIRGraphMetrics ComputeMetrics()
    {
        var metrics = new LLIRGraphMetrics();

        foreach (var op in Operations)
        {
            var opMetrics = op.EstimateCost();
            metrics.TotalFLOPs += opMetrics.FLOPs;
            metrics.TotalIntOps += opMetrics.IntOps;
            metrics.TotalMemoryRead += opMetrics.MemoryRead;
            metrics.TotalMemoryWrite += opMetrics.MemoryWrite;
            metrics.TotalLatencyNs += opMetrics.LatencyNs;

            if (!metrics.OpCountByType.ContainsKey(op.OpType))
            {
                metrics.OpCountByType[op.OpType] = 0;
            }
            metrics.OpCountByType[op.OpType]++;

            if (!metrics.FLOPsByDevice.ContainsKey(op.Device))
            {
                metrics.FLOPsByDevice[op.Device] = 0;
            }
            metrics.FLOPsByDevice[op.Device] += opMetrics.FLOPs;
        }

        metrics.OperationCount = Operations.Count;
        metrics.BufferCount = BufferShapes.Count;
        metrics.PeakMemoryBytes = MemoryPlan?.PeakMemoryBytes ?? EstimatePeakMemory();

        return metrics;
    }

    private long EstimatePeakMemory()
    {
        long peak = 0;
        long current = 0;
        var liveBuffers = new Dictionary<int, long>();

        // Add input buffers
        foreach (var inputId in InputIds)
        {
            if (BufferShapes.TryGetValue(inputId, out var shape))
            {
                var size = shape.Aggregate(1L, (a, b) => a * b) * GetElementSize(BufferTypes.GetValueOrDefault(inputId, IRDataType.Float32));
                liveBuffers[inputId] = size;
                current += size;
            }
        }
        peak = Math.Max(peak, current);

        foreach (var op in Operations)
        {
            // Add output buffer
            var outputSize = op.OutputShape.Aggregate(1L, (a, b) => a * b) *
                            GetElementSize(op.OutputDataType);
            liveBuffers[op.OutputId] = outputSize;
            current += outputSize;
            peak = Math.Max(peak, current);

            // Remove dead buffers (simplified - actual implementation would use liveness)
            // For now, just track peak
        }

        return peak;
    }

    private static int GetElementSize(IRDataType type) => type switch
    {
        IRDataType.Float16 or IRDataType.BFloat16 => 2,
        IRDataType.Float32 or IRDataType.Int32 or IRDataType.UInt32 => 4,
        IRDataType.Float64 or IRDataType.Int64 or IRDataType.UInt64 => 8,
        IRDataType.Int8 or IRDataType.UInt8 or IRDataType.QInt8 or IRDataType.QUInt8 => 1,
        IRDataType.Int16 or IRDataType.UInt16 => 2,
        _ => 4
    };

    /// <summary>
    /// Computes the critical path length.
    /// </summary>
    public long ComputeCriticalPath()
    {
        var latency = new Dictionary<int, long>();

        // Initialize inputs with 0 latency
        foreach (var inputId in InputIds)
        {
            latency[inputId] = 0;
        }

        foreach (var op in Operations)
        {
            var inputLatency = op.InputIds.Length > 0
                ? op.InputIds.Max(id => latency.GetValueOrDefault(id, 0))
                : 0;

            var opLatency = op.EstimateCost().LatencyNs;
            latency[op.OutputId] = inputLatency + opLatency;
        }

        return OutputIds.Count > 0
            ? OutputIds.Max(id => latency.GetValueOrDefault(id, 0))
            : latency.Values.DefaultIfEmpty(0).Max();
    }

    #endregion

    #region Optimization

    /// <summary>
    /// Applies memory optimization to reuse buffers.
    /// </summary>
    public void OptimizeMemory()
    {
        // Compute buffer liveness
        var liveness = ComputeLiveness();

        // Assign memory pools
        var pools = new List<(int lastUse, long size, int poolId)>();
        var poolAssignment = new Dictionary<int, (int poolId, long offset)>();

        foreach (var op in Operations)
        {
            var bufferId = op.OutputId;
            var (firstUse, lastUse) = liveness.GetValueOrDefault(bufferId, (0, Operations.Count));
            var bufferSize = op.OutputShape.Aggregate(1L, (a, b) => a * b) *
                            GetElementSize(op.OutputDataType);

            // Find reusable pool
            int assignedPool = -1;
            long offset = 0;

            for (int i = 0; i < pools.Count; i++)
            {
                var (poolLastUse, poolSize, poolId) = pools[i];
                if (poolLastUse < firstUse && poolSize >= bufferSize)
                {
                    assignedPool = poolId;
                    pools[i] = (lastUse, poolSize, poolId);
                    break;
                }
            }

            if (assignedPool < 0)
            {
                assignedPool = pools.Count;
                pools.Add((lastUse, bufferSize, assignedPool));
            }

            poolAssignment[bufferId] = (assignedPool, offset);

            // Update operation's buffer info
            op.BufferAllocation = new BufferInfo
            {
                SizeBytes = bufferSize,
                MemoryPoolId = assignedPool,
                PoolOffset = offset,
                FirstUseIndex = firstUse,
                LastUseIndex = lastUse
            };
        }

        // Create memory plan
        MemoryPlan = new MemoryPlan
        {
            PoolCount = pools.Count,
            PoolSizes = pools.Select(p => p.size).ToArray(),
            BufferAssignments = poolAssignment,
            PeakMemoryBytes = pools.Sum(p => p.size)
        };
    }

    private Dictionary<int, (int firstUse, int lastUse)> ComputeLiveness()
    {
        var liveness = new Dictionary<int, (int, int)>();

        for (int i = 0; i < Operations.Count; i++)
        {
            var op = Operations[i];

            // Output is first used here
            liveness[op.OutputId] = (i, i);

            // Update last use of inputs
            foreach (var inputId in op.InputIds)
            {
                if (liveness.TryGetValue(inputId, out var existing))
                {
                    liveness[inputId] = (existing.Item1, i);
                }
                else
                {
                    liveness[inputId] = (0, i);
                }
            }
        }

        return liveness;
    }

    /// <summary>
    /// Selects optimal schedules for operations based on device capabilities.
    /// </summary>
    public void AutoSchedule()
    {
        foreach (var op in Operations)
        {
            AutoScheduleOp(op);
        }
    }

    private void AutoScheduleOp(LLIROp op)
    {
        var schedule = op.Schedule;

        if (op.Device == DeviceType.CPU)
        {
            // CPU scheduling
            var vectorWidth = DeviceConfig.CPUVectorWidth;

            // Find best vectorization axis (innermost with size divisible by vector width)
            for (int i = op.OutputShape.Length - 1; i >= 0; i--)
            {
                if (op.OutputShape[i] >= vectorWidth && op.OutputShape[i] % vectorWidth == 0)
                {
                    schedule.VectorAxis = i;
                    schedule.VectorWidth = vectorWidth;
                    break;
                }
            }

            // Parallelization on outermost axis
            if (op.OutputShape.Length > 0 && op.OutputShape[0] >= DeviceConfig.CPUCores)
            {
                schedule.ParallelAxes = new[] { 0 };
            }

            // Tiling for cache
            if (op is MatMulOp matmul)
            {
                var tileSize = (int)Math.Sqrt(DeviceConfig.L2CacheBytes / 3 / 4); // 3 matrices, 4 bytes each
                tileSize = Math.Min(tileSize, 64);
                schedule.TileSizes = new[] { tileSize, tileSize, tileSize };
            }
        }
        else if (op.Device == DeviceType.GPU)
        {
            // GPU scheduling
            var totalElements = op.OutputShape.Aggregate(1L, (a, b) => a * b);

            // Thread block size
            var blockSize = Math.Min(256, (int)totalElements);
            var numBlocks = (int)Math.Ceiling((double)totalElements / blockSize);

            schedule.ThreadBlockDims = new[] { blockSize, 1, 1 };
            schedule.GridDims = new[] { numBlocks, 1, 1 };

            // Shared memory for reductions
            if (op is ReduceOp)
            {
                schedule.SharedMemoryBytes = blockSize * 4;
            }
        }
    }

    #endregion

    #region Utilities

    /// <summary>
    /// Creates a copy of the graph.
    /// </summary>
    public LLIRGraph Clone()
    {
        var clone = new LLIRGraph { Name = Name + "_clone" };

        foreach (var op in Operations)
        {
            // Note: In production, implement proper deep clone for each op type
            clone.Operations.Add(op);
        }

        foreach (var kvp in BufferShapes)
        {
            clone.BufferShapes[kvp.Key] = (int[])kvp.Value.Clone();
        }

        foreach (var kvp in BufferTypes)
        {
            clone.BufferTypes[kvp.Key] = kvp.Value;
        }

        clone.InputIds.AddRange(InputIds);
        clone.OutputIds.AddRange(OutputIds);
        clone.MemoryPlan = MemoryPlan;
        clone.DeviceConfig = DeviceConfig;

        foreach (var kvp in Metadata)
        {
            clone.Metadata[kvp.Key] = kvp.Value;
        }

        return clone;
    }

    /// <summary>
    /// Computes structure hash for caching.
    /// </summary>
    public int ComputeStructureHash()
    {
        int hash = 17;

        foreach (var inputId in InputIds.OrderBy(id => id))
        {
            hash = hash * 31 + inputId;
            if (BufferShapes.TryGetValue(inputId, out var shape))
            {
                foreach (var dim in shape)
                {
                    hash = hash * 31 + dim;
                }
            }
        }

        foreach (var op in Operations)
        {
            hash = hash * 31 + op.OpType.GetHashCode();
            hash = hash * 31 + op.OutputId;
            hash = hash * 31 + op.OutputDataType.GetHashCode();
            foreach (var dim in op.OutputShape)
            {
                hash = hash * 31 + dim;
            }
            foreach (var inputId in op.InputIds)
            {
                hash = hash * 31 + inputId;
            }
        }

        foreach (var outputId in OutputIds.OrderBy(id => id))
        {
            hash = hash * 31 + outputId;
        }

        return hash;
    }

    public override string ToString()
    {
        var metrics = ComputeMetrics();
        return $"LLIRGraph '{Name}': {Operations.Count} ops, {BufferShapes.Count} buffers, " +
               $"{metrics.TotalFLOPs:N0} FLOPs, {metrics.PeakMemoryBytes / 1024.0 / 1024.0:F2} MB peak";
    }

    #endregion
}

/// <summary>
/// Memory allocation plan.
/// </summary>
public class MemoryPlan
{
    public int PoolCount { get; set; }
    public long[] PoolSizes { get; set; } = Array.Empty<long>();
    public Dictionary<int, (int poolId, long offset)> BufferAssignments { get; set; } = new();
    public long PeakMemoryBytes { get; set; }

    public LLIRValidationResult Validate()
    {
        var errors = new List<string>();
        var warnings = new List<string>();

        // Check pool assignments
        foreach (var (bufferId, (poolId, offset)) in BufferAssignments)
        {
            if (poolId < 0 || poolId >= PoolCount)
            {
                errors.Add($"Buffer {bufferId} assigned to invalid pool {poolId}");
            }

            if (offset < 0)
            {
                errors.Add($"Buffer {bufferId} has negative offset {offset}");
            }
        }

        return new LLIRValidationResult
        {
            IsValid = errors.Count == 0,
            Errors = errors,
            Warnings = warnings
        };
    }
}

/// <summary>
/// Device configuration for scheduling.
/// </summary>
public class DeviceConfiguration
{
    // CPU
    public int CPUCores { get; set; } = Environment.ProcessorCount;
    public int CPUVectorWidth { get; set; } = 8; // AVX-256 default
    public long L1CacheBytes { get; set; } = 32 * 1024;
    public long L2CacheBytes { get; set; } = 256 * 1024;
    public long L3CacheBytes { get; set; } = 8 * 1024 * 1024;

    // GPU
    public int GPUSMCount { get; set; } = 0;
    public int GPUMaxThreadsPerBlock { get; set; } = 1024;
    public long GPUSharedMemoryPerBlock { get; set; } = 48 * 1024;
    public long GPUGlobalMemory { get; set; } = 0;
    public bool GPUHasTensorCores { get; set; } = false;

    // Memory bandwidth (GB/s)
    public double CPUMemoryBandwidth { get; set; } = 50;
    public double GPUMemoryBandwidth { get; set; } = 500;

    // Peak compute (GFLOPS)
    public double CPUPeakGFLOPS { get; set; } = 500;
    public double GPUPeakGFLOPS { get; set; } = 10000;
}

/// <summary>
/// LLIR validation result.
/// </summary>
public class LLIRValidationResult
{
    public bool IsValid { get; init; }
    public List<string> Errors { get; init; } = new();
    public List<string> Warnings { get; init; } = new();
}

/// <summary>
/// LLIR graph metrics.
/// </summary>
public class LLIRGraphMetrics
{
    public int OperationCount { get; set; }
    public int BufferCount { get; set; }
    public long TotalFLOPs { get; set; }
    public long TotalIntOps { get; set; }
    public long TotalMemoryRead { get; set; }
    public long TotalMemoryWrite { get; set; }
    public long TotalLatencyNs { get; set; }
    public long PeakMemoryBytes { get; set; }
    public Dictionary<string, int> OpCountByType { get; } = new();
    public Dictionary<DeviceType, long> FLOPsByDevice { get; } = new();

    public double ArithmeticIntensity =>
        (TotalMemoryRead + TotalMemoryWrite) > 0
            ? (double)(TotalFLOPs + TotalIntOps) / (TotalMemoryRead + TotalMemoryWrite)
            : 0;
}
