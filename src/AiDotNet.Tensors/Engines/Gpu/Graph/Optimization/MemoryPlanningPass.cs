using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Engines.Gpu.Graph.Optimization;

/// <summary>
/// Optimization pass that plans buffer reuse to minimize memory allocation.
/// Analyzes liveness of buffers and identifies reuse opportunities.
/// </summary>
public sealed class MemoryPlanningPass : IGraphOptimizationPass
{
    /// <inheritdoc/>
    public string Name => "MemoryPlanning";

    /// <inheritdoc/>
    public int Priority => 400; // Run after stream assignment

    /// <inheritdoc/>
    public bool IsEnabled { get; set; } = true;

    /// <inheritdoc/>
    public List<ExecutionNode> Apply(List<ExecutionNode> nodes, OptimizationContext context)
    {
        if (!IsEnabled)
        {
            return nodes;
        }

        // Compute buffer liveness
        var liveness = ComputeBufferLiveness(nodes);

        // Find reuse opportunities
        var reuseMap = ComputeReuseOpportunities(liveness, context);

        // Store the reuse map in context for the execution layer to use.
        // The execution layer will substitute buffers when executing nodes,
        // reusing memory from buffers that are no longer live.
        foreach (var kvp in reuseMap)
        {
            context.BufferReuseMap[kvp.Key] = kvp.Value;
        }

        // Update statistics
        context.Statistics.PassStats[Name] = new PassStatistics
        {
            PassName = Name,
            TransformationsApplied = reuseMap.Count
        };

        return nodes;
    }

    private static Dictionary<IGpuBuffer, BufferLiveness> ComputeBufferLiveness(List<ExecutionNode> nodes)
    {
        var liveness = new Dictionary<IGpuBuffer, BufferLiveness>();

        for (int i = 0; i < nodes.Count; i++)
        {
            var node = nodes[i];

            // Record first use (definition) for output buffers
            foreach (var output in node.OutputTensors)
            {
                if (!liveness.ContainsKey(output.Buffer))
                {
                    liveness[output.Buffer] = new BufferLiveness
                    {
                        Buffer = output.Buffer,
                        FirstUseIndex = i,
                        LastUseIndex = i,
                        Size = output.Buffer.Size,
                        Role = output.Role
                    };
                }
            }

            // Record last use for input buffers
            foreach (var input in node.InputTensors)
            {
                if (liveness.TryGetValue(input.Buffer, out var info))
                {
                    info.LastUseIndex = Math.Max(info.LastUseIndex, i);
                }
                else
                {
                    // External input - assume live for entire graph
                    liveness[input.Buffer] = new BufferLiveness
                    {
                        Buffer = input.Buffer,
                        FirstUseIndex = 0,
                        LastUseIndex = nodes.Count - 1,
                        Size = input.Buffer.Size,
                        Role = input.Role,
                        IsExternal = true
                    };
                }
            }
        }

        return liveness;
    }

    private static Dictionary<IGpuBuffer, IGpuBuffer> ComputeReuseOpportunities(
        Dictionary<IGpuBuffer, BufferLiveness> liveness,
        OptimizationContext context)
    {
        var reuseMap = new Dictionary<IGpuBuffer, IGpuBuffer>();

        // Sort by size (largest first) to maximize reuse
        var sortedBuffers = liveness.Values
            .Where(l => !l.IsExternal && l.Role != GpuTensorRole.Weight && l.Role != GpuTensorRole.Bias)
            .OrderByDescending(l => l.Size)
            .ToList();

        var availableBuffers = new List<BufferLiveness>();

        foreach (var buffer in sortedBuffers)
        {
            // Find a reusable buffer that is:
            // 1. Not live when this buffer is first used
            // 2. Large enough to hold this buffer's data
            var reusable = availableBuffers
                .FirstOrDefault(a =>
                    a.LastUseIndex < buffer.FirstUseIndex &&
                    a.Size >= buffer.Size);

            if (reusable != null)
            {
                // Reuse this buffer
                reuseMap[buffer.Buffer] = reusable.Buffer;

                // Update the reused buffer's last use
                reusable.LastUseIndex = buffer.LastUseIndex;
            }
            else
            {
                // Add to available pool for future reuse
                availableBuffers.Add(buffer);
            }
        }

        return reuseMap;
    }

    private sealed class BufferLiveness
    {
        public IGpuBuffer Buffer { get; init; } = null!;
        public int FirstUseIndex { get; set; }
        public int LastUseIndex { get; set; }
        public int Size { get; init; }
        public GpuTensorRole Role { get; init; }
        public bool IsExternal { get; init; }
    }
}
