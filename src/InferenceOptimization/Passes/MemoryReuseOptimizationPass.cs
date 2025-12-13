using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Optimizes memory usage by reusing buffers for different operations.
/// Analyzes the lifetime of tensors and assigns memory pools to reduce
/// overall memory footprint during inference.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class MemoryReuseOptimizationPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.MemoryReuseOptimization;
    public override string Name => "Memory Reuse Optimization";

    public override bool Apply(IComputationGraph<T> graph)
    {
        bool modified = false;

        // Perform liveness analysis
        var liveness = PerformLivenessAnalysis(graph);

        // Assign memory pools based on liveness
        var memoryPools = AssignMemoryPools(graph.Nodes, liveness);

        // Mark nodes with their memory pool assignments
        foreach (var kvp in memoryPools)
        {
            kvp.Key.Metadata["MemoryPoolId"] = kvp.Value;
            modified = true;
        }

        return modified;
    }

    private Dictionary<ComputationNode<T>, (int firstUse, int lastUse)> PerformLivenessAnalysis(
        IComputationGraph<T> graph)
    {
        var liveness = new Dictionary<ComputationNode<T>, (int, int)>();
        var topologicalOrder = graph.GetTopologicalOrder();

        for (int i = 0; i < topologicalOrder.Count; i++)
        {
            var node = topologicalOrder[i];

            // First use is when the node is computed
            var firstUse = i;

            // Last use is when the last consumer reads it
            var lastUse = i;

            foreach (var output in node.Outputs)
            {
                var outputIndex = topologicalOrder.IndexOf(output);
                if (outputIndex > lastUse)
                {
                    lastUse = outputIndex;
                }
            }

            liveness[node] = (firstUse, lastUse);
        }

        return liveness;
    }

    private Dictionary<ComputationNode<T>, int> AssignMemoryPools(
        List<ComputationNode<T>> nodes,
        Dictionary<ComputationNode<T>, (int firstUse, int lastUse)> liveness)
    {
        var poolAssignments = new Dictionary<ComputationNode<T>, int>();
        var pools = new List<(int lastUse, long size)>();

        // Sort nodes by first use
        var sortedNodes = nodes
            .Where(n => liveness.ContainsKey(n))
            .OrderBy(n => liveness[n].firstUse)
            .ToList();

        foreach (var node in sortedNodes)
        {
            var (firstUse, lastUse) = liveness[node];
            var tensorSize = EstimateTensorSize(node);

            // Find a pool that's no longer in use
            int assignedPool = -1;

            for (int i = 0; i < pools.Count; i++)
            {
                var (poolLastUse, poolSize) = pools[i];

                // Can reuse this pool if it's no longer active and size matches
                if (poolLastUse < firstUse && poolSize >= tensorSize)
                {
                    assignedPool = i;
                    pools[i] = (lastUse, poolSize);
                    break;
                }
            }

            // If no pool found, create a new one
            if (assignedPool == -1)
            {
                assignedPool = pools.Count;
                pools.Add((lastUse, tensorSize));
            }

            poolAssignments[node] = assignedPool;
        }

        return poolAssignments;
    }

    private long EstimateTensorSize(ComputationNode<T> node)
    {
        // Estimate the memory size of the output tensor
        if (node.OutputShape.Length == 0)
        {
            return 0;
        }

        long size = 1;
        foreach (var dim in node.OutputShape)
        {
            size *= dim;
        }

        // Multiply by size of T (rough estimate)
        var typeSize = typeof(T) == typeof(double) ? 8 :
                      typeof(T) == typeof(float) ? 4 :
                      typeof(T) == typeof(decimal) ? 16 : 8;

        return size * typeSize;
    }

    public override bool CanApply(IComputationGraph<T> graph)
    {
        return base.CanApply(graph) && graph.Nodes.Count > 2;
    }
}
