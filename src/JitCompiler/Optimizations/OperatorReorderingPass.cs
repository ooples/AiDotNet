using AiDotNet.JitCompiler.IR;

namespace AiDotNet.JitCompiler.Optimizations;

/// <summary>
/// Reorders independent operations to maximize data locality and cache line reuse.
/// When tensor A is consumed by operations B and C (both independent of each other),
/// schedule the one that shares more data with A first to keep A's data in cache.
/// </summary>
/// <remarks>
/// <para>
/// CPUs prefetch data in cache lines (typically 64 bytes). When operations are ordered
/// to process the same data sequentially, the data stays in cache and subsequent accesses
/// are fast (~1ns). If unrelated data is processed between accesses, the original data
/// gets evicted and must be reloaded from RAM (~50-100ns).
/// </para>
/// <para>
/// This pass uses a greedy algorithm:
/// 1. Build dependency graph (operation A depends on B if A uses B's output)
/// 2. For each set of independent operations (no dependencies between them),
///    sort by "data affinity" — how much input data they share with the previous operation
/// 3. Operations sharing more data with the previous one run first
/// </para>
/// </remarks>
public class OperatorReorderingPass : IOptimizationPass
{
    public string Name => "OperatorReordering";

    public IRGraph Optimize(IRGraph graph)
    {
        var operations = graph.Operations;
        if (operations.Count <= 2) return graph;

        // Build dependency sets: for each operation, which operations must run before it
        var dependsOn = new Dictionary<int, HashSet<int>>(); // opIndex -> set of opIndices it depends on
        var outputToOp = new Dictionary<int, int>(); // tensorId -> opIndex that produces it

        for (int i = 0; i < operations.Count; i++)
        {
            dependsOn[i] = new HashSet<int>();
            outputToOp[operations[i].OutputId] = i;
        }

        for (int i = 0; i < operations.Count; i++)
        {
            foreach (var inputId in operations[i].InputIds)
            {
                if (outputToOp.TryGetValue(inputId, out var producerIdx))
                {
                    dependsOn[i].Add(producerIdx);
                }
            }
        }

        // Topological sort with locality-aware tie-breaking
        var sorted = new List<IROp>();
        var completed = new HashSet<int>();
        var ready = new List<int>(); // operations whose dependencies are all completed

        // Initialize: find operations with no dependencies
        for (int i = 0; i < operations.Count; i++)
        {
            if (dependsOn[i].Count == 0)
                ready.Add(i);
        }

        int lastOutputId = -1; // track last operation's output for affinity scoring

        while (ready.Count > 0)
        {
            // Pick the ready operation with highest data affinity to the last operation
            int bestIdx = 0;
            int bestScore = -1;

            for (int r = 0; r < ready.Count; r++)
            {
                int opIdx = ready[r];
                int score = ComputeAffinity(operations[opIdx], lastOutputId);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestIdx = r;
                }
            }

            int chosen = ready[bestIdx];
            ready.RemoveAt(bestIdx);

            sorted.Add(operations[chosen]);
            completed.Add(chosen);
            lastOutputId = operations[chosen].OutputId;

            // Check if any new operations became ready
            for (int i = 0; i < operations.Count; i++)
            {
                if (completed.Contains(i)) continue;
                if (ready.Contains(i)) continue;

                bool allDepsCompleted = true;
                foreach (var dep in dependsOn[i])
                {
                    if (!completed.Contains(dep))
                    {
                        allDepsCompleted = false;
                        break;
                    }
                }

                if (allDepsCompleted)
                    ready.Add(i);
            }
        }

        // Replace operations with reordered list
        graph.Operations.Clear();
        graph.Operations.AddRange(sorted);

        // Store reordering stats in metadata
        int reorderCount = 0;
        for (int i = 0; i < sorted.Count && i < operations.Count; i++)
        {
            if (!ReferenceEquals(sorted[i], operations[i]))
                reorderCount++;
        }
        graph.Metadata["OperatorReordering.ReorderedCount"] = reorderCount;

        return graph;
    }

    /// <summary>
    /// Computes data affinity score between an operation and the previous operation's output.
    /// Higher score = more shared data = better cache locality if scheduled next.
    /// </summary>
    private static int ComputeAffinity(IROp op, int lastOutputId)
    {
        if (lastOutputId < 0) return 0;

        // Direct consumer: this operation reads the last operation's output
        // Maximum affinity — the data is still in cache
        foreach (var inputId in op.InputIds)
        {
            if (inputId == lastOutputId)
                return 100;
        }

        // No direct data sharing
        return 0;
    }
}
