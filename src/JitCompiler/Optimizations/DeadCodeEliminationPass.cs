using System.Linq;
using AiDotNet.JitCompiler.IR;

namespace AiDotNet.JitCompiler.Optimizations;

/// <summary>
/// Optimization pass that removes operations whose results are never used.
/// </summary>
/// <remarks>
/// <para>
/// Dead code elimination (DCE) is a compiler optimization that identifies and removes
/// operations whose results don't contribute to the final output. This can occur when:
/// - Intermediate results are computed but never used
/// - Previous optimizations make some operations redundant
/// - The graph was constructed with unnecessary operations
/// </para>
/// <para><b>For Beginners:</b> This removes calculations that don't affect the final result.
///
/// Think of it like cleaning up a recipe:
/// - Original: "Mix A and B. Mix C and D. Use the first mixture for the cake."
/// - Optimized: "Mix A and B. Use the mixture for the cake."
/// - We removed "Mix C and D" because it's never used!
///
/// Why this helps:
/// - Fewer operations to execute (faster)
/// - Less memory needed
/// - Simpler graph to work with
///
/// Example in neural networks:
/// - You might compute an intermediate layer's output
/// - But then decide not to use it in the final prediction
/// - DCE removes that unused layer computation
/// - Saves time and memory!
///
/// This is especially common after other optimizations that might make
/// some operations unnecessary.
/// </para>
/// </remarks>
public class DeadCodeEliminationPass : IOptimizationPass
{
    /// <summary>
    /// Gets the name of this optimization pass.
    /// </summary>
    public string Name => "Dead Code Elimination";

    /// <summary>
    /// Applies dead code elimination to an IR graph.
    /// </summary>
    /// <param name="graph">The IR graph to optimize.</param>
    /// <returns>An optimized IR graph with dead code removed.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a backward traversal from the output nodes to identify
    /// which operations are actually needed. Any operation not reached during this
    /// traversal is dead code and can be safely removed.
    /// </para>
    /// <para><b>For Beginners:</b> This figures out what's needed and removes the rest.
    ///
    /// The process:
    /// 1. Start from the output nodes (what we actually want to compute)
    /// 2. Work backwards to find all operations needed to produce those outputs
    /// 3. Mark those operations as "live" (needed)
    /// 4. Remove all operations that aren't marked as live
    /// 5. Return the cleaned-up graph
    ///
    /// Example transformation:
    /// Before:
    ///   t2 = Add(t0, t1)
    ///   t3 = Mul(t0, t1)      ← Dead! Never used
    ///   t4 = ReLU(t2)
    ///   Output: t4
    ///
    /// After:
    ///   t2 = Add(t0, t1)
    ///   t4 = ReLU(t2)
    ///   Output: t4
    ///
    /// The Mul operation is gone because its result (t3) was never used!
    /// </para>
    /// </remarks>
    public IRGraph Optimize(IRGraph graph)
    {
        // Track which tensors are live (actually needed)
        var liveTensors = new HashSet<int>();

        // All outputs are live
        foreach (var outputId in graph.OutputIds)
        {
            liveTensors.Add(outputId);
        }

        // Work backwards through operations to find all live tensors
        // We need to iterate until no more live tensors are found (fixed point)
        bool changed = true;
        while (changed)
        {
            changed = false;
            int previousCount = liveTensors.Count;

            // Check each operation in reverse order
            for (int i = graph.Operations.Count - 1; i >= 0; i--)
            {
                var op = graph.Operations[i];

                // If this operation's output is live, all its inputs must be live too
                if (liveTensors.Contains(op.OutputId))
                {
                    foreach (var inputId in op.InputIds)
                    {
                        liveTensors.Add(inputId);
                    }
                }
            }

            // Check if we found new live tensors
            changed = liveTensors.Count > previousCount;
        }

        // Build optimized graph with only live operations
        var optimizedGraph = new IRGraph
        {
            InputIds = new List<int>(graph.InputIds),
            OutputIds = new List<int>(graph.OutputIds),
            TensorShapes = new Dictionary<int, int[]>(),
            Metadata = new Dictionary<string, object>(graph.Metadata)
        };

        // Keep only operations whose outputs are live
        int removedCount = 0;
        foreach (var op in graph.Operations.Where(o => liveTensors.Contains(o.OutputId)))
        {
            optimizedGraph.Operations.Add(op);

            // Copy shape information for live tensors
            if (graph.TensorShapes.TryGetValue(op.OutputId, out var shape))
            {
                optimizedGraph.TensorShapes[op.OutputId] = shape;
            }
        }
        removedCount = graph.Operations.Count - optimizedGraph.Operations.Count;

        // Copy shape information for inputs
        foreach (var inputId in graph.InputIds)
        {
            if (graph.TensorShapes.TryGetValue(inputId, out var shape))
            {
                optimizedGraph.TensorShapes[inputId] = shape;
            }
        }

        // Add metadata about optimization results
        if (removedCount > 0)
        {
            optimizedGraph.Metadata["DCE_RemovedOps"] = removedCount;
            optimizedGraph.Metadata["DCE_OriginalOps"] = graph.Operations.Count;
        }

        return optimizedGraph;
    }

    /// <summary>
    /// Identifies dead code in a graph without removing it (for analysis).
    /// </summary>
    /// <param name="graph">The IR graph to analyze.</param>
    /// <returns>A set of tensor IDs that correspond to dead operations.</returns>
    /// <remarks>
    /// <para>
    /// This method performs the same liveness analysis as Optimize but returns
    /// the set of dead tensor IDs instead of creating a new graph. Useful for
    /// debugging and analysis.
    /// </para>
    /// <para><b>For Beginners:</b> This finds dead code without removing it.
    ///
    /// Use this when you want to:
    /// - Analyze the graph to see how much dead code exists
    /// - Debug why certain operations aren't being used
    /// - Generate reports about graph efficiency
    ///
    /// Returns the IDs of operations that would be removed by DCE.
    /// </para>
    /// </remarks>
    public HashSet<int> IdentifyDeadCode(IRGraph graph)
    {
        // Track which tensors are live
        var liveTensors = new HashSet<int>();

        // All outputs are live
        foreach (var outputId in graph.OutputIds)
        {
            liveTensors.Add(outputId);
        }

        // Work backwards to find all live tensors
        bool changed = true;
        while (changed)
        {
            changed = false;
            int previousCount = liveTensors.Count;

            for (int i = graph.Operations.Count - 1; i >= 0; i--)
            {
                var op = graph.Operations[i];
                if (liveTensors.Contains(op.OutputId))
                {
                    foreach (var inputId in op.InputIds)
                    {
                        liveTensors.Add(inputId);
                    }
                }
            }

            changed = liveTensors.Count > previousCount;
        }

        // Find all dead operation outputs
        var deadTensors = new HashSet<int>();
        foreach (var op in graph.Operations.Where(o => !liveTensors.Contains(o.OutputId)))
        {
            deadTensors.Add(op.OutputId);
        }

        return deadTensors;
    }

    /// <summary>
    /// Gets statistics about dead code in a graph.
    /// </summary>
    /// <param name="graph">The IR graph to analyze.</param>
    /// <returns>A tuple of (total operations, live operations, dead operations).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This counts how many operations are dead vs alive.
    ///
    /// Returns:
    /// - Total: Total number of operations in the graph
    /// - Live: Number of operations that contribute to outputs
    /// - Dead: Number of operations that can be removed
    ///
    /// Useful for understanding graph efficiency before and after optimization.
    /// </para>
    /// </remarks>
    public (int Total, int Live, int Dead) GetStatistics(IRGraph graph)
    {
        var deadTensors = IdentifyDeadCode(graph);
        int total = graph.Operations.Count;
        int dead = deadTensors.Count;
        int live = total - dead;

        return (total, live, dead);
    }
}
