using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Fuses consecutive elementwise operations into a single operation.
/// For example: (x + y) * z can be computed in a single fused kernel.
/// This reduces memory bandwidth by avoiding intermediate results.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
/// <remarks>
/// <para>
/// Elementwise fusion is a key optimization for neural network inference that reduces memory
/// bandwidth requirements. Instead of computing each elementwise operation separately and
/// writing intermediate results to memory, fused operations process data in a single pass.
/// </para>
/// <para><b>Fusion Strategy:</b></para>
/// <list type="bullet">
/// <item><description>Only linear chains are fused (single consumer at each step)</description></item>
/// <item><description>Chains must start from "head" nodes (not outputs of other fuseable ops)</description></item>
/// <item><description>Disjoint chains are processed to avoid overlapping fusion attempts</description></item>
/// </list>
/// <para><b>Supported Operations:</b> Add, Subtract, Multiply, Divide, Power, Sqrt, Exp, Log, ReLU, Sigmoid, Tanh</para>
/// <para><b>Performance Impact:</b> Typically 2-3x speedup for memory-bound elementwise sequences.</para>
/// </remarks>
public class ElementwiseFusionPass<T> : OptimizationPassBase<T> where T : struct
{
    /// <inheritdoc/>
    public override OptimizationPassType PassType => OptimizationPassType.ElementwiseFusion;

    /// <inheritdoc/>
    public override string Name => "Elementwise Operation Fusion";

    /// <summary>
    /// The set of operation types that can be fused in an elementwise chain.
    /// </summary>
    private static readonly HashSet<OperationType> ElementwiseOps = new()
    {
        OperationType.Add,
        OperationType.Subtract,
        OperationType.Multiply,
        OperationType.Divide,
        OperationType.Power,
        OperationType.Sqrt,
        OperationType.Exp,
        OperationType.Log,
        OperationType.ReLU,
        OperationType.Sigmoid,
        OperationType.Tanh
    };

    /// <summary>
    /// Applies elementwise fusion optimization to the graph.
    /// </summary>
    /// <param name="graph">The optimization graph to transform.</param>
    /// <returns>True if any operations were fused; false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// The algorithm identifies disjoint chains of elementwise operations by:
    /// <list type="number">
    /// <item><description>Finding chain "heads" - nodes that are not single-consumer outputs of other elementwise ops</description></item>
    /// <item><description>Building chains forward from each head, tracking visited nodes to ensure disjointness</description></item>
    /// <item><description>Fusing chains with 2+ operations into single fused nodes</description></item>
    /// </list>
    /// </para>
    /// <para><b>Thread Safety:</b> This method modifies the graph structure and is not thread-safe.</para>
    /// </remarks>
    public override bool Apply(IOptimizationGraph<T> graph)
    {
        bool modified = false;

        // Track visited nodes to ensure disjoint chains
        var visited = new HashSet<OptimizationNode<T>>();

        // Get all unfused elementwise candidates
        var candidates = graph.Nodes
            .Where(n => ElementwiseOps.Contains(n.OperationType) && !n.IsFused)
            .ToList();

        foreach (var node in candidates)
        {
            // Skip nodes already processed in another chain
            if (visited.Contains(node)) continue;

            // Only start chains from "head" nodes to avoid overlapping chains
            if (!IsChainHead(node)) continue;

            var chain = FindElementwiseChain(node, visited);
            if (chain.Count >= 2)
            {
                FuseElementwiseChain(graph, chain);
                modified = true;
            }
        }

        return modified;
    }

    /// <summary>
    /// Determines whether a node is a valid chain head for elementwise fusion.
    /// </summary>
    /// <param name="node">The node to check.</param>
    /// <returns>True if the node can be the start of a fusion chain; false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// A node is a chain head if it is NOT the single-consumer output of another unfused
    /// elementwise operation. This ensures we start chains at their natural beginning
    /// rather than from the middle, preventing overlapping fusion attempts.
    /// </para>
    /// </remarks>
    private bool IsChainHead(OptimizationNode<T> node)
    {
        // Head if it's not the single-consumer output of another unfused elementwise op
        return !node.Inputs.Any(input =>
            ElementwiseOps.Contains(input.OperationType) &&
            !input.IsFused &&
            input.Outputs.Count == 1 &&
            input.Outputs[0] == node);
    }

    /// <summary>
    /// Discovers a chain of consecutive elementwise operations starting from the given node.
    /// </summary>
    /// <param name="startNode">The starting node for chain discovery.</param>
    /// <param name="visited">Set of already-visited nodes to ensure disjoint chains.</param>
    /// <returns>A list of nodes forming the elementwise chain, starting with the head.</returns>
    /// <remarks>
    /// <para>
    /// Chain discovery follows these rules:
    /// <list type="bullet">
    /// <item><description>Only single-output nodes can extend the chain</description></item>
    /// <item><description>The next node must be an unfused elementwise operation</description></item>
    /// <item><description>The next node must have the current node in its inputs</description></item>
    /// <item><description>Already-visited nodes cannot be added to the chain</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private List<OptimizationNode<T>> FindElementwiseChain(
        OptimizationNode<T> startNode,
        HashSet<OptimizationNode<T>> visited)
    {
        var chain = new List<OptimizationNode<T>> { startNode };
        var current = startNode;
        visited.Add(current);

        // Follow the chain forward through single-consumer elementwise ops
        while (current.Outputs.Count == 1)
        {
            var next = current.Outputs[0];

            // Stop if not a simple single-consumer elementwise progression
            if (!ElementwiseOps.Contains(next.OperationType) ||
                next.IsFused ||
                visited.Contains(next) ||
                !next.Inputs.Contains(current))
            {
                break;
            }

            chain.Add(next);
            current = next;
            visited.Add(current);
        }

        return chain;
    }

    /// <summary>
    /// Fuses a chain of elementwise operations into a single fused node.
    /// </summary>
    /// <param name="graph">The optimization graph being transformed.</param>
    /// <param name="chain">The chain of nodes to fuse, ordered from head to tail.</param>
    /// <remarks>
    /// <para>
    /// The fusion process:
    /// <list type="number">
    /// <item><description>Create a new fused node with combined operation sequence</description></item>
    /// <item><description>Connect all external inputs to the fused node</description></item>
    /// <item><description>Redirect all outputs of the chain tail to the fused node</description></item>
    /// <item><description>Remove all original chain nodes from the graph</description></item>
    /// </list>
    /// </para>
    /// <para><b>Metadata:</b> The fused node stores the operation sequence in its metadata
    /// under the "OperationSequence" key for code generation.</para>
    /// </remarks>
    private void FuseElementwiseChain(IOptimizationGraph<T> graph, List<OptimizationNode<T>> chain)
    {
        var firstNode = chain[0];
        var lastNode = chain[^1];

        // Create fused elementwise node
        var fusedNode = new OptimizationNode<T>
        {
            OperationType = OperationType.Custom,
            Name = $"{firstNode.Name}_elementwise_fused",
            OutputShape = lastNode.OutputShape,
            IsFused = true,
            CanOperateInPlace = chain.All(n => n.CanOperateInPlace),
            FusedFrom = new List<OptimizationNode<T>>(chain)
        };

        // Store the operation sequence for code generation
        fusedNode.Metadata["OperationSequence"] = chain.Select(n => n.OperationType).ToList();

        // Collect all unique inputs from the chain that are not part of the chain itself
        var externalInputs = new HashSet<OptimizationNode<T>>(chain
            .SelectMany(node => node.Inputs)
            .Where(input => !chain.Contains(input)));

        // Connect external inputs to the fused node
        foreach (var input in externalInputs)
        {
            fusedNode.AddInput(input);

            // Remove connections from external inputs to chain nodes
            foreach (var chainNode in chain)
            {
                input.Outputs.Remove(chainNode);
            }
        }

        // Connect outputs - use ToList() to avoid collection modification during iteration
        foreach (var output in lastNode.Outputs.ToList())
        {
            output.ReplaceInput(lastNode, fusedNode);
        }

        // Add fused node to graph
        graph.AddNode(fusedNode);

        // Remove original chain nodes from graph
        foreach (var node in chain)
        {
            // Clear internal chain connections before removal to avoid dangling references
            node.Inputs.Clear();
            node.Outputs.Clear();
            graph.RemoveNode(node);
        }
    }

    /// <inheritdoc/>
    public override bool CanApply(IOptimizationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => ElementwiseOps.Contains(n.OperationType));
    }
}
