using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;

namespace AiDotNet.InferenceOptimization.Passes;

/// <summary>
/// Optimizes tensor layout (NCHW vs NHWC) for better hardware utilization.
/// Different hardware architectures prefer different layouts:
/// - NCHW: Better for GPUs (NVIDIA)
/// - NHWC: Better for some CPUs and TPUs
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
/// <remarks>
/// <para>
/// Tensor layout optimization is crucial for achieving peak performance on different hardware.
/// The layout determines how tensor dimensions are stored in memory:
/// <list type="bullet">
/// <item><description><b>NCHW</b>: Batch-Channel-Height-Width (preferred by NVIDIA GPUs, cuDNN)</description></item>
/// <item><description><b>NHWC</b>: Batch-Height-Width-Channel (preferred by CPUs, TPUs, some mobile GPUs)</description></item>
/// </list>
/// </para>
/// <para><b>How It Works:</b> This pass analyzes the graph to identify nodes with different
/// layout preferences, then inserts transpose operations at layout boundaries to ensure
/// optimal memory access patterns for each operation.</para>
/// <para><b>Performance Impact:</b> Proper layout selection can yield 20-50% speedup
/// for memory-bound convolution operations.</para>
/// </remarks>
public class LayoutOptimizationPass<T> : OptimizationPassBase<T> where T : struct
{
    /// <inheritdoc/>
    public override OptimizationPassType PassType => OptimizationPassType.LayoutOptimization;

    /// <inheritdoc/>
    public override string Name => "Layout Optimization";

    /// <summary>
    /// The target memory layout to optimize for.
    /// </summary>
    private readonly string _targetLayout;

    /// <summary>
    /// Operations that prefer channel-first layout (NCHW).
    /// </summary>
    private static readonly HashSet<OperationType> ChannelFirstOps = new()
    {
        OperationType.Convolution,
        OperationType.Convolution2D,
        OperationType.Conv2D,
        OperationType.BatchNormalization,
        OperationType.MaxPooling,
        OperationType.AveragePooling
    };

    /// <summary>
    /// The set of supported memory layouts for tensor operations.
    /// </summary>
    private static readonly HashSet<string> SupportedLayouts = new() { "NCHW", "NHWC" };

    /// <summary>
    /// Initializes a new instance of the <see cref="LayoutOptimizationPass{T}"/> class.
    /// </summary>
    /// <param name="targetLayout">The target layout to optimize for. Default is "NCHW".</param>
    /// <exception cref="ArgumentException">
    /// Thrown when the specified targetLayout is not supported.
    /// </exception>
    /// <remarks>
    /// <para>Supported layouts:</para>
    /// <list type="bullet">
    /// <item><description>"NCHW" - Batch-Channel-Height-Width (preferred by NVIDIA GPUs)</description></item>
    /// <item><description>"NHWC" - Batch-Height-Width-Channel (preferred by CPUs/TPUs)</description></item>
    /// </list>
    /// </remarks>
    public LayoutOptimizationPass(string targetLayout = "NCHW")
    {
        if (!SupportedLayouts.Contains(targetLayout))
        {
            throw new ArgumentException(
                $"Unsupported layout '{targetLayout}'. Supported layouts: {string.Join(", ", SupportedLayouts)}.",
                nameof(targetLayout));
        }
        _targetLayout = targetLayout;
    }

    /// <summary>
    /// Applies layout optimization to the graph by inserting transpose operations at layout boundaries.
    /// </summary>
    /// <param name="graph">The optimization graph to transform.</param>
    /// <returns>True if any transpose operations were inserted; false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// The optimization process:
    /// <list type="number">
    /// <item><description>Analyze each node's preferred layout based on operation type</description></item>
    /// <item><description>Identify layout mismatches between connected nodes</description></item>
    /// <item><description>Insert transpose operations to convert between layouts</description></item>
    /// </list>
    /// </para>
    /// <para><b>Thread Safety:</b> This method modifies the graph structure and is not thread-safe.</para>
    /// </remarks>
    public override bool Apply(IOptimizationGraph<T> graph)
    {
        bool modified = false;

        // Analyze the graph to determine optimal layout for each node
        var layoutInfo = AnalyzeLayouts(graph);

        // Find nodes that require layout conversion due to mismatched inputs
        var nodesToConvert = graph.Nodes
            .Where(n => RequiresLayoutConversion(n, layoutInfo))
            .ToList();

        // Insert transpose operations where needed
        foreach (var node in nodesToConvert)
        {
            if (InsertLayoutConversion(graph, node, layoutInfo))
            {
                modified = true;
            }
        }

        return modified;
    }

    /// <summary>
    /// Analyzes the graph to determine the preferred layout for each node.
    /// </summary>
    /// <param name="graph">The optimization graph to analyze.</param>
    /// <returns>A dictionary mapping each node to its preferred layout string.</returns>
    private Dictionary<OptimizationNode<T>, string> AnalyzeLayouts(IOptimizationGraph<T> graph)
    {
        var layouts = new Dictionary<OptimizationNode<T>, string>();

        foreach (var node in graph.Nodes)
        {
            // Determine preferred layout for this operation
            var preferredLayout = GetPreferredLayout(node.OperationType);
            layouts[node] = preferredLayout;
        }

        return layouts;
    }

    /// <summary>
    /// Gets the intrinsic preferred memory layout for a given operation type.
    /// </summary>
    /// <param name="opType">The operation type to check.</param>
    /// <returns>The intrinsic preferred layout string ("NCHW", "NHWC", or "AGNOSTIC").</returns>
    /// <remarks>
    /// <para>
    /// This method returns the operation's intrinsic layout preference, which is independent
    /// of the target hardware layout. Operations like convolution, batch normalization, and
    /// pooling inherently prefer NCHW layout due to their memory access patterns.
    /// </para>
    /// <para>
    /// The pass then inserts layout conversions at boundaries between operations with
    /// different intrinsic preferences and the target layout.
    /// </para>
    /// </remarks>
    private string GetPreferredLayout(OperationType opType)
    {
        // Channel-first operations (Conv, BatchNorm, Pooling) intrinsically prefer NCHW
        // regardless of the target hardware layout
        if (ChannelFirstOps.Contains(opType))
        {
            return "NCHW";
        }

        // Most other operations are layout-agnostic
        return "AGNOSTIC";
    }

    /// <summary>
    /// Determines whether a node requires layout conversion based on mismatched input layouts.
    /// </summary>
    /// <param name="node">The node to check.</param>
    /// <param name="layoutInfo">Dictionary of node layout preferences.</param>
    /// <returns>True if the node requires layout conversion for any of its inputs.</returns>
    private bool RequiresLayoutConversion(OptimizationNode<T> node, Dictionary<OptimizationNode<T>, string> layoutInfo)
    {
        // Skip if this node has no layout preference
        if (!layoutInfo.TryGetValue(node, out var nodeLayout) || nodeLayout == "AGNOSTIC")
        {
            return false;
        }

        // Check if any input has a different non-agnostic layout
        return node.Inputs.Any(input =>
            layoutInfo.TryGetValue(input, out var inputLayout) &&
            inputLayout != "AGNOSTIC" &&
            inputLayout != nodeLayout);
    }

    /// <summary>
    /// Inserts transpose operations between mismatched layout boundaries.
    /// </summary>
    /// <param name="graph">The optimization graph to modify.</param>
    /// <param name="node">The node requiring layout conversion.</param>
    /// <param name="layoutInfo">Dictionary of node layout preferences.</param>
    /// <returns>True if any transpose operations were inserted; false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// This method inserts a transpose node between each input with a mismatched layout
    /// and the target node. The transpose node converts the tensor from the input's
    /// layout to the target node's preferred layout.
    /// </para>
    /// <para>
    /// For NCHW to NHWC conversion, the permutation is [0, 2, 3, 1].
    /// For NHWC to NCHW conversion, the permutation is [0, 3, 1, 2].
    /// </para>
    /// </remarks>
    private bool InsertLayoutConversion(
        IOptimizationGraph<T> graph,
        OptimizationNode<T> node,
        Dictionary<OptimizationNode<T>, string> layoutInfo)
    {
        bool inserted = false;

        if (!layoutInfo.TryGetValue(node, out var nodeLayout))
        {
            return false;
        }

        // Find all inputs with mismatched layouts
        var inputsToConvert = node.Inputs
            .Where(input =>
                layoutInfo.TryGetValue(input, out var inputLayout) &&
                inputLayout != "AGNOSTIC" &&
                inputLayout != nodeLayout)
            .ToList();

        foreach (var input in inputsToConvert)
        {
            var inputLayout = layoutInfo[input];

            // Create transpose node for layout conversion
            var transposeNode = new OptimizationNode<T>
            {
                OperationType = OperationType.Transpose,
                Name = $"{input.Name}_to_{nodeLayout}",
                OutputShape = ComputeTransposedShape(input.OutputShape, inputLayout, nodeLayout),
                Metadata = new Dictionary<string, object>
                {
                    ["LayoutConversion"] = true,
                    ["SourceLayout"] = inputLayout,
                    ["TargetLayout"] = nodeLayout,
                    ["Permutation"] = GetLayoutPermutation(inputLayout, nodeLayout)
                }
            };

            // Wire up the transpose node:
            // 1. Transpose receives input from the original source
            transposeNode.AddInput(input);

            // 2. Replace the input on the target node with the transpose output
            // Note: ReplaceInput handles the edge removal internally via oldInput.Outputs.Remove(this)
            node.ReplaceInput(input, transposeNode);

            // 4. Add the transpose node to the graph
            graph.AddNode(transposeNode);

            inserted = true;
        }

        return inserted;
    }

    /// <summary>
    /// Computes the output shape after layout transposition.
    /// </summary>
    /// <param name="inputShape">The input tensor shape.</param>
    /// <param name="sourceLayout">The source layout.</param>
    /// <param name="targetLayout">The target layout.</param>
    /// <returns>The transposed output shape, or the original shape if not exactly 4D.</returns>
    /// <remarks>
    /// <para>
    /// Layout conversion (NCHW ↔ NHWC) only applies to exactly 4D tensors.
    /// Non-4D tensors are returned unchanged since they don't follow the
    /// NCHW/NHWC layout conventions.
    /// </para>
    /// </remarks>
    private int[] ComputeTransposedShape(int[] inputShape, string sourceLayout, string targetLayout)
    {
        // Layout conversion only applies to exactly 4D tensors (NCHW/NHWC format)
        // Non-4D tensors are returned unchanged
        if (inputShape == null || inputShape.Length != 4)
        {
            return inputShape ?? Array.Empty<int>();
        }

        var permutation = GetLayoutPermutation(sourceLayout, targetLayout);
        var outputShape = new int[4];

        for (int i = 0; i < 4; i++)
        {
            outputShape[i] = inputShape[permutation[i]];
        }

        return outputShape;
    }

    /// <summary>
    /// Gets the permutation array for converting between layouts.
    /// </summary>
    /// <param name="sourceLayout">The source layout.</param>
    /// <param name="targetLayout">The target layout.</param>
    /// <returns>The permutation array for the transpose operation.</returns>
    /// <exception cref="NotSupportedException">
    /// Thrown when the layout conversion is not supported (different layouts that aren't NCHW/NHWC).
    /// </exception>
    /// <remarks>
    /// <para>Supported conversions:</para>
    /// <list type="bullet">
    /// <item><description>NCHW → NHWC: permutation [0, 2, 3, 1]</description></item>
    /// <item><description>NHWC → NCHW: permutation [0, 3, 1, 2]</description></item>
    /// <item><description>Same layout: identity permutation [0, 1, 2, 3]</description></item>
    /// </list>
    /// </remarks>
    private int[] GetLayoutPermutation(string sourceLayout, string targetLayout)
    {
        // NCHW indices: N=0, C=1, H=2, W=3
        // NHWC indices: N=0, H=1, W=2, C=3

        if (sourceLayout == targetLayout)
        {
            // Same layout - identity permutation (no-op)
            return new[] { 0, 1, 2, 3 };
        }

        if (sourceLayout == "NCHW" && targetLayout == "NHWC")
        {
            // NCHW -> NHWC: Move C from position 1 to position 3
            return new[] { 0, 2, 3, 1 };
        }

        if (sourceLayout == "NHWC" && targetLayout == "NCHW")
        {
            // NHWC -> NCHW: Move C from position 3 to position 1
            return new[] { 0, 3, 1, 2 };
        }

        // Unsupported conversion - fail fast to catch configuration errors
        throw new NotSupportedException(
            $"Layout conversion from '{sourceLayout}' to '{targetLayout}' is not supported. " +
            "Supported conversions: NCHW <-> NHWC.");
    }

    /// <inheritdoc/>
    public override bool CanApply(IOptimizationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.Convolution ||
                                   n.OperationType == OperationType.Convolution2D ||
                                   n.OperationType == OperationType.Conv2D);
    }
}
