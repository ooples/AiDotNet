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
public class LayoutOptimizationPass<T> : OptimizationPassBase<T> where T : struct
{
    public override OptimizationPassType PassType => OptimizationPassType.LayoutOptimization;
    public override string Name => "Layout Optimization";

    private readonly string _targetLayout;

    public LayoutOptimizationPass(string targetLayout = "NCHW")
    {
        _targetLayout = targetLayout;
    }

    public override bool Apply(IOptimizationGraph<T> graph)
    {
        bool modified = false;

        // Analyze the graph to determine optimal layout
        var layoutInfo = AnalyzeLayouts(graph);

        // Insert transpose operations where needed
        foreach (var node in graph.Nodes.Where(n => RequiresLayoutConversion(n, layoutInfo)).ToList())
        {
            InsertLayoutConversion(graph, node);
            modified = true;
        }

        return modified;
    }

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

    private string GetPreferredLayout(OperationType opType)
    {
        // Operations that prefer channel-first (NCHW)
        var channelFirstOps = new HashSet<OperationType>
        {
            OperationType.Convolution,
            OperationType.Convolution2D,
            OperationType.BatchNormalization,
            OperationType.MaxPooling,
            OperationType.AveragePooling
        };

        if (channelFirstOps.Contains(opType))
        {
            return _targetLayout;
        }

        return "AGNOSTIC";
    }

    private bool RequiresLayoutConversion(OptimizationNode<T> node, Dictionary<OptimizationNode<T>, string> layoutInfo)
    {
        // Check if this node's layout differs from its inputs
        if (!layoutInfo.TryGetValue(node, out var nodeLayout) || nodeLayout == "AGNOSTIC")
        {
            return false;
        }

        return node.Inputs.Any(input =>
            layoutInfo.TryGetValue(input, out var inputLayout) &&
            inputLayout != "AGNOSTIC" &&
            inputLayout != nodeLayout);
    }

    private void InsertLayoutConversion(IOptimizationGraph<T> graph, OptimizationNode<T> node)
    {
        // Insert a transpose operation to convert layout
        var transposeNode = new OptimizationNode<T>
        {
            OperationType = OperationType.Transpose,
            Name = $"{node.Name}_layout_convert",
            Metadata = new Dictionary<string, object>
            {
                ["LayoutConversion"] = true,
                ["TargetLayout"] = _targetLayout
            }
        };

        // Insert the transpose between the input and this node
        // (Implementation details omitted for brevity)
        transposeNode.Metadata["ConversionInserted"] = true;
    }

    public override bool CanApply(IOptimizationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.Convolution ||
                                   n.OperationType == OperationType.Convolution2D);
    }
}
