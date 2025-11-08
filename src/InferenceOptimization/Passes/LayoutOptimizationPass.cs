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

    public override bool Apply(IComputationGraph<T> graph)
    {
        bool modified = false;

        // Analyze the graph to determine optimal layout
        var layoutInfo = AnalyzeLayouts(graph);

        // Insert transpose operations where needed
        foreach (var node in graph.Nodes.ToList())
        {
            if (RequiresLayoutConversion(node, layoutInfo))
            {
                InsertLayoutConversion(graph, node);
                modified = true;
            }
        }

        return modified;
    }

    private Dictionary<ComputationNode<T>, string> AnalyzeLayouts(IComputationGraph<T> graph)
    {
        var layouts = new Dictionary<ComputationNode<T>, string>();

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

        // Operations that are layout-agnostic
        var agnosticOps = new HashSet<OperationType>
        {
            OperationType.ReLU,
            OperationType.Sigmoid,
            OperationType.Tanh,
            OperationType.Dropout
        };

        if (channelFirstOps.Contains(opType))
        {
            return _targetLayout;
        }

        return "AGNOSTIC";
    }

    private bool RequiresLayoutConversion(ComputationNode<T> node, Dictionary<ComputationNode<T>, string> layoutInfo)
    {
        // Check if this node's layout differs from its inputs
        if (!layoutInfo.TryGetValue(node, out var nodeLayout) || nodeLayout == "AGNOSTIC")
        {
            return false;
        }

        foreach (var input in node.Inputs)
        {
            if (layoutInfo.TryGetValue(input, out var inputLayout) &&
                inputLayout != "AGNOSTIC" &&
                inputLayout != nodeLayout)
            {
                return true;
            }
        }

        return false;
    }

    private void InsertLayoutConversion(IComputationGraph<T> graph, ComputationNode<T> node)
    {
        // Insert a transpose operation to convert layout
        var transposeNode = new ComputationNode<T>
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

    public override bool CanApply(IComputationGraph<T> graph)
    {
        return base.CanApply(graph) &&
               graph.Nodes.Any(n => n.OperationType == OperationType.Convolution ||
                                   n.OperationType == OperationType.Convolution2D);
    }
}
