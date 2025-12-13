using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.IR.Common;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.InferenceOptimization.IR.HighLevel;

/// <summary>
/// High-Level Intermediate Representation Node.
/// Represents semantic operations at the model level, similar to TVM's Relay or MLIR's high-level dialects.
/// </summary>
/// <remarks>
/// <para><b>Design Philosophy:</b></para>
/// <para>
/// HLIR represents operations at a semantic level, preserving model structure and enabling
/// high-level optimizations like:
/// - Operator fusion (Conv+BN+ReLU)
/// - Algebraic simplification
/// - Common subexpression elimination
/// - Dead code elimination
/// - Constant folding
/// </para>
///
/// <para><b>Industry Comparison:</b></para>
/// <list type="bullet">
/// <item>TVM Relay: Functional IR with let bindings and closures - we add richer metadata</item>
/// <item>MLIR High-Level: Dialect-based with regions - we add optimization hints</item>
/// <item>XLA HLO: Flat operation list - we add graph structure</item>
/// <item>ONNX: Static graph - we add dynamic shape support and fusion tracking</item>
/// </list>
///
/// <para><b>Exceeds Standards By:</b></para>
/// <list type="bullet">
/// <item>Combining graph-based and SSA-style representations</item>
/// <item>Rich provenance tracking for debugging</item>
/// <item>Built-in fusion pattern matching</item>
/// <item>Hardware-aware cost model hints</item>
/// <item>Full quantization metadata support</item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type for constant values</typeparam>
public class HLIRNode<T> where T : struct
{
    #region Identity

    /// <summary>
    /// Unique identifier for this node. Uses integer for fast lookup (like JitCompiler).
    /// </summary>
    public int Id { get; set; }

    /// <summary>
    /// Human-readable name for debugging and visualization.
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// The semantic operation type.
    /// </summary>
    public OperationType Operation { get; set; }

    #endregion

    #region Graph Structure

    /// <summary>
    /// Input nodes (predecessor edges). Using node references for graph traversal.
    /// </summary>
    public List<HLIRNode<T>> Inputs { get; set; } = new();

    /// <summary>
    /// Output nodes (successor edges). Maintained bidirectionally for efficient traversal.
    /// </summary>
    public List<HLIRNode<T>> Outputs { get; set; } = new();

    /// <summary>
    /// Input tensor IDs for SSA-style representation (compatible with JitCompiler).
    /// </summary>
    public int[] InputIds { get; set; } = Array.Empty<int>();

    #endregion

    #region Type Information

    /// <summary>
    /// Comprehensive type information for the output tensor.
    /// </summary>
    public TensorType OutputType { get; set; } = new();

    /// <summary>
    /// Types of each input (for validation and type inference).
    /// </summary>
    public List<TensorType> InputTypes { get; set; } = new();

    #endregion

    #region Operation Parameters

    /// <summary>
    /// Operation-specific parameters (weights, biases, etc.).
    /// Key: parameter name, Value: parameter value or tensor.
    /// </summary>
    public Dictionary<string, object> Parameters { get; set; } = new();

    /// <summary>
    /// Operation attributes (stride, padding, kernel size, etc.).
    /// </summary>
    public Dictionary<string, object> Attributes { get; set; } = new();

    /// <summary>
    /// Constant tensor value (if this is a constant node).
    /// </summary>
    public Tensor<T>? ConstantValue { get; set; }

    #endregion

    #region Optimization Metadata

    /// <summary>
    /// Whether this node can be eliminated (no side effects).
    /// </summary>
    public bool CanEliminate { get; set; } = true;

    /// <summary>
    /// Whether this node can perform in-place operations.
    /// </summary>
    public bool CanOperateInPlace { get; set; }

    /// <summary>
    /// Whether this node has been fused from multiple operations.
    /// </summary>
    public bool IsFused { get; set; }

    /// <summary>
    /// Original nodes that were fused into this one.
    /// </summary>
    public List<HLIRNode<T>>? FusedFrom { get; set; }

    /// <summary>
    /// Marked for deletion during optimization passes.
    /// </summary>
    public bool IsMarkedForDeletion { get; set; }

    /// <summary>
    /// Cost estimate for scheduling (FLOPs, memory access, etc.).
    /// </summary>
    public OperationCost? Cost { get; set; }

    /// <summary>
    /// Optimization hints for passes.
    /// </summary>
    public OptimizationHints Hints { get; set; } = new();

    #endregion

    #region Provenance (Debugging)

    /// <summary>
    /// Reference to the original layer (for debugging).
    /// </summary>
    public object? OriginalLayer { get; set; }

    /// <summary>
    /// Source location information for debugging.
    /// </summary>
    public SourceLocation? SourceLocation { get; set; }

    /// <summary>
    /// Provenance chain showing how this node was derived.
    /// </summary>
    public List<string> Provenance { get; set; } = new();

    #endregion

    #region Methods

    /// <summary>
    /// Adds an input node with bidirectional linking.
    /// </summary>
    public void AddInput(HLIRNode<T> input)
    {
        if (!Inputs.Contains(input))
        {
            Inputs.Add(input);
            InputTypes.Add(input.OutputType.Clone());
        }
        if (!input.Outputs.Contains(this))
        {
            input.Outputs.Add(this);
        }
    }

    /// <summary>
    /// Removes an input node with bidirectional unlinking.
    /// </summary>
    public void RemoveInput(HLIRNode<T> input)
    {
        var index = Inputs.IndexOf(input);
        if (index >= 0)
        {
            Inputs.RemoveAt(index);
            if (index < InputTypes.Count)
            {
                InputTypes.RemoveAt(index);
            }
        }
        input.Outputs.Remove(this);
    }

    /// <summary>
    /// Replaces an input node with another.
    /// </summary>
    public void ReplaceInput(HLIRNode<T> oldInput, HLIRNode<T> newInput)
    {
        var index = Inputs.IndexOf(oldInput);
        if (index >= 0)
        {
            Inputs[index] = newInput;
            if (index < InputTypes.Count)
            {
                InputTypes[index] = newInput.OutputType.Clone();
            }
            oldInput.Outputs.Remove(this);
            if (!newInput.Outputs.Contains(this))
            {
                newInput.Outputs.Add(this);
            }
        }
    }

    /// <summary>
    /// Checks if this node has any consumers.
    /// </summary>
    public bool HasConsumers => Outputs.Count > 0;

    /// <summary>
    /// Gets the number of consumers.
    /// </summary>
    public int ConsumerCount => Outputs.Count;

    /// <summary>
    /// Performs type inference based on operation and inputs.
    /// </summary>
    public void InferOutputType()
    {
        // Default implementation - specific operations override
        if (InputTypes.Count > 0)
        {
            OutputType = InputTypes[0].Clone();
        }
    }

    /// <summary>
    /// Validates this node's structure.
    /// </summary>
    public bool Validate()
    {
        // Basic validation
        if (Id < 0) return false;
        if (OutputType == null) return false;

        // Input/output consistency
        foreach (var input in Inputs)
        {
            if (!input.Outputs.Contains(this)) return false;
        }

        foreach (var output in Outputs)
        {
            if (!output.Inputs.Contains(this)) return false;
        }

        return true;
    }

    /// <summary>
    /// Creates a deep copy of this node (without connections).
    /// </summary>
    public HLIRNode<T> Clone()
    {
        return new HLIRNode<T>
        {
            Id = -1, // New ID should be assigned by graph
            Name = Name + "_clone",
            Operation = Operation,
            OutputType = OutputType.Clone(),
            Parameters = new Dictionary<string, object>(Parameters),
            Attributes = new Dictionary<string, object>(Attributes),
            ConstantValue = ConstantValue,
            CanEliminate = CanEliminate,
            CanOperateInPlace = CanOperateInPlace,
            IsFused = IsFused,
            OriginalLayer = OriginalLayer,
            Hints = Hints.Clone()
        };
    }

    /// <summary>
    /// Adds provenance information.
    /// </summary>
    public void AddProvenance(string info)
    {
        Provenance.Add($"[{DateTime.UtcNow:HH:mm:ss}] {info}");
    }

    public override string ToString()
    {
        var inputStr = Inputs.Count > 0
            ? string.Join(", ", Inputs.Select(i => $"n{i.Id}"))
            : "none";
        return $"n{Id}: {Name} ({Operation}) [{OutputType}] <- ({inputStr})";
    }

    #endregion
}

/// <summary>
/// Cost estimate for an operation (for scheduling and optimization).
/// </summary>
public class OperationCost
{
    /// <summary>
    /// Estimated FLOPs (floating-point operations).
    /// </summary>
    public long FLOPs { get; set; }

    /// <summary>
    /// Estimated memory read in bytes.
    /// </summary>
    public long MemoryRead { get; set; }

    /// <summary>
    /// Estimated memory write in bytes.
    /// </summary>
    public long MemoryWrite { get; set; }

    /// <summary>
    /// Arithmetic intensity (FLOPs / memory bytes).
    /// Higher means more compute-bound, lower means more memory-bound.
    /// </summary>
    public double ArithmeticIntensity =>
        (MemoryRead + MemoryWrite) > 0
            ? (double)FLOPs / (MemoryRead + MemoryWrite)
            : 0;

    /// <summary>
    /// Whether this operation is likely memory-bound.
    /// </summary>
    public bool IsMemoryBound => ArithmeticIntensity < 10;

    /// <summary>
    /// Estimated latency in nanoseconds (device-specific).
    /// </summary>
    public long EstimatedLatencyNs { get; set; }
}

/// <summary>
/// Optimization hints for passes.
/// </summary>
public class OptimizationHints
{
    /// <summary>
    /// Preferred device for execution.
    /// </summary>
    public DeviceType PreferredDevice { get; set; } = DeviceType.Auto;

    /// <summary>
    /// Whether to prioritize memory efficiency.
    /// </summary>
    public bool PrioritizeMemory { get; set; }

    /// <summary>
    /// Whether to prioritize latency.
    /// </summary>
    public bool PrioritizeLatency { get; set; }

    /// <summary>
    /// Whether this node is a good fusion candidate.
    /// </summary>
    public bool IsFusionCandidate { get; set; } = true;

    /// <summary>
    /// Custom scheduling priority (higher = earlier).
    /// </summary>
    public int SchedulingPriority { get; set; }

    /// <summary>
    /// Tile sizes for tiled execution.
    /// </summary>
    public int[]? TileSizes { get; set; }

    /// <summary>
    /// Whether to use vectorization.
    /// </summary>
    public bool EnableVectorization { get; set; } = true;

    /// <summary>
    /// Whether to use parallelization.
    /// </summary>
    public bool EnableParallelization { get; set; } = true;

    public OptimizationHints Clone() => new()
    {
        PreferredDevice = PreferredDevice,
        PrioritizeMemory = PrioritizeMemory,
        PrioritizeLatency = PrioritizeLatency,
        IsFusionCandidate = IsFusionCandidate,
        SchedulingPriority = SchedulingPriority,
        TileSizes = TileSizes != null ? (int[])TileSizes.Clone() : null,
        EnableVectorization = EnableVectorization,
        EnableParallelization = EnableParallelization
    };
}

/// <summary>
/// Source location for debugging.
/// </summary>
public class SourceLocation
{
    public string? FileName { get; set; }
    public int Line { get; set; }
    public int Column { get; set; }
    public string? FunctionName { get; set; }

    public override string ToString() =>
        $"{FileName ?? "unknown"}:{Line}:{Column} in {FunctionName ?? "unknown"}";
}
