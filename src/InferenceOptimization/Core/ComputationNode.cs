using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.InferenceOptimization.Core;

/// <summary>
/// Represents a single operation node in the computation graph.
/// Each node contains operation type, inputs, outputs, and parameters.
/// </summary>
/// <typeparam name="T">The numeric type (double, float, decimal)</typeparam>
public class ComputationNode<T> where T : struct
{
    /// <summary>
    /// Unique identifier for this node in the graph.
    /// </summary>
    public string Id { get; set; }

    /// <summary>
    /// The type of operation this node performs.
    /// </summary>
    public OperationType OperationType { get; set; }

    /// <summary>
    /// Human-readable name for this node (e.g., "conv1", "bn1", "relu1").
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// Input nodes that feed data into this node.
    /// </summary>
    public List<ComputationNode<T>> Inputs { get; set; }

    /// <summary>
    /// Output nodes that consume data from this node.
    /// </summary>
    public List<ComputationNode<T>> Outputs { get; set; }

    /// <summary>
    /// Shape of the output tensor produced by this node.
    /// </summary>
    public int[] OutputShape { get; set; }

    /// <summary>
    /// Parameters associated with this operation (e.g., weights, biases).
    /// </summary>
    public Dictionary<string, object> Parameters { get; set; }

    /// <summary>
    /// Metadata for this node (e.g., stride, padding, kernel size).
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; }

    /// <summary>
    /// Constant value if this is a constant node.
    /// </summary>
    public Tensor<T>? ConstantValue { get; set; }

    /// <summary>
    /// Indicates if this node can be eliminated (e.g., has no side effects).
    /// </summary>
    public bool CanEliminate { get; set; }

    /// <summary>
    /// Indicates if this node can perform in-place operations.
    /// </summary>
    public bool CanOperate InPlace { get; set; }

    /// <summary>
    /// Reference to the original layer (if applicable).
    /// </summary>
    public object? OriginalLayer { get; set; }

    /// <summary>
    /// Indicates if this node has been marked for deletion.
    /// </summary>
    public bool IsMarkedForDeletion { get; set; }

    /// <summary>
    /// Indicates if this node is a fused operation.
    /// </summary>
    public bool IsFused { get; set; }

    /// <summary>
    /// If this is a fused node, contains the original nodes that were fused.
    /// </summary>
    public List<ComputationNode<T>>? FusedFrom { get; set; }

    public ComputationNode()
    {
        Id = Guid.NewGuid().ToString();
        Name = string.Empty;
        Inputs = new List<ComputationNode<T>>();
        Outputs = new List<ComputationNode<T>>();
        OutputShape = Array.Empty<int>();
        Parameters = new Dictionary<string, object>();
        Metadata = new Dictionary<string, object>();
        CanEliminate = true;
        CanOperateInPlace = false;
        IsMarkedForDeletion = false;
        IsFused = false;
    }

    /// <summary>
    /// Adds an input node to this node.
    /// </summary>
    public void AddInput(ComputationNode<T> inputNode)
    {
        if (!Inputs.Contains(inputNode))
        {
            Inputs.Add(inputNode);
        }

        if (!inputNode.Outputs.Contains(this))
        {
            inputNode.Outputs.Add(this);
        }
    }

    /// <summary>
    /// Removes an input node from this node.
    /// </summary>
    public void RemoveInput(ComputationNode<T> inputNode)
    {
        Inputs.Remove(inputNode);
        inputNode.Outputs.Remove(this);
    }

    /// <summary>
    /// Replaces an input node with another node.
    /// </summary>
    public void ReplaceInput(ComputationNode<T> oldInput, ComputationNode<T> newInput)
    {
        var index = Inputs.IndexOf(oldInput);
        if (index >= 0)
        {
            Inputs[index] = newInput;
            oldInput.Outputs.Remove(this);

            if (!newInput.Outputs.Contains(this))
            {
                newInput.Outputs.Add(this);
            }
        }
    }

    /// <summary>
    /// Checks if this node has any consumers (output nodes).
    /// </summary>
    public bool HasConsumers() => Outputs.Count > 0;

    /// <summary>
    /// Gets the number of consumers for this node.
    /// </summary>
    public int ConsumerCount() => Outputs.Count;

    /// <summary>
    /// Creates a deep copy of this node (without connections).
    /// </summary>
    public ComputationNode<T> Clone()
    {
        return new ComputationNode<T>
        {
            Id = Guid.NewGuid().ToString(), // New ID for clone
            OperationType = OperationType,
            Name = Name + "_clone",
            OutputShape = (int[])OutputShape.Clone(),
            Parameters = new Dictionary<string, object>(Parameters),
            Metadata = new Dictionary<string, object>(Metadata),
            ConstantValue = ConstantValue,
            CanEliminate = CanEliminate,
            CanOperateInPlace = CanOperateInPlace,
            OriginalLayer = OriginalLayer,
            IsFused = IsFused
        };
    }

    public override string ToString()
    {
        var inputCount = Inputs.Count;
        var outputCount = Outputs.Count;
        var shape = OutputShape.Length > 0 ? $"[{string.Join(", ", OutputShape)}]" : "[]";
        return $"{Name} ({OperationType}) - Inputs: {inputCount}, Outputs: {outputCount}, Shape: {shape}";
    }
}
