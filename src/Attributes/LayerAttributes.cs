using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.Attributes;

/// <summary>
/// Declares architectural properties of a neural network layer for cataloging and test generation.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class LayerPropertyAttribute : Attribute
{
    /// <summary>Whether the layer has trainable parameters (weights/biases). Default: true.</summary>
    public bool IsTrainable { get; set; } = true;

    /// <summary>Whether the layer supports backpropagation gradient computation. Default: true.</summary>
    public bool SupportsBackpropagation { get; set; } = true;

    /// <summary>Whether the layer behaves differently during training vs inference (e.g., Dropout, BatchNorm). Default: false.</summary>
    public bool HasTrainingMode { get; set; }

    /// <summary>Expected input rank (number of dimensions). 0 means any rank. Default: 0.</summary>
    public int ExpectedInputRank { get; set; }

    /// <summary>Whether the layer changes the shape of its input. Default: false.</summary>
    public bool ChangesShape { get; set; }

    /// <summary>Whether the layer supports in-place operation (output can alias input). Default: false.</summary>
    public bool SupportsInPlace { get; set; }

    /// <summary>Whether the layer is stateful across forward passes (RNN hidden state, running mean). Default: false.</summary>
    public bool IsStateful { get; set; }

    /// <summary>Relative computational cost. Default: Medium.</summary>
    public ComputeCost Cost { get; set; } = ComputeCost.Medium;
}

/// <summary>
/// Specifies the architectural category of a neural network layer.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class LayerCategoryAttribute : Attribute
{
    /// <summary>The category.</summary>
    public LayerCategory Category { get; }
    /// <summary>Creates a new LayerCategory attribute.</summary>
    public LayerCategoryAttribute(LayerCategory category) => Category = category;
}

/// <summary>
/// Specifies a task that a neural network layer performs.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class LayerTaskAttribute : Attribute
{
    /// <summary>The task.</summary>
    public LayerTask Task { get; }
    /// <summary>Creates a new LayerTask attribute.</summary>
    public LayerTaskAttribute(LayerTask task) => Task = task;
}
