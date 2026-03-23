using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Declares mathematical properties of an activation function for automatic test generation
/// and cataloging. The source generator reads these to configure invariant test parameters.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class ActivationPropertyAttribute : Attribute
{
    /// <summary>Whether the activation is monotonically non-decreasing. Default: true.</summary>
    public bool IsMonotonic { get; set; } = true;

    /// <summary>Whether Activate(0) produces exactly 0. Default: true.</summary>
    public bool ZeroPreserving { get; set; } = true;

    /// <summary>Whether the output is bounded. Default: false.</summary>
    public bool IsBounded { get; set; }

    /// <summary>Whether this operates on vectors (Softmax) vs scalars (ReLU). Default: false.</summary>
    public bool IsVectorActivation { get; set; }

    /// <summary>Whether it has learnable parameters (PReLU). Default: false.</summary>
    public bool HasLearnableParameters { get; set; }

    /// <summary>Whether it's differentiable everywhere. Default: true.</summary>
    public bool IsDifferentiable { get; set; } = true;

    /// <summary>Relative computational cost. Default: Medium.</summary>
    public ComputeCost Cost { get; set; } = ComputeCost.Medium;
}

/// <summary>
/// Specifies the architectural category of an activation function.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class ActivationCategoryAttribute : Attribute
{
    /// <summary>The category.</summary>
    public ActivationCategory Category { get; }
    /// <summary>Creates a new ActivationCategory attribute.</summary>
    public ActivationCategoryAttribute(ActivationCategory category) => Category = category;
}

/// <summary>
/// Specifies a task where this activation function is commonly used.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class ActivationTaskAttribute : Attribute
{
    /// <summary>The task.</summary>
    public ActivationTask Task { get; }
    /// <summary>Creates a new ActivationTask attribute.</summary>
    public ActivationTaskAttribute(ActivationTask task) => Task = task;
}
