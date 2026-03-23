using AiDotNet.Enums;

namespace AiDotNet.Attributes;

/// <summary>
/// Declares mathematical properties of a loss function for automatic test generation and cataloging.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class LossPropertyAttribute : Attribute
{
    /// <summary>Whether the loss is always ≥ 0. Default: true.</summary>
    public bool IsNonNegative { get; set; } = true;

    /// <summary>Whether L(x,x) == 0. Default: true.</summary>
    public bool ZeroForIdentical { get; set; } = true;

    /// <summary>Whether L(x,y) == L(y,x). Default: false.</summary>
    public bool IsSymmetric { get; set; }

    /// <summary>Whether inputs must be in [0,1]. Default: false.</summary>
    public bool RequiresProbabilityInputs { get; set; }

    /// <summary>Whether per-class weights are supported. Default: false.</summary>
    public bool SupportsClassWeights { get; set; }

    /// <summary>Whether designed for imbalanced data (Focal, Dice). Default: false.</summary>
    public bool HandlesImbalancedData { get; set; }

    /// <summary>Whether robust to outliers (Huber, MAE). Default: false.</summary>
    public bool IsRobustToOutliers { get; set; }

    /// <summary>Expected output format. Default: Continuous.</summary>
    public OutputType ExpectedOutput { get; set; } = OutputType.Continuous;

    /// <summary>
    /// The method signature shape this loss uses for its primary calculation.
    /// The test scaffold generator uses this to select the correct test base class.
    /// Default: VectorVector (standard CalculateLoss(Vector, Vector) interface).
    /// </summary>
    public LossApiShape ApiShape { get; set; } = LossApiShape.VectorVector;

    /// <summary>
    /// The format of test data that this loss function expects.
    /// The test base class uses this to generate appropriate test vectors.
    /// Default: Continuous (standard [0,1] range values).
    /// </summary>
    public LossTestInputFormat TestInputFormat { get; set; } = LossTestInputFormat.Continuous;
}

/// <summary>
/// Specifies the category of a loss function.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class LossCategoryAttribute : Attribute
{
    /// <summary>The category.</summary>
    public LossCategory Category { get; }
    /// <summary>Creates a new LossCategory attribute.</summary>
    public LossCategoryAttribute(LossCategory category) => Category = category;
}

/// <summary>
/// Specifies a task this loss function is designed for.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class LossTaskAttribute : Attribute
{
    /// <summary>The task.</summary>
    public LossTask Task { get; }
    /// <summary>Creates a new LossTask attribute.</summary>
    public LossTaskAttribute(LossTask task) => Task = task;
}
