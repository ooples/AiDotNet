namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents the logarithmic map in the Poincare ball model in the IR.
/// </summary>
/// <remarks>
/// <para>
/// The logarithmic map is the inverse of the exponential map. It projects a point
/// back to the tangent space at a base point. Essential for computing directions
/// and distances in hyperbolic space.
/// </para>
/// <para><b>For Beginners:</b> This is the reverse of the exponential map.
/// Given two points in hyperbolic space, the log map tells you the "direction"
/// (tangent vector) you'd need to go from one to the other.
/// </para>
/// </remarks>
public class PoincareLogMapOp : IROp
{
    /// <summary>
    /// Gets or sets the curvature parameter.
    /// </summary>
    public double Curvature { get; set; } = -1.0;

    /// <summary>
    /// Validates that this operation is correctly formed.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: base point, target point
        if (InputIds.Length != 2) return false;
        if (Curvature >= 0) return false;
        return true;
    }
}
