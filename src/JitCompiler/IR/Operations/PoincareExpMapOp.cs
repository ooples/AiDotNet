namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents the exponential map in the Poincare ball model in the IR.
/// </summary>
/// <remarks>
/// <para>
/// The exponential map projects a tangent vector at a base point onto the manifold.
/// It's essential for optimization in hyperbolic space (moving along geodesics).
/// </para>
/// <para><b>For Beginners:</b> When training hyperbolic neural networks, we need
/// to move points in a curved space. The exponential map converts a "direction"
/// (tangent vector) into an actual movement along the curved surface.
/// </para>
/// </remarks>
public class PoincareExpMapOp : IROp
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
        // Inputs: base point, tangent vector
        if (InputIds.Length != 2) return false;
        if (Curvature >= 0) return false;
        return true;
    }
}
