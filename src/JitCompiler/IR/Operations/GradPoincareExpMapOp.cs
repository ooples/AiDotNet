namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents the gradient of the Poincare exponential map in the IR.
/// </summary>
/// <remarks>
/// <para>
/// The exponential map gradient is used in Riemannian gradient descent
/// for hyperbolic neural networks. It computes how changes in the tangent
/// vector affect the resulting point on the manifold.
/// </para>
/// </remarks>
public class GradPoincareExpMapOp : IROp
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
        // Inputs: base point, tangent vector, gradient of output
        if (InputIds.Length != 3) return false;
        return true;
    }
}
