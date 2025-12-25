namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents the gradient of Mobius addition in the IR.
/// </summary>
/// <remarks>
/// <para>
/// The Mobius addition gradient involves computing the Jacobian of the
/// Mobius addition formula with respect to both inputs. This is essential
/// for Riemannian optimization in hyperbolic neural networks.
/// </para>
/// </remarks>
public class GradMobiusAddOp : IROp
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
        // Inputs: x, y, gradient of output
        if (InputIds.Length != 3) return false;
        return true;
    }
}
