namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents Mobius addition in the Poincare ball model in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Mobius addition is the fundamental addition operation in hyperbolic space.
/// For the Poincare ball model with curvature c:
/// x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
/// </para>
/// <para><b>For Beginners:</b> In hyperbolic space, adding two points isn't as simple
/// as regular addition. Mobius addition accounts for the curved nature of the space,
/// ensuring results stay within the valid region (the Poincare ball).
/// </para>
/// </remarks>
public class MobiusAddOp : IROp
{
    /// <summary>
    /// Gets or sets the curvature parameter (typically negative for hyperbolic space).
    /// </summary>
    public double Curvature { get; set; } = -1.0;

    /// <summary>
    /// Validates that this operation is correctly formed.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: two vectors
        if (InputIds.Length != 2) return false;
        // Curvature should be negative for hyperbolic space
        if (Curvature >= 0) return false;
        return true;
    }
}
