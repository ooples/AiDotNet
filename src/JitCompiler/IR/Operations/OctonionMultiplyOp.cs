namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents octonion multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Octonion multiplication is non-associative: (a*b)*c != a*(b*c).
/// This operation takes 16 inputs (8 components for each of 2 octonions)
/// and produces 8 outputs (the product octonion).
/// </para>
/// <para><b>For Beginners:</b> Octonions are 8-dimensional numbers used in
/// advanced neural networks for certain geometric transformations.
/// Their multiplication follows special rules different from regular numbers.
/// </para>
/// </remarks>
public class OctonionMultiplyOp : IROp
{
    /// <summary>
    /// Validates that this operation is correctly formed.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: a_scalar, a_e1..a_e7, b_scalar, b_e1..b_e7 (16 total)
        if (InputIds.Length != 16) return false;
        return true;
    }
}
