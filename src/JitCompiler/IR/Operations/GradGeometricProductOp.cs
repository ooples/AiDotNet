namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents the gradient of the geometric product in the IR.
/// </summary>
/// <remarks>
/// <para>
/// For geometric product C = A * B, the gradients are:
/// - d_A = d_C * B~ (right multiplication by reverse of B)
/// - d_B = A~ * d_C (left multiplication by reverse of A)
/// where ~ denotes the multivector reverse operation.
/// </para>
/// </remarks>
public class GradGeometricProductOp : IROp
{
    /// <summary>
    /// Gets or sets the number of positive-signature basis vectors (p).
    /// </summary>
    public int PositiveSignature { get; set; }

    /// <summary>
    /// Gets or sets the number of negative-signature basis vectors (q).
    /// </summary>
    public int NegativeSignature { get; set; }

    /// <summary>
    /// Gets or sets the number of zero-signature basis vectors (r).
    /// </summary>
    public int ZeroSignature { get; set; }

    /// <summary>
    /// Validates that this operation is correctly formed.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: left operand, right operand, gradient of output
        if (InputIds.Length != 3) return false;
        // Validate signature properties (dimension = p + q + r must be positive)
        if (PositiveSignature < 0 || NegativeSignature < 0 || ZeroSignature < 0) return false;
        if (PositiveSignature + NegativeSignature + ZeroSignature <= 0) return false;
        return true;
    }
}
