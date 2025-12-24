namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents the geometric product of two multivectors in the IR.
/// </summary>
/// <remarks>
/// <para>
/// The geometric product is the fundamental operation in Clifford/geometric algebra.
/// It combines the inner product and outer product: AB = A·B + A∧B.
/// </para>
/// <para><b>For Beginners:</b> The geometric product is a way to multiply
/// multivectors (objects that can represent scalars, vectors, planes, and volumes).
/// It's more powerful than the dot product or cross product because it preserves
/// all geometric information.
/// </para>
/// </remarks>
public class GeometricProductOp : IROp
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
        // Inputs: two multivectors
        if (InputIds.Length != 2) return false;
        // Signature must be non-negative
        if (PositiveSignature < 0 || NegativeSignature < 0 || ZeroSignature < 0) return false;
        return true;
    }
}
