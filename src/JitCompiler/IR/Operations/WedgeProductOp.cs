namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents the wedge (outer) product of two multivectors in the IR.
/// </summary>
/// <remarks>
/// <para>
/// The wedge product creates higher-dimensional objects from lower ones:
/// - Vector ∧ Vector → Bivector (oriented area)
/// - Vector ∧ Bivector → Trivector (oriented volume)
/// </para>
/// <para><b>For Beginners:</b> The wedge product is like the cross product but
/// more general. It measures "how much" two objects span together in space.
/// </para>
/// </remarks>
public class WedgeProductOp : IROp
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
        if (InputIds.Length != 2) return false;
        if (PositiveSignature < 0 || NegativeSignature < 0 || ZeroSignature < 0) return false;
        return true;
    }
}
