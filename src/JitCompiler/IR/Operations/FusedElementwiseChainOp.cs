namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Fused chain of element-wise operations.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Combines multiple element-wise ops into one.
///
/// Instead of:
///   t1 = Add(a, b)
///   t2 = ReLU(t1)
///   t3 = Multiply(t2, c)
///
/// One fused operation processes all three steps together.
/// Saves memory by not storing intermediate results.
/// </para>
/// </remarks>
public class FusedElementwiseChainOp : IROp
{
    /// <summary>Gets or sets the list of operation names in the chain.</summary>
    public List<string> OperationNames { get; set; } = new List<string>();

    /// <summary>Validates the operation chain.</summary>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (OperationNames.Count < 2) return false;
        return true;
    }
}
