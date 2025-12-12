namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise exponential function in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Exp().
/// Computes e^x for each element: result[i] = exp(a[i]).
/// </para>
/// <para><b>For Beginners:</b> Calculates e raised to the power of each element.
///
/// Example:
/// exp([0, 1, 2]) ≈ [1.0, 2.718, 7.389]
/// </para>
/// </remarks>
public class ExpOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
