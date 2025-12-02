namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise natural logarithm in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Log().
/// Computes natural log for each element: result[i] = ln(a[i]).
/// </para>
/// <para><b>For Beginners:</b> Calculates the natural logarithm of each element.
///
/// Example:
/// log([1, 2.718, 7.389]) ≈ [0, 1, 2]
/// </para>
/// </remarks>
public class LogOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
