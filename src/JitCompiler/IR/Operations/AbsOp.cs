namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise absolute value in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Abs().
/// Takes the absolute value of each element: result[i] = |a[i]|.
/// </para>
/// <para><b>For Beginners:</b> Makes all values positive (removes the sign).
///
/// Example:
/// |[-1, 2, -3]| = [1, 2, 3]
///
/// This is useful for:
/// - Computing L1 norms (sum of absolute values)
/// - Laplacian kernel calculations (which use L1 distance)
/// - Error magnitude calculations
/// </para>
/// </remarks>
public class AbsOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
