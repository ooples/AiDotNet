namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise square root in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Sqrt().
/// Computes square root for each element: result[i] = √a[i].
/// </para>
/// <para><b>For Beginners:</b> Calculates the square root of each element.
///
/// Example:
/// sqrt([1, 4, 9, 16]) = [1, 2, 3, 4]
/// </para>
/// </remarks>
public class SqrtOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
