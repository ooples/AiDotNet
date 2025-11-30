namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise subtraction in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Subtract().
/// Performs element-wise subtraction: result[i] = a[i] - b[i].
/// </para>
/// <para><b>For Beginners:</b> Subtracts one tensor from another, element by element.
///
/// Example:
/// [5, 7, 9] - [1, 2, 3] = [4, 5, 6]
/// </para>
/// </remarks>
public class SubtractOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }
}
