namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise addition in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Add().
/// Performs element-wise addition of two tensors: result[i] = a[i] + b[i].
/// </para>
/// <para><b>For Beginners:</b> Adds two tensors together, element by element.
///
/// Example:
/// [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
///
/// Supports broadcasting:
/// [1, 2, 3] + 5 = [6, 7, 8]
/// </para>
/// </remarks>
public class AddOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }
}
