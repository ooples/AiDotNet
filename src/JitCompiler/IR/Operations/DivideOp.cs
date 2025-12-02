namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise division in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Divide().
/// Performs element-wise division: result[i] = a[i] / b[i].
/// </para>
/// <para><b>For Beginners:</b> Divides one tensor by another, element by element.
///
/// Example:
/// [10, 20, 30] / [2, 4, 5] = [5, 5, 6]
/// </para>
/// </remarks>
public class DivideOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }
}
