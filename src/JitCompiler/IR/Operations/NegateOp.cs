namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents element-wise negation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Negate().
/// Negates each element: result[i] = -a[i].
/// </para>
/// <para><b>For Beginners:</b> Flips the sign of each element.
///
/// Example:
/// -[1, -2, 3] = [-1, 2, -3]
/// </para>
/// </remarks>
public class NegateOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
