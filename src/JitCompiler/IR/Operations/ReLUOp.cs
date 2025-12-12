namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents ReLU (Rectified Linear Unit) activation in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.ReLU().
/// Computes max(0, x) for each element: result[i] = max(0, a[i]).
/// </para>
/// <para><b>For Beginners:</b> Keeps positive values, zeros out negative values.
///
/// Example:
/// ReLU([-2, -1, 0, 1, 2]) = [0, 0, 0, 1, 2]
///
/// Very common in neural networks because it's simple and effective.
/// </para>
/// </remarks>
public class ReLUOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
