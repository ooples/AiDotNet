namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents matrix transpose in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.Transpose().
/// Transposes a matrix: swaps rows and columns.
/// </para>
/// <para><b>For Beginners:</b> Flips a matrix along its diagonal.
///
/// Example:
/// [[1, 2, 3],     [[1, 4],
///  [4, 5, 6]]  →   [2, 5],
///                  [3, 6]]
///
/// Shape changes from [2, 3] to [3, 2].
///
/// Common in matrix math and backpropagation.
/// </para>
/// </remarks>
public class TransposeOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 1) return false;
        return true;
    }
}
