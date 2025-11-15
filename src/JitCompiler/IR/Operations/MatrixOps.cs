namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents matrix multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations.MatrixMultiply().
/// Performs matrix multiplication (dot product): C = A × B.
/// For 2D matrices: C[i,j] = Σ(A[i,k] * B[k,j]).
/// </para>
/// <para><b>For Beginners:</b> Multiplies two matrices together (not element-wise!).
///
/// Example:
/// [2, 3] matrix × [3, 4] matrix = [2, 4] matrix
///
/// This is the standard matrix multiplication from linear algebra.
/// Inner dimensions must match (3 in this example).
///
/// Very common operation in neural networks - used for dense layers.
/// </para>
/// </remarks>
public class MatMulOp : IROp
{
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        if (InputIds.Length != 2) return false;
        return true;
    }
}

/// <summary>
/// Represents matrix transpose in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations.Transpose().
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
