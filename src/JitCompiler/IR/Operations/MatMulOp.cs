namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents matrix multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// Corresponds to TensorOperations<T>.MatrixMultiply().
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
