namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents the gradient of sparse matrix-matrix multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// For SpMM C = A * B:
/// - Gradient w.r.t. B: d_B = A^T * d_C
/// - Gradient w.r.t. A (sparse): d_A[i,j] = sum_k(d_C[i,k] * B[j,k]) for non-zero positions
/// </para>
/// </remarks>
public class GradSpMMOp : IROp
{
    /// <summary>
    /// Gets or sets the number of rows in the sparse matrix.
    /// </summary>
    public int SparseRows { get; set; }

    /// <summary>
    /// Gets or sets the number of columns in the sparse matrix.
    /// </summary>
    public int SparseColumns { get; set; }

    /// <summary>
    /// Gets or sets the number of columns in the dense matrix.
    /// </summary>
    public int DenseColumns { get; set; }

    /// <summary>
    /// Validates that this operation is correctly formed.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: sparse matrix components, dense matrix, gradient of output
        if (InputIds.Length != 5) return false;
        return true;
    }
}
