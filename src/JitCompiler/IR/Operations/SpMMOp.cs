namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents sparse matrix-matrix multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// SpMM computes C = A * B where A is a sparse matrix and B is a dense matrix.
/// This is essential for batched graph neural network operations.
/// </para>
/// <para><b>For Beginners:</b> Like SpMV but the second operand is a matrix instead
/// of a vector. This allows processing multiple feature vectors at once, which
/// is common in neural network training with batches.
/// </para>
/// </remarks>
public class SpMMOp : IROp
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
    /// Gets or sets the number of non-zero elements.
    /// </summary>
    public int NonZeroCount { get; set; }

    /// <summary>
    /// Validates that this operation is correctly formed.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: sparse matrix (row_ptr, col_idx, values), dense matrix
        if (InputIds.Length != 4) return false;
        if (SparseRows <= 0 || SparseColumns <= 0 || DenseColumns <= 0) return false;
        return true;
    }
}
