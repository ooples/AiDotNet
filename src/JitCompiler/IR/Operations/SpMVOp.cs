namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents sparse matrix-vector multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// SpMV computes y = A * x where A is a sparse matrix and x is a dense vector.
/// This is a fundamental operation for graph neural networks and sparse models.
/// </para>
/// <para><b>For Beginners:</b> Sparse matrices have mostly zero values. Instead of
/// storing all those zeros, we only store the non-zero elements. SpMV multiplies
/// such a sparse matrix by a regular (dense) vector efficiently.
/// </para>
/// </remarks>
public class SpMVOp : IROp
{
    /// <summary>
    /// Gets or sets the number of rows in the sparse matrix.
    /// </summary>
    public int Rows { get; set; }

    /// <summary>
    /// Gets or sets the number of columns in the sparse matrix.
    /// </summary>
    public int Columns { get; set; }

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
        // Inputs: sparse matrix (represented as row_ptr, col_idx, values), dense vector
        if (InputIds.Length != 4) return false;
        if (Rows <= 0 || Columns <= 0) return false;
        return true;
    }
}
