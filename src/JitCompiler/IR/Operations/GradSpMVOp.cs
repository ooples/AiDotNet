namespace AiDotNet.JitCompiler.IR.Operations;

/// <summary>
/// Represents the gradient of sparse matrix-vector multiplication in the IR.
/// </summary>
/// <remarks>
/// <para>
/// For SpMV y = A * x:
/// - Gradient w.r.t. x: d_x = A^T * d_y
/// - Gradient w.r.t. A (sparse): d_A[i,j] = d_y[i] * x[j] for non-zero positions
/// </para>
/// </remarks>
public class GradSpMVOp : IROp
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
    /// Validates that this operation is correctly formed.
    /// </summary>
    /// <returns>True if valid, false otherwise.</returns>
    public override bool Validate()
    {
        if (!base.Validate()) return false;
        // Inputs: sparse matrix components, input vector, gradient of output
        if (InputIds.Length != 5) return false;
        return true;
    }
}
