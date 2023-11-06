namespace AiDotNet.Models;

/// <summary>
/// Options for multiple regression models.
/// </summary>
public class MultipleRegressionOptions : RegressionOptions
{
    /// <summary>
    /// Different types of matrix decomposition. 
    /// 
    /// The decomposition is used to solve linear equations and to calculate the determinant and the inverse of a matrix.
    /// 
    /// The Cholesky decomposition is used for symmetric positive-definite matrices.
    /// The Eigenvalue decomposition is used for non-symmetric matrices.
    /// The Gram-Schmidt decomposition is used for non-symmetric matrices.
    /// The LU decomposition is used for non-symmetric matrices.
    /// The QR decomposition is used for non-symmetric matrices.
    /// The SVD decomposition is used for non-symmetric matrices.
    /// </summary>
    public MatrixDecomposition MatrixDecomposition { get; set; } = MatrixDecomposition.Cholesky;

    /// <summary>
    /// Inserts a column of 1s at the beginning of the matrix. This is used to calculate the y intercept.
    /// </summary>
    public bool UseIntercept { get; set; }

    /// <summary>
    /// Matrix layout will either be column arrays or row arrays. Default is column arrays.
    ///
    /// Column arrays example
    /// { x1, x2, x3 }, { y1, y2, y3 }
    /// 
    /// Row arrays example
    /// { x1, y1 }, { x2, y2 }, { x3, y3 }
    /// </summary>
    public MatrixLayout MatrixLayout { get; set; } = MatrixLayout.ColumnArrays;
}