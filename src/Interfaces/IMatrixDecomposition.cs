namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a matrix decomposition that can be used to solve linear systems and invert matrices.
/// </summary>
/// <remarks>
/// Matrix decomposition is a technique that breaks down a complex matrix into simpler components,
/// making it easier to solve mathematical problems like linear equations or matrix inversion.
/// Common decompositions include LU, QR, Cholesky, and SVD (Singular Value Decomposition).
/// </remarks>
/// <typeparam name="T">The numeric data type used in the matrix (e.g., float, double).</typeparam>
public interface IMatrixDecomposition<T>
{
    /// <summary>
    /// Gets the original matrix that was decomposed.
    /// </summary>
    /// <value>
    /// The original matrix used to create this decomposition.
    /// </value>
    Matrix<T> A { get; }

    /// <summary>
    /// Solves a linear system of equations Ax = b, where A is the decomposed matrix.
    /// </summary>
    /// <remarks>
    /// This method efficiently finds the solution vector x without explicitly calculating
    /// the inverse of matrix A, which is computationally expensive and numerically unstable.
    /// </remarks>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    Vector<T> Solve(Vector<T> b);

    /// <summary>
    /// Calculates the inverse of the original matrix A.
    /// </summary>
    /// <remarks>
    /// The inverse of a matrix A is another matrix A⁻¹ such that A × A⁻¹ = I, where I is the identity matrix.
    /// Not all matrices have inverses - only square matrices with non-zero determinants are invertible.
    /// Matrix decompositions provide efficient ways to compute inverses when they exist.
    /// </remarks>
    /// <returns>The inverse of the original matrix A.</returns>
    Matrix<T> Invert();
}