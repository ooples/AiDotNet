namespace AiDotNet.Interfaces;

/// <summary>
/// Represents a matrix decomposition that can be used to solve linear systems and invert matrices.
/// </summary>
/// <remarks>
/// Matrix decomposition is a technique that breaks down a complex matrix into simpler components,
/// making it easier to solve mathematical problems like linear equations or matrix inversion.
/// Common decompositions include LU, QR, Cholesky, and SVD (Singular Value Decomposition).
/// 
/// <b>For Beginners:</b> Think of matrix decomposition like breaking down a complex number into factors.
/// 
/// For example, the number 12 can be broken down into 3 × 4, making it easier to work with.
/// Similarly, matrix decomposition breaks down a complex matrix into simpler parts that are
/// easier to work with mathematically.
/// 
/// Imagine you have a puzzle (the matrix) that's hard to solve directly:
/// - Matrix decomposition splits this puzzle into smaller, more manageable pieces
/// - These pieces can then be used to solve problems more efficiently
/// - Different types of decompositions (LU, QR, etc.) are like different ways of breaking down the puzzle
/// 
/// In machine learning, matrix decompositions are used to:
/// - Solve systems of equations efficiently (like finding the best-fit line in regression)
/// - Reduce the dimensionality of data (like in Principal Component Analysis)
/// - Find patterns in data (like in recommendation systems)
/// - Speed up calculations that would otherwise be too slow or unstable
/// </remarks>
/// <typeparam name="T">The numeric data type used in the matrix (e.g., float, double).</typeparam>
[AiDotNet.Configuration.YamlConfigurable("MatrixDecomposition")]
public interface IMatrixDecomposition<T>
{
    /// <summary>
    /// Gets the original matrix that was decomposed.
    /// </summary>
    /// <value>
    /// The original matrix used to create this decomposition.
    /// </value>
    /// <remarks>
    /// <b>For Beginners:</b> This property simply gives you access to the original matrix that was broken down.
    /// 
    /// It's like keeping a copy of the original puzzle while you're working with its pieces.
    /// Sometimes you need to refer back to the original matrix for various calculations or to
    /// verify your results.
    /// </remarks>
    Matrix<T> A { get; }

    /// <summary>
    /// Solves a linear system of equations Ax = b, where A is the decomposed matrix.
    /// </summary>
    /// <remarks>
    /// This method efficiently finds the solution vector x without explicitly calculating
    /// the inverse of matrix A, which is computationally expensive and numerically unstable.
    /// 
    /// <b>For Beginners:</b> This method solves equations of the form "Ax = b" where:
    /// - A is your matrix (like a transformation or system of equations)
    /// - b is a known vector (like a set of results or outcomes)
    /// - x is what you're trying to find (like the unknown variables)
    /// 
    /// Real-world example: Imagine you're trying to find the best mix of ingredients for a recipe:
    /// - Matrix A represents how each ingredient affects taste, texture, and nutrition
    /// - Vector b represents your desired taste, texture, and nutrition targets
    /// - Vector x (the solution) tells you how much of each ingredient to use
    /// 
    /// In machine learning, this is used for:
    /// - Finding the best coefficients in linear regression
    /// - Solving for weights in certain neural network calculations
    /// - Transforming data in specific ways
    /// 
    /// This method is much faster and more accurate than directly calculating A?¹ and then
    /// multiplying by b, which is why decompositions are so valuable.
    /// </remarks>
    /// <param name="b">The right-hand side vector of the equation Ax = b.</param>
    /// <returns>The solution vector x that satisfies Ax = b.</returns>
    Vector<T> Solve(Vector<T> b);

    /// <summary>
    /// Calculates the inverse of the original matrix A.
    /// </summary>
    /// <remarks>
    /// The inverse of a matrix A is another matrix A?¹ such that A × A?¹ = I, where I is the identity matrix.
    /// Not all matrices have inverses - only square matrices with non-zero determinants are invertible.
    /// Matrix decompositions provide efficient ways to compute inverses when they exist.
    /// 
    /// <b>For Beginners:</b> The inverse of a matrix is similar to the reciprocal of a number.
    /// 
    /// Just as 5 × (1/5) = 1, a matrix multiplied by its inverse gives the identity matrix
    /// (which is like the number 1 in matrix form, with 1's on the diagonal and 0's elsewhere).
    /// 
    /// For example, if matrix A represents a transformation (like rotating or scaling),
    /// then A?¹ represents the opposite transformation that "undoes" the original:
    /// - If A rotates data clockwise, A?¹ rotates it counterclockwise
    /// - If A scales data up by 2x, A?¹ scales it down by 1/2
    /// 
    /// In machine learning, matrix inverses are used for:
    /// - Solving systems of linear equations
    /// - Computing certain statistical measures
    /// - Implementing some learning algorithms like linear regression
    /// 
    /// However, directly computing inverses can be numerically unstable and slow,
    /// which is why decomposition methods (like this interface provides) are preferred
    /// for actually solving problems.
    /// </remarks>
    /// <returns>The inverse of the original matrix A.</returns>
    Matrix<T> Invert();
}
