namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for LDL decomposition of matrices.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LDL decomposition is a way to break down a symmetric matrix into simpler parts that make 
/// calculations much easier and faster.
/// 
/// Imagine you have a complex puzzle (the matrix) that you need to solve. LDL decomposition breaks this puzzle 
/// into three simpler pieces:
/// 
/// 1. L - A lower triangular matrix (has values only on and below the diagonal)
/// 2. D - A diagonal matrix (has values only along the diagonal)
/// 3. L^T - The transpose of L (L flipped over its diagonal)
/// 
/// So instead of solving one complex puzzle, you can solve three simpler ones in sequence.
/// 
/// Why is this important in AI and machine learning?
/// 
/// 1. Solving Linear Systems: Many AI algorithms need to solve equations like Ax = b. LDL decomposition makes 
///    this much faster and more stable.
/// 
/// 2. Matrix Inversion: Finding the inverse of a matrix (like dividing by a matrix) becomes much easier.
/// 
/// 3. Numerical Stability: LDL decomposition is more numerically stable than directly inverting matrices, 
///    meaning it's less prone to calculation errors.
/// 
/// 4. Covariance Matrices: In machine learning, we often work with covariance matrices that are symmetric 
///    and positive definite - perfect candidates for LDL decomposition.
/// 
/// 5. Optimization Problems: Many machine learning algorithms involve optimization that requires solving 
///    systems of linear equations repeatedly.
/// 
/// This enum specifies which specific algorithm to use for performing the LDL decomposition, as different 
/// methods have different performance characteristics depending on the matrix properties.
/// </para>
/// </remarks>
public enum LdlAlgorithmType
{
    /// <summary>
    /// Uses a modified Cholesky decomposition approach to compute the LDL factorization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Cholesky method is a specialized approach for LDL decomposition that works 
    /// specifically with symmetric, positive-definite matrices (a special type of matrix common in 
    /// machine learning).
    /// 
    /// Imagine you're trying to find the square root of a number - Cholesky decomposition is like finding 
    /// the "square root" of a matrix. It's a more efficient version of LDL decomposition for certain types 
    /// of matrices.
    /// 
    /// The Cholesky method:
    /// 
    /// 1. Is about twice as efficient as other decomposition methods for applicable matrices
    /// 
    /// 2. Is very numerically stable (resistant to calculation errors)
    /// 
    /// 3. Works best with symmetric, positive-definite matrices (common in statistics and machine learning)
    /// 
    /// 4. Is widely used in Monte Carlo simulations, Kalman filters, and optimization algorithms
    /// 
    /// This is typically the default choice for LDL decomposition when you know your matrix is symmetric 
    /// and positive-definite, as it offers the best performance and stability for these cases.
    /// 
    /// In machine learning, this is particularly useful for working with covariance matrices, kernel matrices, 
    /// and Hessian matrices in optimization problems.
    /// </para>
    /// </remarks>
    Cholesky,

    /// <summary>
    /// Uses the Crout algorithm to compute the LDL factorization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Crout method is a more general approach for LDL decomposition that can handle 
    /// a wider variety of matrices, including those that aren't positive-definite.
    /// 
    /// While Cholesky is like finding the square root of a matrix (which only works for certain matrices), 
    /// Crout is like finding a more general factorization that works for more types of matrices.
    /// 
    /// The Crout method:
    /// 
    /// 1. Can handle symmetric matrices that aren't positive-definite (matrices with negative eigenvalues)
    /// 
    /// 2. Is more versatile than Cholesky but slightly less efficient
    /// 
    /// 3. Can be modified to detect and handle matrices that are nearly singular (close to having no inverse)
    /// 
    /// 4. Is useful when you're not certain about the properties of your matrix
    /// 
    /// This method is particularly valuable when working with matrices that might not be positive-definite, 
    /// such as certain correlation matrices, matrices derived from physical systems with constraints, or 
    /// matrices that have been affected by numerical errors.
    /// 
    /// In machine learning, this can be useful when working with indefinite kernels or when implementing 
    /// robust versions of algorithms that need to handle edge cases.
    /// </para>
    /// </remarks>
    Crout
}
