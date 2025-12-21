namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for Takagi factorization of complex symmetric matrices.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Takagi factorization is a special type of matrix decomposition that works specifically 
/// with complex symmetric matrices. A complex symmetric matrix is a square matrix that equals its own 
/// transpose, even when the elements are complex numbers.
/// 
/// In simple terms, Takagi factorization breaks down a complex symmetric matrix A into:
/// 
/// A = U × D × U^T
/// 
/// Where:
/// - U is a unitary matrix (similar to a rotation in higher dimensions)
/// - D is a diagonal matrix with non-negative real numbers
/// - U^T is the transpose of U
/// 
/// This factorization is useful in quantum physics, signal processing, and various machine learning 
/// applications where complex data needs to be analyzed.
/// 
/// Different algorithms can be used to compute this factorization, each with its own advantages 
/// and trade-offs in terms of speed, accuracy, and memory usage.
/// </para>
/// </remarks>
public enum TakagiAlgorithmType
{
    /// <summary>
    /// Uses the Jacobi algorithm for Takagi factorization, which is particularly accurate for small matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Jacobi algorithm works by applying a series of simple transformations 
    /// (called Jacobi rotations) to gradually convert the matrix into the desired form.
    /// 
    /// Imagine trying to flatten a bumpy surface by repeatedly smoothing out the biggest bump 
    /// each time until the surface becomes flat. The Jacobi method:
    /// 
    /// 1. Identifies the largest off-diagonal element
    /// 2. Applies a rotation to eliminate that element
    /// 3. Repeats until all off-diagonal elements are very small
    /// 
    /// This method is:
    /// - Very accurate
    /// - Easy to understand
    /// - Works well for small matrices
    /// - Can be slower for large matrices
    /// </para>
    /// </remarks>
    Jacobi,

    /// <summary>
    /// Uses the QR algorithm for Takagi factorization, which is efficient for medium-sized matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The QR algorithm is a powerful method that uses QR decomposition 
    /// (breaking a matrix into an orthogonal matrix Q and an upper triangular matrix R) 
    /// repeatedly to find the factorization.
    /// 
    /// Think of it as repeatedly refining your answer:
    /// 
    /// 1. Break the matrix into Q and R factors
    /// 2. Multiply them back together in the opposite order (RQ)
    /// 3. Repeat until the matrix converges to the desired form
    /// 
    /// This method:
    /// - Is faster than Jacobi for larger matrices
    /// - Has good numerical stability
    /// - Is widely used in scientific computing
    /// - Requires more complex mathematics under the hood
    /// </para>
    /// </remarks>
    QR,

    /// <summary>
    /// Uses eigendecomposition to compute the Takagi factorization, leveraging the relationship between 
    /// Takagi factorization and eigendecomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Eigendecomposition is a way to break down a matrix using special vectors 
    /// (eigenvectors) and values (eigenvalues) that have unique properties.
    /// 
    /// For Takagi factorization, this method:
    /// 
    /// 1. Converts the complex symmetric matrix problem into an eigenvalue problem
    /// 2. Solves the eigenvalue problem using standard techniques
    /// 3. Constructs the Takagi factorization from the eigenvectors and eigenvalues
    /// 
    /// This approach:
    /// - Can leverage highly optimized eigenvalue solvers
    /// - Works well for dense matrices
    /// - Provides good accuracy
    /// - Is mathematically elegant
    /// </para>
    /// </remarks>
    EigenDecomposition,

    /// <summary>
    /// Uses the Power Iteration method to compute the Takagi factorization, which is efficient for finding 
    /// the largest singular values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Power Iteration is a simple but effective method that finds the most important 
    /// components of the factorization first.
    /// 
    /// Imagine trying to find the tallest mountain by always walking uphill:
    /// 
    /// 1. Start with a random direction (vector)
    /// 2. Repeatedly multiply by the matrix
    /// 3. The vector will naturally align with the direction of greatest "stretch"
    /// 4. After finding one component, subtract its influence and repeat for the next
    /// 
    /// This method:
    /// - Is conceptually simple
    /// - Requires minimal memory
    /// - Works well when you only need a few components
    /// - Can be slower to converge if components are similar in size
    /// - Is particularly useful for sparse matrices (mostly zeros)
    /// </para>
    /// </remarks>
    PowerIteration,

    /// <summary>
    /// Uses the Lanczos Iteration method for Takagi factorization, which is efficient for large, sparse matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Lanczos Iteration is an advanced technique that builds a much smaller matrix 
    /// that captures the essential properties of the original large matrix.
    /// 
    /// Think of it like creating a small, accurate model of a large system:
    /// 
    /// 1. It starts with a vector and builds a special sequence of vectors
    /// 2. These vectors form a basis for a smaller matrix (called the tridiagonal matrix)
    /// 3. This smaller matrix is much easier to work with but preserves the important properties
    /// 4. The factorization of the small matrix can be used to approximate the original
    /// 
    /// This method:
    /// - Is extremely efficient for large, sparse matrices
    /// - Uses much less memory than direct methods
    /// - Can find a few components very quickly
    /// - Is widely used in search engines, recommendation systems, and scientific computing
    /// - May have numerical stability issues that require careful handling
    /// </para>
    /// </remarks>
    LanczosIteration
}
