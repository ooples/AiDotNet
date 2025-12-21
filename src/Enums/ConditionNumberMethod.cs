namespace AiDotNet.Enums;

/// <summary>
/// Specifies different methods for calculating the condition number of a matrix.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The condition number is a measure of how sensitive a matrix is to changes in input.
/// 
/// Think of it like checking how stable a table is - a table with uneven legs (high condition number) 
/// will wobble a lot with small changes, while a stable table (low condition number) remains steady.
/// 
/// In machine learning and numerical computing:
/// - A low condition number (close to 1) indicates a "well-conditioned" matrix that produces reliable results
/// - A high condition number indicates an "ill-conditioned" matrix that might amplify small errors
/// - An infinite condition number means the matrix is singular (cannot be inverted)
/// 
/// This is important because ill-conditioned matrices can cause numerical problems in algorithms,
/// leading to inaccurate results or slow convergence.
/// </para>
/// </remarks>
public enum ConditionNumberMethod
{
    /// <summary>
    /// Calculates the condition number using Singular Value Decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SVD (Singular Value Decomposition) is a way to break down a matrix into three simpler matrices.
    /// 
    /// Using SVD, the condition number is calculated as the ratio of the largest singular value to the smallest non-zero singular value.
    /// 
    /// Advantages:
    /// - Most accurate method
    /// - Works for any matrix shape (not just square matrices)
    /// - Provides the standard definition of the condition number
    /// 
    /// Limitations:
    /// - Computationally expensive for large matrices
    /// - Requires more memory than other methods
    /// 
    /// This is generally the preferred method when accuracy is important and the matrix is not too large.
    /// </para>
    /// </remarks>
    SVD,

    /// <summary>
    /// Calculates the condition number using the L1 norm (sum of absolute column values).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The L1 norm condition number uses the sum of absolute values in each column to estimate how sensitive the matrix is.
    /// 
    /// It's calculated by multiplying the L1 norm of the matrix by the L1 norm of its inverse.
    /// 
    /// Advantages:
    /// - Faster to compute than SVD
    /// - Gives a good approximation for many practical cases
    /// - Easier to understand conceptually
    /// 
    /// Limitations:
    /// - Less accurate than SVD
    /// - Only works for square matrices
    /// - Requires computing the matrix inverse, which might be unstable for nearly singular matrices
    /// 
    /// This method provides a good balance between speed and accuracy for many applications.
    /// </para>
    /// </remarks>
    L1Norm,

    /// <summary>
    /// Calculates the condition number using the Infinity norm (maximum absolute row sum).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Infinity norm condition number uses the maximum row sum of absolute values to estimate matrix sensitivity.
    /// 
    /// It's calculated by multiplying the Infinity norm of the matrix by the Infinity norm of its inverse.
    /// 
    /// Advantages:
    /// - Relatively quick to compute
    /// - Provides a different perspective on matrix sensitivity
    /// - Sometimes more appropriate for certain types of problems
    /// 
    /// Limitations:
    /// - Less accurate than SVD
    /// - Only works for square matrices
    /// - Requires computing the matrix inverse
    /// 
    /// This method is useful when you want to understand how the matrix behaves with respect to changes in specific rows.
    /// </para>
    /// </remarks>
    InfinityNorm,

    /// <summary>
    /// Estimates the condition number using the power iteration method.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Power Iteration is an iterative method that estimates the largest and smallest eigenvalues 
    /// without computing the full matrix decomposition.
    /// 
    /// Instead of calculating everything exactly, it uses repeated multiplication to converge to an estimate.
    /// Think of it like repeatedly zooming in on the most important parts of the matrix.
    /// 
    /// Advantages:
    /// - Much faster for very large matrices
    /// - Uses less memory than SVD
    /// - Can provide a good approximation with fewer calculations
    /// 
    /// Limitations:
    /// - Only provides an estimate, not the exact condition number
    /// - May require many iterations to converge
    /// - May not converge well for certain types of matrices
    /// 
    /// This method is particularly useful for large-scale problems where SVD would be too computationally expensive.
    /// </para>
    /// </remarks>
    PowerIteration
}
