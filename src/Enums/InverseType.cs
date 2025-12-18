namespace AiDotNet.Enums;

/// <summary>
/// Specifies different algorithms for calculating matrix inverses in mathematical operations.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A matrix inverse is like finding the opposite of a number. Just as 1/5 is the 
/// inverse of 5 (because 5 × 1/5 = 1), a matrix inverse is a special matrix that, when multiplied 
/// with the original matrix, gives the identity matrix (the matrix equivalent of the number 1).
/// 
/// Matrix inverses are important in AI and machine learning for:
/// - Solving systems of equations
/// - Finding optimal parameters in linear regression
/// - Transforming data
/// - Many other mathematical operations
/// 
/// Different algorithms for finding inverses have different trade-offs in terms of speed, 
/// accuracy, and memory usage. This enum lets you choose which algorithm to use.
/// </para>
/// </remarks>
public enum InverseType
{
    /// <summary>
    /// A divide-and-conquer algorithm for matrix inversion that's efficient for large matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Strassen algorithm is a clever approach that breaks down large matrix 
    /// operations into smaller ones, making calculations faster for big matrices.
    /// 
    /// Think of it as:
    /// - Like solving a big puzzle by breaking it into smaller, manageable pieces
    /// - More efficient than traditional methods for large matrices
    /// - Uses fewer multiplication operations (which are computationally expensive)
    /// - A good balance between speed and accuracy
    /// 
    /// Best used when:
    /// - Working with large matrices (typically larger than 128×128)
    /// - Speed is important
    /// - You have sufficient memory available
    /// - The matrix is well-conditioned (not close to being non-invertible)
    /// </para>
    /// </remarks>
    Strassen,

    /// <summary>
    /// An iterative algorithm that approximates the inverse through successive refinements.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Newton's method (also called Newton-Raphson) finds the inverse by making 
    /// an initial guess and then repeatedly improving that guess.
    /// 
    /// Think of it as:
    /// - Like homing in on a target by making better and better guesses
    /// - Starts with an approximation and refines it iteratively
    /// - Can be very efficient when a good initial guess is available
    /// - Works well for special types of matrices
    /// 
    /// Best used when:
    /// - You already have a good approximate inverse
    /// - Working with special matrix types (like diagonally dominant matrices)
    /// - The matrix has special properties that make Newton's method converge quickly
    /// - You need to update an inverse after small changes to the original matrix
    /// </para>
    /// </remarks>
    Newton,

    /// <summary>
    /// A direct method for finding matrix inverses using elementary row operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gaussian-Jordan method is a systematic approach that transforms 
    /// the original matrix into the identity matrix while simultaneously transforming the 
    /// identity matrix into the inverse.
    /// 
    /// Think of it as:
    /// - Like solving a step-by-step puzzle following clear rules
    /// - The most commonly taught method in linear algebra classes
    /// - Reliable and works for almost any invertible matrix
    /// - Easy to understand and implement
    /// 
    /// Best used when:
    /// - You need a reliable, general-purpose method
    /// - Working with small to medium-sized matrices
    /// - Accuracy is more important than speed
    /// - You need to understand each step of the process
    /// - The matrix is well-conditioned (not close to being non-invertible)
    /// </para>
    /// </remarks>
    GaussianJordan
}
