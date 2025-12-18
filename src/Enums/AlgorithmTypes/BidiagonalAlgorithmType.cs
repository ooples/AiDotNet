namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for bidiagonal matrix decomposition.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Bidiagonal decomposition is a technique used in linear algebra and machine learning 
/// to simplify complex matrices (tables of numbers) into a special form that makes further calculations easier.
/// 
/// A bidiagonal matrix is a special type of matrix where non-zero values appear only on the main diagonal 
/// (from top-left to bottom-right) and either just above or just below this diagonal. All other values are zero.
/// 
/// This decomposition is often used as a step in solving systems of equations, finding eigenvalues, 
/// or performing Singular Value Decomposition (SVD), which is crucial for many machine learning algorithms 
/// like Principal Component Analysis (PCA), recommendation systems, and image compression.
/// 
/// Think of it like simplifying a complex recipe into basic steps that are easier to follow - 
/// the bidiagonal form makes complex matrix operations more manageable.
/// 
/// This enum lists different mathematical approaches to perform this decomposition, each with its own 
/// advantages in terms of accuracy, speed, or memory usage.
/// </para>
/// </remarks>
public enum BidiagonalAlgorithmType
{
    /// <summary>
    /// Uses Householder reflections to transform a matrix into bidiagonal form.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Householder method uses special mathematical operations called "reflections" 
    /// to gradually transform a matrix into bidiagonal form.
    /// 
    /// Imagine you have a room full of furniture (your original matrix) and you want to rearrange it 
    /// into a very specific layout (the bidiagonal form). The Householder method is like moving all the 
    /// furniture at once in carefully planned steps, where each step (reflection) moves multiple pieces 
    /// into better positions.
    /// 
    /// This method is generally very stable and accurate, making it a popular choice for many applications.
    /// It works well for dense matrices (matrices with mostly non-zero values) and is the standard approach 
    /// in many numerical libraries.
    /// </para>
    /// </remarks>
    Householder,

    /// <summary>
    /// Uses Givens rotations to transform a matrix into bidiagonal form.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Givens method uses mathematical operations called "rotations" to gradually 
    /// transform a matrix into bidiagonal form.
    /// 
    /// Continuing with our furniture analogy, while Householder moves multiple pieces at once, 
    /// the Givens method is like rearranging the room by moving just two pieces of furniture at a time. 
    /// It's more targeted and precise.
    /// 
    /// This precision makes Givens rotations particularly useful for sparse matrices (matrices with 
    /// mostly zero values) or when you need to preserve certain structures in your matrix. It might 
    /// take more individual steps than Householder, but each step is simpler and can be more efficient 
    /// in certain situations.
    /// </para>
    /// </remarks>
    Givens,

    /// <summary>
    /// Uses the Lanczos algorithm to transform a matrix into bidiagonal form.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Lanczos method is an iterative approach that is particularly efficient for 
    /// very large, sparse matrices (huge tables with mostly zero values).
    /// 
    /// In our furniture analogy, if you had an enormous warehouse with just a few pieces of furniture 
    /// scattered throughout (a large sparse matrix), the Lanczos method would be like having a map that 
    /// takes you directly to each piece without having to check every empty space.
    /// 
    /// This method is especially valuable in machine learning and data science when dealing with very 
    /// large datasets, as it can find approximate solutions much faster than the other methods. It's 
    /// particularly useful when you don't need the exact decomposition but a good approximation is sufficient.
    /// </para>
    /// </remarks>
    Lanczos
}
