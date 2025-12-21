namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for Hessenberg decomposition of matrices.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Hessenberg decomposition is a way to transform a complex matrix into a simpler form 
/// that makes further calculations much easier and faster.
/// 
/// Imagine you have a cluttered desk with papers scattered everywhere. Hessenberg decomposition is like 
/// organizing that desk so that all papers are neatly stacked in one corner, making it much easier to 
/// find what you need. In mathematical terms, it transforms a matrix so that all elements below the first 
/// subdiagonal are zero (creating a staircase-like pattern).
/// 
/// Why is this important in AI and machine learning?
/// 
/// 1. Eigenvalue Calculations: Many AI algorithms need to find eigenvalues of matrices (special values that 
///    help understand the fundamental properties of data). Hessenberg form makes finding these values much faster.
/// 
/// 2. Computational Efficiency: Converting to Hessenberg form reduces the number of operations needed for 
///    many matrix calculations from O(n³) to O(n²), making algorithms run much faster for large datasets.
/// 
/// 3. Numerical Stability: These transformations improve the accuracy of calculations by reducing 
///    rounding errors that can accumulate when working with floating-point numbers.
/// 
/// 4. Dimensionality Reduction: In some machine learning applications, Hessenberg decomposition can help 
///    identify important patterns in high-dimensional data.
/// 
/// This enum specifies which specific algorithm to use for performing the Hessenberg decomposition, as 
/// different methods have different performance characteristics depending on the matrix size and structure.
/// </para>
/// </remarks>
public enum HessenbergAlgorithmType
{
    /// <summary>
    /// Uses Householder reflections to compute the Hessenberg form.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Householder method uses special mathematical operations called "reflections" to 
    /// transform a matrix into Hessenberg form.
    /// 
    /// Imagine you're rearranging furniture in a room by flipping the entire room over an imaginary line - 
    /// that's similar to how a Householder reflection works. It reflects vectors across carefully chosen planes 
    /// to zero out elements in the matrix.
    /// 
    /// The Householder method is:
    /// 
    /// 1. Very numerically stable (resistant to rounding errors)
    /// 
    /// 2. Efficient for dense matrices (matrices where most elements are non-zero)
    /// 
    /// 3. The most commonly used method in practice
    /// 
    /// 4. Well-suited for parallel computing
    /// 
    /// This is typically the default choice for Hessenberg decomposition because of its excellent balance of 
    /// stability and performance across a wide range of matrix types.
    /// </para>
    /// </remarks>
    Householder,

    /// <summary>
    /// Uses Givens rotations to compute the Hessenberg form.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Givens method uses mathematical operations called "rotations" to transform a matrix 
    /// into Hessenberg form.
    /// 
    /// While Householder reflections flip entire spaces at once, Givens rotations are more like turning a dial - 
    /// they rotate just two rows or columns at a time. Imagine adjusting just two sliders on a mixing board 
    /// rather than moving all sliders at once.
    /// 
    /// The Givens method is particularly useful when:
    /// 
    /// 1. Working with sparse matrices (matrices with mostly zeros)
    /// 
    /// 2. You need to preserve certain structures in your matrix
    /// 
    /// 3. You're updating an existing decomposition after small changes to the original matrix
    /// 
    /// 4. Working with matrices that have special patterns
    /// 
    /// While generally slower than Householder for dense matrices, Givens rotations can be more efficient for 
    /// certain specialized applications where you only need to modify a few elements of the matrix.
    /// </para>
    /// </remarks>
    Givens,

    /// <summary>
    /// Uses elementary transformations to compute the Hessenberg form.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Elementary Transformations method uses the most basic matrix operations to transform 
    /// a matrix into Hessenberg form.
    /// 
    /// These operations are like the fundamental building blocks of matrix manipulation - simple operations like 
    /// multiplying a row by a constant, adding one row to another, or swapping rows. Think of it like cooking 
    /// with just the most basic ingredients and techniques.
    /// 
    /// This approach:
    /// 
    /// 1. Is conceptually simpler and easier to understand
    /// 
    /// 2. Can be useful for educational purposes
    /// 
    /// 3. Works well for certain specialized matrix structures
    /// 
    /// 4. May be easier to implement in some programming environments
    /// 
    /// However, it's generally less computationally efficient and less numerically stable than methods like 
    /// Householder or Givens for general-purpose use. It's most appropriate for small matrices or special cases 
    /// where its simplicity offers an advantage.
    /// </para>
    /// </remarks>
    ElementaryTransformations,

    /// <summary>
    /// Uses the Implicit QR algorithm to compute the Hessenberg form.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Implicit QR method combines Hessenberg decomposition with eigenvalue calculations in 
    /// a single, efficient process.
    /// 
    /// Rather than first converting to Hessenberg form and then finding eigenvalues, this method does both 
    /// simultaneously. It's like cooking a one-pot meal instead of preparing each ingredient separately.
    /// 
    /// The Implicit QR method:
    /// 
    /// 1. Is highly efficient when you need both the Hessenberg form and eigenvalues
    /// 
    /// 2. Reduces the total number of operations required
    /// 
    /// 3. Often has better numerical properties for the combined task
    /// 
    /// 4. Is the method of choice in many professional numerical libraries
    /// 
    /// This method is particularly valuable in applications like principal component analysis (PCA), signal 
    /// processing, and vibration analysis, where eigenvalues are the ultimate goal of the computation.
    /// </para>
    /// </remarks>
    ImplicitQR,

    /// <summary>
    /// Uses the Lanczos algorithm to compute the Hessenberg form for symmetric matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Lanczos method is a specialized algorithm designed specifically for symmetric matrices 
    /// (matrices that are mirror images across their diagonal).
    /// 
    /// For symmetric matrices, the Hessenberg form is actually tridiagonal (non-zero elements only on the main 
    /// diagonal and the diagonals just above and below it). The Lanczos algorithm exploits this special structure 
    /// to work much more efficiently.
    /// 
    /// Imagine having a shortcut through a maze because you know a special property of the maze - that's what 
    /// Lanczos does by taking advantage of matrix symmetry.
    /// 
    /// The Lanczos method:
    /// 
    /// 1. Is extremely efficient for large, symmetric matrices
    /// 
    /// 2. Uses much less memory than other methods
    /// 
    /// 3. Can find approximate eigenvalues without computing the full decomposition
    /// 
    /// 4. Is particularly useful in quantum mechanics, structural analysis, and graph theory applications
    /// 
    /// This method should only be used with symmetric matrices. For non-symmetric matrices, one of the other 
    /// methods should be chosen instead.
    /// </para>
    /// </remarks>
    Lanczos
}
