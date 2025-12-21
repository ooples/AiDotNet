namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for converting a matrix to tridiagonal form.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A tridiagonal matrix is a special type of square matrix where non-zero values appear only 
/// on the main diagonal and the diagonals directly above and below it. All other elements are zero.
/// 
/// For example, a 5×5 tridiagonal matrix looks like this (where * represents non-zero values):
/// 
/// * * 0 0 0
/// * * * 0 0
/// 0 * * * 0
/// 0 0 * * *
/// 0 0 0 * *
/// 
/// Converting a matrix to tridiagonal form is an important step in many numerical algorithms, especially 
/// when finding eigenvalues and eigenvectors. It simplifies the original problem by transforming a dense 
/// matrix (with many non-zero elements) into a simpler form that's easier to work with.
/// 
/// This process is like simplifying a complex equation before solving it - the answer remains the same, 
/// but the work becomes much easier.
/// </para>
/// </remarks>
public enum TridiagonalAlgorithmType
{
    /// <summary>
    /// Uses Householder reflections to convert a matrix to tridiagonal form.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Householder method uses special transformations called "reflections" to 
    /// systematically zero out elements in the matrix.
    /// 
    /// Imagine you have a mirror that can reflect vectors in a special way. The Householder method:
    /// 
    /// 1. Places these "mathematical mirrors" carefully to reflect parts of the matrix
    /// 2. Each reflection zeros out an entire column below the diagonal in one step
    /// 3. After applying reflections to all columns, the matrix becomes tridiagonal
    /// 
    /// This method:
    /// - Is very stable numerically (less prone to calculation errors)
    /// - Works efficiently for dense matrices
    /// - Requires fewer operations than some other methods
    /// - Is the most commonly used approach for tridiagonalization
    /// - Preserves matrix symmetry if the original matrix is symmetric
    /// </para>
    /// </remarks>
    Householder,

    /// <summary>
    /// Uses Givens rotations to convert a matrix to tridiagonal form.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Givens method uses a series of simple rotations to gradually transform 
    /// the matrix into tridiagonal form.
    /// 
    /// Think of it like carefully turning knobs to adjust values:
    /// 
    /// 1. Each Givens rotation zeros out just one element at a time
    /// 2. The rotation affects only two rows and two columns
    /// 3. Many rotations are applied in sequence until all required elements become zero
    /// 
    /// This method:
    /// - Is very precise and stable
    /// - Can be more efficient when only a few elements need to be zeroed
    /// - Works well for sparse matrices (mostly zeros)
    /// - Is easier to parallelize (use multiple processors)
    /// - Can be more selective about which elements to transform
    /// </para>
    /// </remarks>
    Givens,

    /// <summary>
    /// Uses the Lanczos algorithm to convert a matrix to tridiagonal form, particularly efficient for large, sparse matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Lanczos algorithm takes a completely different approach by building a tridiagonal 
    /// matrix that approximates the properties of the original matrix.
    /// 
    /// Imagine creating a simplified model that captures the essential behavior of a complex system:
    /// 
    /// 1. It starts with a vector and generates a sequence of special vectors
    /// 2. These vectors form a basis for a new space
    /// 3. When the original matrix is expressed in this new basis, it becomes tridiagonal
    /// 
    /// This method:
    /// - Is extremely efficient for large, sparse matrices
    /// - Doesn't transform the original matrix but creates an equivalent tridiagonal one
    /// - Uses much less memory than direct methods
    /// - Is particularly useful in iterative methods where you don't need the exact transformation
    /// - Can find approximate eigenvalues very quickly
    /// - Is widely used in search engines, machine learning, and scientific computing
    /// </para>
    /// </remarks>
    Lanczos
}
