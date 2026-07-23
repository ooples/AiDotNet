namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for UDU' decomposition of matrices.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> UDU' decomposition is a way to break down a symmetric matrix into simpler components 
/// that are easier to work with. The "U" stands for an upper triangular matrix (values only on and above 
/// the diagonal), "D" stands for a diagonal matrix (values only on the diagonal), and "U'" is the 
/// transpose of U.
/// 
/// This decomposition expresses a matrix A as: A = U × D × U'
/// 
/// Think of it like breaking down a complex shape into basic building blocks:
/// - U is like the structure
/// - D contains the scaling factors
/// - U' is the mirror image of U
/// 
/// UDU' decomposition is particularly useful for:
/// - Solving systems of linear equations
/// - Matrix inversion
/// - Numerical stability in computations
/// - Certain statistical and engineering applications
/// 
/// It's similar to other decompositions like LU or Cholesky, but has specific advantages for 
/// symmetric matrices, especially in terms of computational efficiency and numerical stability.
/// </para>
/// </remarks>
public enum UduAlgorithmType
{
    /// <summary>
    /// Uses the Crout algorithm for UDU' decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Crout algorithm is a specific method for performing matrix decomposition 
    /// that computes the elements of the decomposition in a particular order.
    /// 
    /// In the context of UDU' decomposition, the Crout method:
    /// 
    /// 1. Works column by column from left to right
    /// 2. For each column, it first computes the diagonal element
    /// 3. Then computes the elements above the diagonal
    /// 
    /// This approach:
    /// - Is numerically stable
    /// - Can be implemented efficiently
    /// - Works well for dense matrices
    /// - Requires less memory manipulation than some alternatives
    /// - Is particularly good when the matrix has certain patterns
    /// 
    /// The Crout algorithm is named after Prescott Durand Crout, who developed this approach 
    /// for matrix decomposition.
    /// </para>
    /// </remarks>
    Crout,

    /// <summary>
    /// Uses the Doolittle algorithm for UDU' decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Doolittle algorithm is another method for matrix decomposition that 
    /// differs from Crout in the order of computation and some implementation details.
    /// 
    /// In the context of UDU' decomposition, the Doolittle method:
    /// 
    /// 1. Works row by row from top to bottom
    /// 2. For each row, it first computes the elements to the right of the diagonal
    /// 3. Then computes the diagonal element
    /// 
    /// This approach:
    /// - Has similar numerical stability to Crout
    /// - May be more efficient for certain matrix structures
    /// - Can be easier to parallelize in some cases
    /// - Often requires the same amount of computation as Crout
    /// - May be preferred in specific applications due to its computational pattern
    /// 
    /// The Doolittle algorithm is named after Myrick Hascall Doolittle, who contributed to 
    /// the development of this decomposition technique.
    /// </para>
    /// </remarks>
    Doolittle
}
