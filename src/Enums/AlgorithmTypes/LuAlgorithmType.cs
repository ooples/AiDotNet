namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for LU decomposition of matrices.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LU decomposition is a way to break down a matrix into two simpler parts that make 
/// calculations much easier and faster.
/// 
/// Imagine you have a complex math problem (the matrix) that you need to solve. LU decomposition breaks this problem 
/// into two simpler pieces:
/// 
/// 1. L - A lower triangular matrix (has values only on and below the diagonal)
/// 2. U - An upper triangular matrix (has values only on and above the diagonal)
/// 
/// So instead of solving one complex problem, you can solve two simpler ones in sequence.
/// 
/// Why is this important in AI and machine learning?
/// 
/// 1. Solving Linear Systems: Many AI algorithms need to solve equations like Ax = b. LU decomposition makes 
///    this much faster.
/// 
/// 2. Matrix Inversion: Finding the inverse of a matrix becomes much easier with LU decomposition.
/// 
/// 3. Determinant Calculation: The determinant (a special number associated with a matrix) can be easily 
///    calculated from the LU decomposition.
/// 
/// 4. Efficient Repeated Solving: If you need to solve multiple problems with the same matrix but different 
///    right-hand sides, LU decomposition lets you do the hard work just once.
/// 
/// 5. Feature Transformation: In machine learning, LU decomposition can help transform features to make 
///    algorithms work better.
/// 
/// This enum specifies which specific algorithm to use for performing the LU decomposition, as different 
/// methods have different performance characteristics depending on the matrix properties.
/// </para>
/// </remarks>
public enum LuAlgorithmType
{
    /// <summary>
    /// Uses the Doolittle algorithm to compute the LU factorization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Doolittle method is a specific way to perform LU decomposition where the diagonal 
    /// elements of L (the lower triangular matrix) are all set to 1.
    /// 
    /// Imagine you're building a house - the Doolittle method is like having a standard foundation (the 1's on 
    /// the diagonal of L) and then building the rest of the structure on top of that foundation.
    /// 
    /// The Doolittle method:
    /// 
    /// 1. Is one of the most common and straightforward LU decomposition methods
    /// 
    /// 2. Works well for general square matrices
    /// 
    /// 3. Is relatively easy to implement and understand
    /// 
    /// 4. Produces an L matrix with 1's on the diagonal
    /// 
    /// This is often the default choice for LU decomposition when no special properties of the matrix are known.
    /// 
    /// In machine learning, this is useful for general-purpose matrix operations like solving linear systems 
    /// in regression problems or when implementing algorithms from scratch.
    /// </para>
    /// </remarks>
    Doolittle,

    /// <summary>
    /// Uses the Crout algorithm to compute the LU factorization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Crout method is similar to Doolittle, but it sets the diagonal elements of U 
    /// (the upper triangular matrix) to 1 instead of L.
    /// 
    /// If Doolittle is like building a house with a standard foundation, Crout is like building with a 
    /// standard roof structure (the 1's on the diagonal of U) and customizing the foundation.
    /// 
    /// The Crout method:
    /// 
    /// 1. Is mathematically equivalent to Doolittle but organizes calculations differently
    /// 
    /// 2. Produces a U matrix with 1's on the diagonal
    /// 
    /// 3. May be more convenient for certain types of problems
    /// 
    /// 4. Can sometimes be more numerically stable for certain matrix types
    /// 
    /// The choice between Doolittle and Crout often comes down to personal preference or specific 
    /// implementation details, as they produce equivalent results.
    /// 
    /// In machine learning applications, both methods can be used interchangeably for solving linear systems, 
    /// matrix inversion, or other operations requiring LU decomposition.
    /// </para>
    /// </remarks>
    Crout,

    /// <summary>
    /// Uses LU decomposition with partial pivoting for improved numerical stability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Partial pivoting is a technique that makes LU decomposition more reliable by rearranging 
    /// the rows of the matrix during the calculation.
    /// 
    /// Imagine you're cooking a complex recipe. Partial pivoting is like making sure you always use the 
    /// strongest ingredient available at each step to avoid the recipe failing.
    /// 
    /// Without pivoting, if a very small number (close to zero) appears in a critical position, it can cause 
    /// large calculation errors. Partial pivoting prevents this by swapping rows to get larger numbers in 
    /// these critical positions.
    /// 
    /// The Partial Pivoting method:
    /// 
    /// 1. Is much more numerically stable than basic Doolittle or Crout
    /// 
    /// 2. Works well for general matrices, even those that would cause problems for basic methods
    /// 
    /// 3. Keeps track of row permutations using a permutation matrix or vector
    /// 
    /// 4. Is the standard choice for most practical applications
    /// 
    /// In machine learning, this stability is crucial when working with real-world data that might be 
    /// ill-conditioned or when high precision is required, such as in complex optimization problems or 
    /// when working with features that have very different scales.
    /// </para>
    /// </remarks>
    PartialPivoting,

    /// <summary>
    /// Uses LU decomposition with complete pivoting for maximum numerical stability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Complete pivoting takes the stability idea from partial pivoting even further by 
    /// rearranging both rows AND columns during the calculation.
    /// 
    /// Extending our cooking analogy, complete pivoting is like reorganizing both your ingredients (rows) 
    /// AND your cooking steps (columns) to ensure the best possible outcome.
    /// 
    /// Complete pivoting looks for the largest element in the entire remaining submatrix at each step, not 
    /// just in a single column.
    /// 
    /// The Complete Pivoting method:
    /// 
    /// 1. Provides the best possible numerical stability
    /// 
    /// 2. Is useful for very ill-conditioned matrices (matrices that are almost singular)
    /// 
    /// 3. Requires more computation than partial pivoting
    /// 
    /// 4. Keeps track of both row and column permutations
    /// 
    /// This method is the most robust choice when working with extremely challenging matrices where even 
    /// partial pivoting might not be sufficient.
    /// 
    /// In machine learning, complete pivoting might be necessary when working with highly correlated features, 
    /// in sensitive numerical optimization problems, or when implementing algorithms that require extremely 
    /// high precision.
    /// </para>
    /// </remarks>
    CompletePivoting,

    /// <summary>
    /// Uses the Cholesky algorithm for LU decomposition of symmetric positive-definite matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Cholesky method is a specialized form of LU decomposition that only works for a 
    /// specific type of matrix (symmetric positive-definite), but is twice as efficient when applicable.
    /// 
    /// Imagine you have a special type of puzzle that has a hidden symmetry. If you recognize this symmetry, 
    /// you can solve it in half the time using a special technique - that's what Cholesky does.
    /// 
    /// A symmetric positive-definite matrix has two special properties:
    /// 1. It's symmetric (the value at position [i,j] equals the value at [j,i])
    /// 2. It has a special mathematical property called "positive-definiteness"
    /// 
    /// Many matrices in machine learning are symmetric positive-definite, including:
    /// - Covariance matrices (describing how features vary together)
    /// - Kernel matrices in kernel methods
    /// - Hessian matrices in optimization problems
    /// 
    /// The Cholesky method:
    /// 
    /// 1. Is about twice as fast as regular LU decomposition for applicable matrices
    /// 
    /// 2. Produces a special form where U is actually the transpose of L (so you only need to store one matrix)
    /// 
    /// 3. Is very numerically stable
    /// 
    /// 4. Will fail if the matrix is not symmetric positive-definite
    /// 
    /// In machine learning, Cholesky decomposition is commonly used in implementing algorithms like Gaussian 
    /// processes, certain forms of linear regression, Mahalanobis distance calculations, and Monte Carlo 
    /// simulations.
    /// </para>
    /// </remarks>
    Cholesky
}
