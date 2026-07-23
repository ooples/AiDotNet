namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for computing the QR decomposition of matrices.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> QR decomposition is a fundamental technique in linear algebra that breaks down a matrix into 
/// two components: Q (an orthogonal matrix) and R (an upper triangular matrix).
/// 
/// Think of it like breaking down a complex movement (like throwing a ball) into two simpler movements:
/// 1. First, rotating your body to face the right direction (the Q part)
/// 2. Then, moving your arm forward in a straight line (the R part)
/// 
/// In matrix terms:
/// - If A is your original matrix, QR decomposition gives you A = QR
/// - Q has perpendicular columns with unit length (orthogonal)
/// - R is upper triangular (has zeros below the diagonal)
/// 
/// Why is QR decomposition important in AI and machine learning?
/// 
/// 1. Solving Linear Systems: Helps solve equations more efficiently and stably
/// 
/// 2. Least Squares Problems: Essential for finding the best-fit line or curve through data points
/// 
/// 3. Eigenvalue Calculations: Used in dimensionality reduction techniques like PCA
/// 
/// 4. Feature Extraction: Helps identify important patterns in high-dimensional data
/// 
/// 5. Numerical Stability: Makes many calculations more reliable, especially with nearly dependent data
/// 
/// This enum specifies which specific algorithm to use for computing the QR decomposition, as different 
/// methods have different performance characteristics depending on the matrix properties.
/// </para>
/// </remarks>
public enum QrAlgorithmType
{
    /// <summary>
    /// Uses the classical Gram-Schmidt process to compute the QR decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gram-Schmidt process is like creating a set of perfectly perpendicular directions 
    /// from a set of possibly overlapping directions.
    /// 
    /// Imagine you're in a room and want to describe locations using "forward/backward," "left/right," and 
    /// "up/down." If your initial directions aren't perfectly aligned with the room's walls (they're skewed), 
    /// Gram-Schmidt helps you straighten them out to get clean, perpendicular directions.
    /// 
    /// The process works by:
    /// 1. Taking the first vector as is
    /// 2. For each subsequent vector, removing any components that point in the same direction as previous vectors
    /// 3. Then normalizing each vector (making its length equal to 1)
    /// 
    /// The classical Gram-Schmidt method:
    /// 
    /// 1. Is conceptually simple and easy to understand
    /// 
    /// 2. Works well for small matrices and educational purposes
    /// 
    /// 3. Is less numerically stable than other methods when used with floating-point arithmetic
    /// 
    /// 4. Has a straightforward implementation
    /// 
    /// In machine learning applications, understanding Gram-Schmidt helps in feature engineering when you want 
    /// to create independent features from correlated ones, or when implementing certain types of neural network 
    /// layers that benefit from orthogonal weights.
    /// </para>
    /// </remarks>
    GramSchmidt,

    /// <summary>
    /// Uses Householder reflections to compute the QR decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Householder method uses special transformations called "reflections" to gradually 
    /// transform a matrix into triangular form.
    /// 
    /// Imagine you have a mirror and you're using it to reflect light onto specific targets. A Householder 
    /// reflection is like perfectly positioning a mirror to transform multiple points at once in a very 
    /// specific way.
    /// 
    /// For each column of the matrix:
    /// 1. The method creates a special "mirror" (Householder reflector)
    /// 2. This mirror is positioned to reflect the column in such a way that all elements below the diagonal become zero
    /// 3. This reflection is applied to the remaining columns as well
    /// 
    /// The Householder method:
    /// 
    /// 1. Is numerically very stable (much more so than classical Gram-Schmidt)
    /// 
    /// 2. Is efficient for medium to large matrices
    /// 
    /// 3. Requires fewer arithmetic operations than Givens rotations for dense matrices
    /// 
    /// 4. Is the method of choice in many professional numerical libraries
    /// 
    /// 5. Transforms entire columns at once rather than working element by element
    /// 
    /// In machine learning, this stable QR decomposition is often used behind the scenes in algorithms that 
    /// require reliable numerical linear algebra, such as in training certain types of models, in dimensionality 
    /// reduction techniques, or when solving least squares problems with many features.
    /// </para>
    /// </remarks>
    Householder,

    /// <summary>
    /// Uses Givens rotations to compute the QR decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Givens method uses a series of simple rotations to gradually transform a matrix 
    /// into triangular form.
    /// 
    /// Think of it like carefully turning a combination lock: you make one precise rotation at a time, each 
    /// affecting just two numbers, until the whole lock reaches the correct position.
    /// 
    /// The method works by:
    /// 1. Identifying two elements in a column
    /// 2. Creating a rotation that makes one of these elements zero
    /// 3. Applying this rotation to the entire matrix
    /// 4. Repeating until all elements below the diagonal are zero
    /// 
    /// The Givens rotation method:
    /// 
    /// 1. Is very precise and numerically stable
    /// 
    /// 2. Works on just two rows at a time, making it ideal for sparse matrices
    /// 
    /// 3. Can be easily parallelized for certain matrix structures
    /// 
    /// 4. Is excellent for making targeted changes to matrices
    /// 
    /// 5. Is particularly useful when you only need to zero out specific elements
    /// 
    /// In machine learning applications, Givens rotations might be used when working with sparse data matrices, 
    /// when updating existing QR decompositions after small changes to the data, or in specialized algorithms 
    /// that process streaming data where the matrix is built up row by row.
    /// </para>
    /// </remarks>
    Givens,

    /// <summary>
    /// Uses the Modified Gram-Schmidt process to compute the QR decomposition with improved numerical stability.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Modified Gram-Schmidt process is an improved version of the classical Gram-Schmidt 
    /// that addresses numerical stability issues.
    /// 
    /// Imagine you're building a house: the classical Gram-Schmidt is like measuring and cutting all your lumber 
    /// at the beginning (which might lead to accumulated errors), while the modified version is like measuring 
    /// and cutting each piece just before you use it (reducing error buildup).
    /// 
    /// The key difference is:
    /// - Classical: Computes all projections using the original vectors
    /// - Modified: Updates the vectors after each projection, using the most recent versions
    /// 
    /// The Modified Gram-Schmidt method:
    /// 
    /// 1. Is much more numerically stable than the classical version
    /// 
    /// 2. Produces more accurate results, especially for ill-conditioned matrices
    /// 
    /// 3. Has the same computational complexity as the classical version
    /// 
    /// 4. Is still conceptually straightforward to understand
    /// 
    /// 5. Is suitable for practical applications, not just educational purposes
    /// 
    /// In machine learning, this improved stability is important when working with real-world data that might 
    /// have nearly linearly dependent features, when implementing algorithms that require orthogonal bases, 
    /// or when the precision of the decomposition directly affects the quality of predictions or classifications.
    /// </para>
    /// </remarks>
    ModifiedGramSchmidt,

    /// <summary>
    /// Uses an iterative version of the Gram-Schmidt process that repeats the orthogonalization step to achieve higher accuracy.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Iterative Gram-Schmidt method takes the modified Gram-Schmidt process and repeats it 
    /// multiple times to achieve even better numerical accuracy.
    /// 
    /// Think of it like cleaning a window: the first pass removes most of the dirt, but you might go over it 
    /// again to catch the spots you missed and make it truly spotless.
    /// 
    /// The process works by:
    /// 1. Performing a complete Modified Gram-Schmidt orthogonalization
    /// 2. Checking if the resulting vectors are sufficiently orthogonal
    /// 3. If not, applying the orthogonalization process again to further refine the results
    /// 4. Repeating until a desired level of orthogonality is achieved
    /// 
    /// The Iterative Gram-Schmidt method:
    /// 
    /// 1. Provides the highest numerical accuracy among Gram-Schmidt variants
    /// 
    /// 2. Is useful for very ill-conditioned matrices where even Modified Gram-Schmidt might not be sufficient
    /// 
    /// 3. Requires more computational time due to the repeated orthogonalization
    /// 
    /// 4. Can achieve results comparable to Householder reflections in terms of stability
    /// 
    /// 5. Allows for a trade-off between speed and accuracy by adjusting the number of iterations
    /// 
    /// In machine learning applications, this high-precision orthogonalization might be used when working with 
    /// extremely sensitive models, when the data has many highly correlated features, or in specialized scientific 
    /// applications where even small numerical errors could propagate and affect results significantly.
    /// </para>
    /// </remarks>
    IterativeGramSchmidt
}
