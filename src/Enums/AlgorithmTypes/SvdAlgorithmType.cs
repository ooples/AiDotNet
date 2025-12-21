namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for Singular Value Decomposition (SVD).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Singular Value Decomposition (SVD) is a powerful mathematical technique that breaks down a matrix 
/// (which you can think of as a table of numbers) into three simpler component matrices. It's like taking apart a 
/// complex machine to understand how it works.
/// 
/// Here's what SVD does in simple terms:
/// 
/// 1. It takes a matrix A and decomposes it into three matrices: U, S (Sigma), and V^T
///    A = U × S × V^T
/// 
/// 2. Each of these matrices has special properties:
///    - U contains the "left singular vectors" (think of these as the basic patterns in the rows of A)
///    - S is a diagonal matrix containing the "singular values" (think of these as importance scores)
///    - V^T contains the "right singular vectors" (think of these as the basic patterns in the columns of A)
/// 
/// Why is SVD important in AI and machine learning?
/// 
/// 1. Dimensionality Reduction: SVD helps compress data by keeping only the most important components
/// 
/// 2. Noise Reduction: By removing components with small singular values, we can filter out noise
/// 
/// 3. Recommendation Systems: SVD powers many recommendation algorithms (like those used by Netflix)
/// 
/// 4. Image Processing: It's used for image compression and facial recognition
/// 
/// 5. Natural Language Processing: SVD is used in techniques like Latent Semantic Analysis
/// 
/// 6. Data Visualization: It can help reduce high-dimensional data to 2D or 3D for visualization
/// 
/// This enum specifies which specific algorithm to use for computing the SVD, as different methods have different 
/// performance characteristics and may be more suitable for certain types of matrices or applications.
/// </para>
/// </remarks>
public enum SvdAlgorithmType
{
    /// <summary>
    /// Uses the Golub-Reinsch algorithm for SVD computation, which is the classical approach.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Golub-Reinsch algorithm is the "classic" method for computing SVD. It's like the 
    /// standard recipe that has been trusted for decades.
    /// 
    /// This algorithm works in two main steps:
    /// 
    /// 1. First, it reduces the original matrix to a bidiagonal form (a simpler matrix with non-zero elements 
    ///    only on the main diagonal and the diagonal just above it)
    /// 
    /// 2. Then, it iteratively computes the SVD of this bidiagonal matrix
    /// 
    /// The Golub-Reinsch approach:
    /// 
    /// 1. Is numerically stable (gives accurate results even with challenging matrices)
    /// 
    /// 2. Works well for small to medium-sized dense matrices
    /// 
    /// 3. Has predictable performance across different types of matrices
    /// 
    /// 4. Is well-studied and understood
    /// 
    /// 5. Computes the full SVD (all singular values and vectors)
    /// 
    /// This method is particularly useful when:
    /// 
    /// 1. You need high accuracy
    /// 
    /// 2. Your matrix is dense and not too large
    /// 
    /// 3. You need all singular values and vectors
    /// 
    /// 4. You want a reliable, well-tested approach
    /// 
    /// In machine learning applications, the Golub-Reinsch algorithm provides a solid foundation for techniques 
    /// like Principal Component Analysis (PCA), where accuracy in computing the decomposition is important.
    /// </para>
    /// </remarks>
    GolubReinsch,

    /// <summary>
    /// Uses the Jacobi algorithm for SVD computation, which is particularly accurate for small matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Jacobi algorithm takes a different approach to computing SVD by using a series of 
    /// rotations to gradually transform the matrix.
    /// 
    /// Imagine you're trying to align a crooked picture frame. The Jacobi method is like making a series of small 
    /// adjustments, rotating it bit by bit until it's perfectly straight:
    /// 
    /// 1. It looks for the largest off-diagonal element in the matrix
    /// 
    /// 2. It applies a rotation to make that element zero
    /// 
    /// 3. It repeats this process many times until all off-diagonal elements are very close to zero
    /// 
    /// The Jacobi approach:
    /// 
    /// 1. Is extremely accurate, often more precise than other methods
    /// 
    /// 2. Works particularly well for small matrices
    /// 
    /// 3. Is easy to parallelize (can use multiple processors efficiently)
    /// 
    /// 4. Converges more slowly for large matrices
    /// 
    /// 5. Is simpler to understand and implement than some other methods
    /// 
    /// This method is particularly valuable when:
    /// 
    /// 1. You need very high numerical precision
    /// 
    /// 2. You're working with small matrices
    /// 
    /// 3. You have parallel computing resources available
    /// 
    /// 4. The matrix has special properties (like being symmetric)
    /// 
    /// In machine learning applications, the Jacobi algorithm can be useful for sensitive applications where 
    /// numerical precision is critical, such as in certain scientific computing tasks or when working with 
    /// ill-conditioned matrices where other methods might be less stable.
    /// </para>
    /// </remarks>
    Jacobi,

    /// <summary>
    /// Uses a randomized algorithm for SVD computation, which is faster but provides an approximation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Randomized SVD algorithms use probability and random sampling to quickly compute an 
    /// approximate SVD, trading some accuracy for significant speed improvements.
    /// 
    /// Think of it like taking a survey: instead of asking everyone in a city about their opinion, you might 
    /// randomly sample a few hundred people to get a good approximation much more quickly:
    /// 
    /// 1. It first creates a smaller matrix by randomly projecting the original large matrix
    /// 
    /// 2. It then computes the SVD of this much smaller matrix
    /// 
    /// 3. Finally, it converts this result back to an approximate SVD of the original matrix
    /// 
    /// The Randomized approach:
    /// 
    /// 1. Is much faster than classical methods for large matrices
    /// 
    /// 2. Requires less memory
    /// 
    /// 3. Provides an approximation rather than an exact result
    /// 
    /// 4. Works particularly well when the matrix has rapidly decaying singular values
    /// 
    /// 5. Can be tuned to balance speed versus accuracy
    /// 
    /// This method is particularly useful when:
    /// 
    /// 1. You're working with very large matrices
    /// 
    /// 2. You need results quickly
    /// 
    /// 3. An approximate solution is acceptable
    /// 
    /// 4. You're doing exploratory data analysis
    /// 
    /// 5. The matrix has a low effective rank (most of the information is contained in a few components)
    /// 
    /// In machine learning applications, Randomized SVD enables processing of large datasets that would be 
    /// impractical with classical methods, making it valuable for tasks like large-scale topic modeling, 
    /// image processing, or analyzing massive recommendation systems.
    /// </para>
    /// </remarks>
    Randomized,

    /// <summary>
    /// Uses the Power Iteration method for SVD computation, which is efficient for finding the largest singular values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Power Iteration method is a simple but powerful approach that's especially good at 
    /// finding the largest singular values and their corresponding vectors.
    /// 
    /// Imagine you're trying to find the tallest mountain in a range. The Power Iteration method is like starting 
    /// at a random point and always walking uphill - eventually, you'll reach the highest peak:
    /// 
    /// 1. It starts with a random vector
    /// 
    /// 2. It repeatedly multiplies this vector by the matrix (and its transpose)
    /// 
    /// 3. The vector gradually aligns with the direction of the largest singular value
    /// 
    /// 4. After finding one singular value/vector pair, it can be "deflated" to find the next largest
    /// 
    /// The Power Iteration approach:
    /// 
    /// 1. Is conceptually simple and easy to implement
    /// 
    /// 2. Requires minimal memory
    /// 
    /// 3. Is particularly efficient for sparse matrices (matrices with mostly zeros)
    /// 
    /// 4. Converges quickly to the largest singular values
    /// 
    /// 5. May converge slowly if the largest singular values are close in magnitude
    /// 
    /// This method is particularly valuable when:
    /// 
    /// 1. You only need the few largest singular values and vectors
    /// 
    /// 2. You're working with sparse matrices
    /// 
    /// 3. Memory efficiency is important
    /// 
    /// 4. You need a simple, robust approach
    /// 
    /// In machine learning applications, Power Iteration is useful for tasks like PageRank computation (used by 
    /// Google's search algorithm), finding the principal components in PCA when only a few components are needed, 
    /// or in spectral clustering algorithms.
    /// </para>
    /// </remarks>
    PowerIteration,

    /// <summary>
    /// Uses the Truncated SVD algorithm, which computes only the k largest singular values and their corresponding vectors.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Truncated SVD focuses on computing only a specified number (k) of the largest singular values 
    /// and their corresponding vectors, rather than the complete decomposition.
    /// 
    /// Think of it like summarizing a book by only keeping the most important chapters:
    /// 
    /// 1. It specifically targets the k largest singular values
    /// 
    /// 2. It ignores the smaller singular values that often represent noise or less important information
    /// 
    /// 3. It produces a lower-rank approximation of the original matrix
    /// 
    /// The Truncated SVD approach:
    /// 
    /// 1. Is much faster than computing the full SVD
    /// 
    /// 2. Requires significantly less memory
    /// 
    /// 3. Often captures the most important information in the data
    /// 
    /// 4. Is directly applicable to dimensionality reduction
    /// 
    /// 5. Forms the basis of techniques like Latent Semantic Analysis
    /// 
    /// This method is particularly useful when:
    /// 
    /// 1. You only care about the most significant components
    /// 
    /// 2. You're using SVD for dimensionality reduction
    /// 
    /// 3. You're working with large matrices
    /// 
    /// 4. You want to filter out noise by removing small singular values
    /// 
    /// In machine learning applications, Truncated SVD is widely used for dimensionality reduction in text analysis 
    /// (as in Latent Semantic Analysis), collaborative filtering for recommendation systems, and as a preprocessing 
    /// step to make large datasets more manageable for other algorithms.
    /// </para>
    /// </remarks>
    TruncatedSVD,

    /// <summary>
    /// Uses the Divide and Conquer algorithm for SVD computation, which is efficient for large matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Divide and Conquer approach breaks down a large problem into smaller, more manageable 
    /// sub-problems, solves them separately, and then combines their solutions.
    /// 
    /// Think of it like a team working on a big project:
    /// 
    /// 1. First, the matrix is divided into smaller sub-matrices
    /// 
    /// 2. SVD is computed for each of these smaller matrices (which is faster and easier)
    /// 
    /// 3. These partial results are cleverly combined to form the SVD of the original matrix
    /// 
    /// The Divide and Conquer approach:
    /// 
    /// 1. Is significantly faster than classical methods for large matrices
    /// 
    /// 2. Has excellent numerical stability
    /// 
    /// 3. Can compute the full SVD efficiently
    /// 
    /// 4. Takes advantage of modern computer architectures
    /// 
    /// 5. Works well in parallel computing environments
    /// 
    /// This method is particularly valuable when:
    /// 
    /// 1. You're working with large matrices
    /// 
    /// 2. You need the complete SVD (all singular values and vectors)
    /// 
    /// 3. You want good performance without sacrificing accuracy
    /// 
    /// 4. You have multiple processors or cores available
    /// 
    /// In machine learning applications, the Divide and Conquer approach enables efficient processing of large 
    /// datasets while maintaining high accuracy, making it suitable for applications like image processing, 
    /// natural language processing, and large-scale data analysis where both performance and precision are important.
    /// </para>
    /// </remarks>
    DividedAndConquer
}
