namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for Cholesky decomposition of matrices.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Cholesky decomposition is a way to break down a special type of matrix (called a 
/// symmetric positive-definite matrix) into simpler parts that make calculations faster and more stable.
/// 
/// In simple terms, it's like factoring a number (e.g., 12 = 3 × 4), but for matrices. The Cholesky 
/// decomposition factors a matrix into a lower triangular matrix and its transpose (mirror image).
/// 
/// Why is this useful? Many problems in machine learning, statistics, and optimization require solving 
/// systems of equations. Cholesky decomposition makes these calculations much faster and more accurate.
/// 
/// For example, when fitting a linear regression model, calculating the weights often involves Cholesky 
/// decomposition behind the scenes. It's also used in Monte Carlo simulations, Kalman filters, and many 
/// other algorithms.
/// 
/// This enum lists different mathematical approaches to perform Cholesky decomposition, each with its own 
/// advantages depending on the specific problem you're solving.
/// </para>
/// </remarks>
public enum CholeskyAlgorithmType
{
    /// <summary>
    /// Uses the Crout algorithm for Cholesky decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Crout method computes the Cholesky decomposition by working through the matrix 
    /// one column at a time.
    /// 
    /// Imagine building a pyramid brick by brick, where you complete each column from bottom to top before 
    /// moving to the next column. This approach is straightforward to implement and understand.
    /// 
    /// The Crout method is often taught in introductory linear algebra courses because of its clarity, 
    /// though it may not always be the most computationally efficient for very large matrices.
    /// </para>
    /// </remarks>
    Crout,

    /// <summary>
    /// Uses the Banachiewicz algorithm for Cholesky decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Banachiewicz method computes the Cholesky decomposition by working through the 
    /// matrix one element at a time in a specific order.
    /// 
    /// Unlike the Crout method which works column by column, Banachiewicz calculates each element using 
    /// previously computed elements. It's like solving a puzzle where each piece depends on pieces you've 
    /// already placed.
    /// 
    /// This method is often more efficient for certain types of matrices and is commonly used in many 
    /// numerical software packages. It's particularly good for dense matrices (matrices with few zero values).
    /// </para>
    /// </remarks>
    Banachiewicz,

    /// <summary>
    /// Uses the LDL decomposition, a variant of Cholesky decomposition.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The LDL method is a variation of Cholesky decomposition that breaks the matrix into 
    /// three parts: a lower triangular matrix (L), a diagonal matrix (D), and the transpose of L.
    /// 
    /// Think of it as organizing a company into departments (L), identifying the key decision-makers in 
    /// each department (D), and then showing how departments communicate with each other (L transpose).
    /// 
    /// The advantage of LDL is that it can handle certain matrices that the standard Cholesky decomposition 
    /// struggles with, particularly those that are "nearly" positive-definite. It's also more numerically 
    /// stable in some cases, meaning it's less likely to introduce small errors during calculations.
    /// </para>
    /// </remarks>
    LDL,

    /// <summary>
    /// Uses a block-based approach to Cholesky decomposition for large matrices.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Block Cholesky method divides the original large matrix into smaller "blocks" 
    /// or sub-matrices, and then performs Cholesky decomposition on these blocks.
    /// 
    /// Imagine breaking a large jigsaw puzzle into smaller, more manageable sections, solving each section, 
    /// and then combining them back together.
    /// 
    /// This approach is particularly efficient for very large matrices because:
    /// 1. It can take advantage of modern computer architectures that process blocks of data efficiently
    /// 2. It allows for parallel processing (solving multiple blocks at the same time)
    /// 3. It can make better use of computer memory by working with smaller pieces at a time
    /// 
    /// Block Cholesky is often used in high-performance computing applications where the matrices are too 
    /// large to process efficiently as a single unit.
    /// </para>
    /// </remarks>
    BlockCholesky
}
