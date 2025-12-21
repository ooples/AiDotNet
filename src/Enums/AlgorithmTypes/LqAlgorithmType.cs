namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for LQ decomposition of matrices.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LQ decomposition is a way to break down a matrix into two simpler parts that make 
/// calculations much easier and faster.
/// 
/// Imagine you have a complex recipe (the matrix) that you need to follow. LQ decomposition breaks this recipe 
/// into two simpler steps:
/// 
/// 1. L - A lower triangular matrix (has values only on and below the diagonal)
/// 2. Q - An orthogonal matrix (a special type of matrix where columns/rows are perpendicular to each other)
/// 
/// So instead of following one complex recipe, you can follow two simpler ones in sequence.
/// 
/// Why is this important in AI and machine learning?
/// 
/// 1. Solving Least Squares Problems: Many machine learning algorithms involve finding the best fit for data, 
///    which often requires solving least squares problems.
/// 
/// 2. Dimensionality Reduction: LQ decomposition can help reduce the number of features in your data while 
///    preserving important information.
/// 
/// 3. Data Transformation: It allows you to transform your data into a more useful form for analysis.
/// 
/// 4. Numerical Stability: LQ decomposition provides a stable way to perform calculations that might otherwise 
///    be prone to errors.
/// 
/// 5. Feature Extraction: It can help identify the most important features in your data.
/// 
/// LQ decomposition is closely related to QR decomposition (which is more commonly discussed), but LQ works 
/// on the rows of a matrix rather than the columns. Think of it as QR decomposition's "mirror image."
/// 
/// This enum specifies which specific algorithm to use for performing the LQ decomposition, as different 
/// methods have different performance characteristics depending on the matrix properties.
/// </para>
/// </remarks>
public enum LqAlgorithmType
{
    /// <summary>
    /// Uses Householder reflections to compute the LQ factorization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Householder method uses special mathematical operations called "reflections" 
    /// to gradually transform the original matrix into the L and Q components.
    /// 
    /// Imagine you're folding origami - each fold (reflection) transforms your paper in a specific way. 
    /// After a series of carefully chosen folds, you end up with your desired shape. Householder reflections 
    /// work similarly to transform matrices.
    /// 
    /// The Householder method:
    /// 
    /// 1. Is numerically very stable (resistant to calculation errors)
    /// 
    /// 2. Works efficiently for most types of matrices
    /// 
    /// 3. Requires less computation than other methods for large matrices
    /// 
    /// 4. Preserves the structure of sparse matrices better than some alternatives
    /// 
    /// This is typically the default choice for LQ decomposition in most applications because it offers 
    /// an excellent balance of stability, accuracy, and efficiency.
    /// 
    /// In machine learning, this is particularly useful for solving large least squares problems, such as 
    /// in linear regression with many features, or when processing large datasets where numerical stability 
    /// is important.
    /// </para>
    /// </remarks>
    Householder,

    /// <summary>
    /// Uses the Gram-Schmidt process to compute the LQ factorization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Gram-Schmidt method works by taking the rows of your matrix and making them 
    /// perpendicular (orthogonal) to each other, one at a time.
    /// 
    /// Imagine you're arranging furniture in a room. You place the first piece wherever you want. For the 
    /// second piece, you make sure it's aligned with the walls (perpendicular). For the third piece, you 
    /// make sure it's aligned with both the walls and doesn't interfere with the previous pieces. 
    /// Gram-Schmidt works similarly by "aligning" each row with respect to previous rows.
    /// 
    /// The Gram-Schmidt method:
    /// 
    /// 1. Is conceptually simpler and easier to understand than other methods
    /// 
    /// 2. Works well for small matrices or when you need to explain the process
    /// 
    /// 3. Is more straightforward to implement from scratch
    /// 
    /// 4. Can be modified to work incrementally (row by row)
    /// 
    /// However, the classical Gram-Schmidt method can suffer from numerical instability with larger matrices. 
    /// Modern implementations use a modified version that improves stability.
    /// 
    /// In machine learning, this method is useful for educational purposes, for smaller problems, or when 
    /// you need to process data incrementally (one row at a time) rather than all at once.
    /// </para>
    /// </remarks>
    GramSchmidt,

    /// <summary>
    /// Uses Givens rotations to compute the LQ factorization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Givens method uses a series of simple rotations to gradually transform the 
    /// original matrix into the L and Q components.
    /// 
    /// Imagine you're adjusting the position of a picture frame on a wall. Instead of moving it all at once, 
    /// you make small rotations - a little to the left, a little up, etc. - until it's perfectly aligned. 
    /// Givens rotations work similarly by making a series of simple rotations to transform the matrix.
    /// 
    /// The Givens method:
    /// 
    /// 1. Is very precise and numerically stable
    /// 
    /// 2. Can target specific elements of the matrix (unlike Householder which affects entire rows)
    /// 
    /// 3. Works well for sparse matrices (matrices with lots of zeros)
    /// 
    /// 4. Is excellent for making small updates to an existing decomposition
    /// 
    /// 5. Can be easily parallelized for certain matrix structures
    /// 
    /// While generally slower than Householder for dense matrices, Givens rotations excel when working with 
    /// sparse matrices or when you need to update an existing decomposition without recalculating everything.
    /// 
    /// In machine learning, this is particularly useful for online learning scenarios where data arrives 
    /// sequentially and you want to update your model incrementally, or when working with sparse feature 
    /// matrices common in text analysis or recommendation systems.
    /// </para>
    /// </remarks>
    Givens
}
