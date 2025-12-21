namespace AiDotNet.Enums.AlgorithmTypes;

/// <summary>
/// Represents different algorithm types for the Gram-Schmidt orthogonalization process.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Gram-Schmidt process is a method for converting a set of vectors into a set of 
/// orthogonal vectors (vectors that are perpendicular to each other).
/// 
/// Imagine you're in a room with walls that aren't at right angles to each other. The Gram-Schmidt process 
/// is like rearranging these walls so they're all perpendicular, making the room easier to measure and work with.
/// 
/// Why is this important in AI and machine learning?
/// 
/// 1. Feature Independence: In machine learning, we often want features that are independent of each other. 
///    Orthogonal vectors represent completely independent features, which can improve model performance.
/// 
/// 2. Dimensionality Reduction: Techniques like Principal Component Analysis (PCA) use orthogonalization to 
///    find the most important directions in your data.
/// 
/// 3. Numerical Stability: Many algorithms work better when using orthogonal vectors because calculations 
///    become more stable and accurate.
/// 
/// 4. Solving Systems of Equations: Orthogonal vectors make solving certain mathematical problems much easier.
/// 
/// The Gram-Schmidt process takes a set of vectors and, one by one, makes each new vector perpendicular to 
/// all previous vectors. This creates a new set of vectors that span the same space but are all perpendicular 
/// to each other.
/// 
/// This enum specifies which variation of the Gram-Schmidt algorithm to use, as there are different 
/// implementations with different numerical properties.
/// </para>
/// </remarks>
public enum GramSchmidtAlgorithmType
{
    /// <summary>
    /// Uses the Classical Gram-Schmidt algorithm for orthogonalization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Classical Gram-Schmidt algorithm is the original and most straightforward 
    /// implementation of the orthogonalization process.
    /// 
    /// It works by taking each vector in sequence and subtracting from it any components that point in the 
    /// same direction as the previously orthogonalized vectors. This leaves only the component that's 
    /// perpendicular to all previous vectors.
    /// 
    /// Think of it like building a team where each new person brings only skills that nobody else on the 
    /// team has yet. The first person brings all their skills, but the second person only contributes skills 
    /// that the first person doesn't have, and so on.
    /// 
    /// The Classical method is simple to understand and implement. However, it can suffer from numerical 
    /// instability when used with floating-point arithmetic on computers, especially for large sets of vectors. 
    /// This means small rounding errors can accumulate and cause the vectors to not be perfectly orthogonal.
    /// 
    /// It's often suitable for educational purposes and for problems where high precision isn't critical or 
    /// where the vectors are known to be well-behaved.
    /// </para>
    /// </remarks>
    Classical,

    /// <summary>
    /// Uses the Modified Gram-Schmidt algorithm for orthogonalization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Modified Gram-Schmidt algorithm is a more numerically stable version of the 
    /// orthogonalization process.
    /// 
    /// While the Classical method computes all projections using the original vectors, the Modified method 
    /// updates the vectors after each projection. This reduces the accumulation of rounding errors in 
    /// computer calculations.
    /// 
    /// Imagine washing a very dirty cloth. The Classical method is like soaking the cloth once in clean water 
    /// and expecting all the dirt to come out. The Modified method is like rinsing the cloth multiple times 
    /// with fresh water after each rinse - it's more thorough and gets better results.
    /// 
    /// The Modified Gram-Schmidt algorithm is preferred in most practical applications, especially when:
    /// 
    /// 1. Working with large datasets
    /// 
    /// 2. High precision is required
    /// 
    /// 3. The vectors might be nearly linearly dependent (almost pointing in the same direction)
    /// 
    /// 4. The calculations are part of a larger algorithm where errors could propagate
    /// 
    /// Most professional software libraries use the Modified Gram-Schmidt algorithm by default because of its 
    /// superior numerical properties.
    /// </para>
    /// </remarks>
    Modified
}
