namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Tanimoto kernel (also known as the Jaccard kernel) for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Tanimoto kernel is a similarity measure that is particularly useful for binary data
/// and chemical fingerprints, but can be applied to any vector data. It measures the ratio
/// of the intersection to the union of the features.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Tanimoto kernel is special because it focuses on the overlap between features relative to their
/// combined magnitude. It's like measuring how much two sets have in common compared to their total size.
/// </para>
/// <para>
/// Think of it like this: Imagine you have two shopping lists. The Tanimoto kernel would measure
/// how many items appear on both lists (the overlap) divided by the total number of unique items
/// across both lists. A value of 1 means the lists are identical, while a value close to 0 means
/// they have very little in common.
/// </para>
/// <para>
/// The formula for the Tanimoto kernel is:
/// k(x, y) = (x · y) / (||x||² + ||y||² - x · y)
/// where:
/// - x · y is the dot product between vectors x and y
/// - ||x||² is the squared norm of vector x (dot product of x with itself)
/// - ||y||² is the squared norm of vector y (dot product of y with itself)
/// </para>
/// <para>
/// Common uses include:
/// - Comparing chemical compounds (molecular fingerprints)
/// - Document similarity in text analysis
/// - Binary feature vectors in machine learning
/// - Recommendation systems
/// </para>
/// </remarks>
public class TanimotoKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that allows the kernel to perform mathematical
    /// operations regardless of what numeric type (like double, float, decimal) you're using.
    /// You don't need to interact with this directly.
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Tanimoto kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Tanimoto kernel for use. Unlike some other
    /// kernels, the Tanimoto kernel doesn't have any parameters that you need to set - it works
    /// right out of the box.
    /// </para>
    /// <para>
    /// The Tanimoto kernel is particularly good at comparing:
    /// - Binary data (data consisting of 0s and 1s)
    /// - Chemical fingerprints (representations of molecular structure)
    /// - Document feature vectors (representations of text documents)
    /// </para>
    /// <para>
    /// When might you want to use this kernel?
    /// - When your data represents sets or collections of features
    /// - When you're working with binary or sparse data
    /// - When you want to measure similarity based on the overlap between features
    /// - When you're doing chemical similarity analysis
    /// </para>
    /// </remarks>
    public TanimotoKernel()
    {
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Tanimoto kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Tanimoto kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Computing the dot product of the two vectors (a measure of their overlap)
    /// 2. Computing the squared norm of each vector (dot product with itself)
    /// 3. Dividing the dot product by the sum of the squared norms minus the dot product
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - A value of 1 means the vectors are identical
    /// - A value of 0 means the vectors have no overlap
    /// - Values in between represent partial similarity
    /// </para>
    /// <para>
    /// What is a dot product? It's a way to multiply two vectors together. For each position in the vectors,
    /// you multiply the corresponding values and then add all these products together. For example, the dot
    /// product of [1, 2, 3] and [4, 5, 6] is (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32.
    /// </para>
    /// <para>
    /// What makes this kernel special is its focus on the ratio of shared information to total information,
    /// which makes it particularly useful for comparing sets or binary data.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T dotProduct = x1.DotProduct(x2);
        T x1Norm = x1.DotProduct(x1);
        T x2Norm = x2.DotProduct(x2);

        return _numOps.Divide(dotProduct, _numOps.Subtract(_numOps.Add(x1Norm, x2Norm), dotProduct));
    }
}
