namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Linear kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Linear kernel is the simplest kernel function, which computes the dot product between two vectors.
/// It's equivalent to performing linear regression or linear classification in the original feature space.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Linear kernel is the most basic type of kernel - it simply measures similarity as the dot product
/// between two data points.
/// </para>
/// <para>
/// Think of the dot product as a way to measure how much two vectors "agree" with each other:
/// - If the vectors point in similar directions with similar magnitudes, the dot product will be large and positive.
/// - If the vectors point in opposite directions, the dot product will be negative.
/// - If the vectors are perpendicular (at right angles), the dot product will be zero.
/// </para>
/// <para>
/// The Linear kernel is useful when you believe your data can be separated by a straight line (in 2D),
/// a plane (in 3D), or a hyperplane (in higher dimensions). It's often a good first choice when you're
/// not sure which kernel to use, as it's simple and computationally efficient.
/// </para>
/// <para>
/// Unlike more complex kernels, the Linear kernel doesn't transform your data into a higher-dimensional space.
/// This means it works well when your data already has enough features to be linearly separable, but it might
/// not perform as well when your data requires more complex decision boundaries.
/// </para>
/// </remarks>
public class LinearKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// Calculates the Linear kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Linear kernel formula.
    /// </para>
    /// <para>
    /// The calculation is simply the dot product of the two vectors. The dot product is calculated by:
    /// 1. Multiplying corresponding elements of the two vectors
    /// 2. Adding up all these products
    /// </para>
    /// <para>
    /// For example, if x1 = [1, 2, 3] and x2 = [4, 5, 6], then:
    /// dot product = (1 × 4) + (2 × 5) + (3 × 6) = 4 + 10 + 18 = 32
    /// </para>
    /// <para>
    /// The result can be any number:
    /// - A large positive number suggests the vectors are similar and pointing in the same direction
    /// - A number close to zero suggests the vectors are dissimilar or nearly perpendicular
    /// - A negative number suggests the vectors are pointing in opposite directions
    /// </para>
    /// <para>
    /// Unlike some other kernels that always return values between 0 and 1, the Linear kernel's output
    /// depends on the magnitude of your vectors. This means that if you scale your input data, the kernel
    /// values will also scale.
    /// </para>
    /// <para>
    /// The Linear kernel is particularly useful for text classification, where documents are often represented
    /// as high-dimensional but sparse vectors (like in the bag-of-words model). It's also computationally
    /// efficient, making it suitable for large datasets.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        return x1.DotProduct(x2);
    }
}
