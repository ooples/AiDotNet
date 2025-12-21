namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Circular kernel function for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Circular kernel is a compact (finite support) kernel based on the circular function from
/// statistics. It produces non-zero values only for points within a certain distance of each other,
/// defined by the sigma parameter.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Circular kernel is special because it has a clear "cutoff point" - if two data points are too far
/// apart (farther than sigma), they're considered completely different (similarity = 0).
/// </para>
/// <para>
/// Think of the Circular kernel like a neighborhood with a strict boundary. Points inside the neighborhood
/// have varying degrees of similarity based on how close they are to each other. But once you step outside
/// the neighborhood boundary (defined by sigma), there's no similarity at all.
/// </para>
/// <para>
/// This property makes the Circular kernel useful for problems where you want to focus only on local
/// patterns and completely ignore distant relationships.
/// </para>
/// </remarks>
public class CircularKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The scaling parameter that controls the radius of influence for the kernel.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Think of sigma as the "neighborhood radius" for your similarity measurement.
    /// Any two points that are farther apart than this radius will be considered completely dissimilar
    /// (they'll have a similarity value of 0).
    /// 
    /// Within this radius, points that are closer together will have higher similarity values.
    /// Adjusting sigma lets you control how far the "neighborhood" extends - a larger sigma means
    /// more distant points will still be considered somewhat similar.
    /// </remarks>
    private readonly T _sigma;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Circular kernel with an optional scaling parameter.
    /// </summary>
    /// <param name="sigma">The scaling parameter that controls the radius of influence. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Circular kernel with your chosen settings.
    /// If you don't specify any settings, it will use a default value of 1.0 for sigma,
    /// which works well for many problems.
    /// </para>
    /// <para>
    /// The sigma parameter defines the "neighborhood radius" - points farther apart than sigma
    /// will have zero similarity. Within this radius, the similarity decreases as points get farther apart,
    /// following a specific mathematical curve based on the circular function.
    /// </para>
    /// <para>
    /// If your data points are typically close together, you might want a smaller sigma.
    /// If they're spread out, a larger sigma might work better. If you're just starting out,
    /// you can use the default value and adjust later if needed.
    /// </para>
    /// </remarks>
    public CircularKernel(T? sigma = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _sigma = sigma ?? _numOps.FromDouble(1.0);
    }

    /// <summary>
    /// Calculates the Circular kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Circular kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the Euclidean distance between the two vectors
    /// 2. Normalizing this distance by dividing by sigma
    /// 3. If the normalized distance is greater than 1 (meaning the points are outside the "neighborhood radius"),
    ///    the similarity is 0
    /// 4. Otherwise, the similarity is calculated using a formula based on the arc cosine of the normalized distance
    /// </para>
    /// <para>
    /// The result is a number between 0 and 1, where:
    /// - Values closer to 1 mean the vectors are very similar (close together)
    /// - Values closer to 0 mean the vectors are less similar (farther apart)
    /// - Exactly 0 means the vectors are too far apart to be considered similar at all
    /// </para>
    /// <para>
    /// The circular shape of this kernel's influence gives it its name - if you were to visualize
    /// all points that have the same similarity to a reference point, they would form a circle.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var diff = x1.Subtract(x2);
        var distance = _numOps.Sqrt(diff.DotProduct(diff));
        var normalizedDistance = _numOps.Divide(distance, _sigma);

        if (_numOps.GreaterThan(normalizedDistance, _numOps.One))
        {
            return _numOps.Zero;
        }

        var theta = MathHelper.ArcCos(normalizedDistance);
        return _numOps.Divide(_numOps.Multiply(_numOps.FromDouble(2.0), theta), MathHelper.Pi<T>());
    }
}
