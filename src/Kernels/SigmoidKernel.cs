namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Sigmoid kernel for measuring similarity between data points.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The Sigmoid kernel is derived from the neural network field and is related to the activation
/// function used in artificial neural networks. It can be used to model certain types of non-linear
/// relationships in data.
/// </para>
/// <para>
/// <b>For Beginners:</b> A kernel function is a mathematical tool that measures how similar two data points are.
/// The Sigmoid kernel is special because it can capture certain types of relationships that other
/// kernels can't. It's inspired by how neurons in artificial neural networks work.
/// </para>
/// <para>
/// Think of it like this: The Sigmoid kernel looks at how two data points interact with each other
/// (their dot product) and then transforms this interaction using a special S-shaped curve (the hyperbolic
/// tangent function). This helps it detect complex patterns in your data.
/// </para>
/// <para>
/// The formula for the Sigmoid kernel is:
/// k(x, y) = tanh(a(x·y) + c)
/// where:
/// - x and y are the two data points being compared
/// - x·y is the dot product between them
/// - a (alpha) controls the steepness of the S-curve
/// - c is a parameter that shifts the curve horizontally
/// - tanh is the hyperbolic tangent function (an S-shaped curve)
/// </para>
/// <para>
/// Common uses include:
/// - Text classification problems
/// - Some types of image recognition tasks
/// - Problems where the data might have complex, non-linear relationships
/// </para>
/// <para>
/// Note: Unlike many other kernels, the Sigmoid kernel is not always positive definite, which means
/// it might not work well with all machine learning algorithms. It's most commonly used with
/// Support Vector Machines.
/// </para>
/// </remarks>
public class SigmoidKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The scaling parameter that controls the steepness of the sigmoid curve.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The alpha parameter controls how steep the S-curve is:
    /// 
    /// Think of it like this:
    /// - Larger alpha values (e.g., 2.0): Make the curve steeper, creating sharper transitions
    /// - Smaller alpha values (e.g., 0.5): Make the curve more gradual, creating smoother transitions
    /// 
    /// The default value is 1.0, which provides a balanced sensitivity for many applications.
    /// </remarks>
    private readonly T _alpha;

    /// <summary>
    /// The shifting parameter that moves the sigmoid curve horizontally.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The c parameter shifts the S-curve left or right:
    /// 
    /// Think of it like this:
    /// - Positive c values: Shift the curve to the left
    /// - Negative c values: Shift the curve to the right
    /// - Zero (the default): Centers the curve at the origin
    /// 
    /// Adjusting this parameter can help when your data has a natural bias or offset.
    /// </remarks>
    private readonly T _c;

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
    /// Initializes a new instance of the Sigmoid kernel with optional scaling and shifting parameters.
    /// </summary>
    /// <param name="alpha">The scaling parameter that controls the steepness of the sigmoid curve. Default is 1.0.</param>
    /// <param name="c">The shifting parameter that moves the sigmoid curve horizontally. Default is 0.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Sigmoid kernel for use. You can optionally
    /// provide values for the alpha and c parameters:
    /// </para>
    /// <para>
    /// The alpha parameter controls the steepness of the S-curve:
    /// - Larger values make the curve steeper (sharper transitions)
    /// - Smaller values make the curve more gradual (smoother transitions)
    /// - The default value is 1.0, which works well for many applications
    /// </para>
    /// <para>
    /// The c parameter shifts the S-curve horizontally:
    /// - Positive values shift the curve to the left
    /// - Negative values shift the curve to the right
    /// - The default value is 0.0, which centers the curve at the origin
    /// </para>
    /// <para>
    /// When might you want to change these parameters?
    /// - Adjust alpha when you want to control how sensitive the kernel is to changes in similarity
    /// - Adjust c when your data has a natural bias or offset that you want to account for
    /// </para>
    /// <para>
    /// The best values often depend on your specific dataset and problem, so you might
    /// need to experiment with different values to find what works best.
    /// </para>
    /// </remarks>
    public SigmoidKernel(T? alpha = default, T? c = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        // For value types (e.g., double), T? is just T, and default(T) = 0, not null.
        // The ?? operator never triggers. Use explicit checks for the intended defaults.
        bool alphaIsDefault = alpha is null || EqualityComparer<T>.Default.Equals(alpha, default);
        _alpha = alphaIsDefault ? _numOps.FromDouble(1.0) : (alpha ?? _numOps.FromDouble(1.0));
        // c=0.0 is the intended default, so the ?? pattern works correctly here
        _c = c ?? _numOps.FromDouble(0.0);
    }

    /// <summary>
    /// Calculates the Sigmoid kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The kernel value representing the similarity between the two vectors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method takes two data points (represented as vectors) and calculates
    /// how similar they are to each other using the Sigmoid kernel formula.
    /// </para>
    /// <para>
    /// The calculation works by:
    /// 1. Finding the dot product between the two vectors (a measure of their alignment)
    /// 2. Scaling this value by alpha
    /// 3. Adding the constant c
    /// 4. Applying the hyperbolic tangent function (tanh) to get the final result
    /// </para>
    /// <para>
    /// The result is a similarity measure where:
    /// - Values closer to 1 indicate high positive similarity
    /// - Values closer to -1 indicate high negative similarity
    /// - Values near 0 indicate little similarity
    /// </para>
    /// <para>
    /// What makes this kernel special is that it can capture both positive and negative relationships
    /// between data points, unlike many other kernels that only measure positive similarity.
    /// This makes it useful for certain types of problems, especially those related to neural networks.
    /// </para>
    /// <para>
    /// Note: The Sigmoid kernel doesn't always satisfy the mathematical properties required for all
    /// kernel-based algorithms. It works best with Support Vector Machines and similar methods.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        var dotProduct = x1.DotProduct(x2);
        return MathHelper.Tanh(_numOps.Add(_numOps.Multiply(_alpha, dotProduct), _c));
    }
}
