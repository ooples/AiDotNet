namespace AiDotNet.Kernels;

/// <summary>
/// Implements the Constant kernel, which returns a constant value regardless of input.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Constant kernel is the simplest possible kernel - it always returns
/// the same value, no matter what inputs you give it.
///
/// In mathematical terms: k(x, x') = c
///
/// Where c is a constant value (the "constant value" parameter).
/// </para>
/// <para>
/// Why would you want a kernel that ignores the input?
///
/// The Constant kernel is primarily used as a building block for more complex kernels:
///
/// 1. **Scaling other kernels**: When you multiply the Constant kernel with another kernel
///    (like RBF), you can control the overall scale of your predictions.
///    - ConstantKernel(c) * RBFKernel() gives you a scaled RBF kernel
///
/// 2. **Adding bias**: When you add a Constant kernel to another kernel, you're essentially
///    adding a constant offset to all predictions.
///    - RBFKernel() + ConstantKernel(c) adds a "baseline" to your model
///
/// 3. **Modeling a constant mean**: In Gaussian Processes, the Constant kernel represents
///    the assumption that all points have some shared, constant relationship.
///
/// By itself, the Constant kernel assumes that all data points are equally similar to each
/// other - not very useful! But combined with other kernels, it's a powerful tool.
/// </para>
/// </remarks>
public class ConstantKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The constant value returned by this kernel for all inputs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the value the kernel always returns, regardless of the inputs.
    ///
    /// When using the Constant kernel for scaling (multiplying with other kernels):
    /// - Larger values amplify the effect of other kernels
    /// - Smaller values reduce the effect of other kernels
    ///
    /// When using the Constant kernel for bias (adding to other kernels):
    /// - This value represents a "baseline" similarity between all points
    /// - Larger values mean all points share more in common
    /// </para>
    /// </remarks>
    private readonly double _constantValue;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the Constant kernel.
    /// </summary>
    /// <param name="constantValue">The constant value to return. Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the Constant kernel with your chosen value.
    ///
    /// The default value of 1.0 is often used when the Constant kernel is used for scaling,
    /// as multiplying by 1 doesn't change other kernels. You can then adjust this value
    /// during hyperparameter optimization to find the best scale for your problem.
    ///
    /// If you're using the Constant kernel to add a bias, choose a value that represents
    /// your prior belief about the average similarity between all points.
    /// </para>
    /// </remarks>
    public ConstantKernel(double constantValue = 1.0)
    {
        _constantValue = constantValue;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Constant kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector (ignored).</param>
    /// <param name="x2">The second vector (ignored).</param>
    /// <returns>The constant value, regardless of inputs.</returns>
    /// <exception cref="ArgumentException">Thrown when the vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method always returns the same constant value,
    /// no matter what vectors you pass in. The vectors are still checked to have
    /// the same length for consistency with other kernels, but their actual values
    /// don't affect the result.
    ///
    /// This means:
    /// - k([1, 2, 3], [4, 5, 6]) = constantValue
    /// - k([0, 0, 0], [0, 0, 0]) = constantValue
    /// - k([any vector], [any other vector of same length]) = constantValue
    ///
    /// The input validation ensures that when you combine kernels, dimension
    /// mismatches are caught even by the Constant kernel.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        return _numOps.FromDouble(_constantValue);
    }
}
