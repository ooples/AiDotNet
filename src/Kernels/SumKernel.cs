namespace AiDotNet.Kernels;

/// <summary>
/// Implements a Sum kernel that combines multiple kernels by adding their outputs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Sum kernel allows you to combine multiple kernels by adding
/// their similarity scores together. This creates a more expressive model that can capture
/// different types of patterns simultaneously.
///
/// In mathematical terms: k_sum(x, x') = k1(x, x') + k2(x, x') + ... + kn(x, x')
/// </para>
/// <para>
/// Why use Sum kernels?
///
/// 1. **Multiple pattern types**: Combine a smooth kernel (RBF) with a periodic kernel
///    to model data with both smooth trends and seasonal patterns.
///
/// 2. **Additive structure**: If your function can be decomposed into additive components,
///    each handled by a different kernel.
///
/// 3. **Noise modeling**: Add a WhiteNoise kernel to model observation noise.
///
/// Example combinations:
/// - RBF + WhiteNoise: Smooth function with measurement noise
/// - RBF + Periodic: Smooth trend with seasonal variation
/// - Linear + RBF: Linear trend with non-linear deviations
/// - RBF(short) + RBF(long): Capture both local and global patterns
/// </para>
/// </remarks>
public class SumKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The list of kernels to sum together.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This stores all the kernels that will be combined.
    /// Each kernel contributes its own similarity measure to the final result.
    /// </para>
    /// </remarks>
    private readonly IKernelFunction<T>[] _kernels;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Sum kernel from an array of kernels.
    /// </summary>
    /// <param name="kernels">The kernels to combine by summation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a kernel that sums the outputs of all provided kernels.
    ///
    /// Example usage:
    /// var sumKernel = new SumKernel&lt;double&gt;(
    ///     new GaussianKernel&lt;double&gt;(sigma: 1.0),
    ///     new WhiteNoiseKernel&lt;double&gt;(noiseVariance: 0.1)
    /// );
    ///
    /// This creates a kernel that models smooth patterns (RBF) plus noise.
    /// </para>
    /// </remarks>
    public SumKernel(params IKernelFunction<T>[] kernels)
    {
        if (kernels is null || kernels.Length == 0)
            throw new ArgumentException("At least one kernel must be provided.", nameof(kernels));

        _kernels = kernels;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Sum kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The sum of all component kernel values.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes the similarity by adding up the similarities
    /// from each component kernel.
    ///
    /// For example, with RBF + WhiteNoise:
    /// - RBF might return 0.8 (points are somewhat close)
    /// - WhiteNoise returns 0.1 if same point, 0 otherwise
    /// - Total: 0.8 + 0.1 = 0.9 (for same point) or 0.8 (for different points)
    ///
    /// This additive structure means each kernel independently contributes to the total,
    /// allowing the GP to decompose the function into interpretable components.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        T sum = _numOps.Zero;
        foreach (var kernel in _kernels)
        {
            sum = _numOps.Add(sum, kernel.Calculate(x1, x2));
        }

        return sum;
    }

    /// <summary>
    /// Gets the component kernels in this sum.
    /// </summary>
    /// <returns>Array of component kernels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you access the individual kernels that make up the sum.
    /// Useful for inspecting or modifying the kernel structure.
    /// </para>
    /// </remarks>
    public IKernelFunction<T>[] GetKernels() => _kernels;
}
