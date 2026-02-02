namespace AiDotNet.Kernels;

/// <summary>
/// Implements a Product kernel that combines multiple kernels by multiplying their outputs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Product kernel combines multiple kernels by multiplying their
/// similarity scores. This creates interactions between different pattern types.
///
/// In mathematical terms: k_prod(x, x') = k1(x, x') × k2(x, x') × ... × kn(x, x')
/// </para>
/// <para>
/// Why use Product kernels?
///
/// 1. **Scaling**: Multiply a kernel by a Constant kernel to scale its output
///    - ConstantKernel(c) × RBF = Scaled RBF kernel
///
/// 2. **Interaction effects**: When patterns only appear under certain conditions
///    - RBF × Periodic: Smooth patterns that also vary periodically
///
/// 3. **Automatic Relevance Determination (ARD)**: Create dimension-specific scaling
///    by multiplying kernels over different feature subsets
///
/// 4. **Non-stationary patterns**: Combine with location-dependent kernels
///
/// Key difference from Sum kernel:
/// - Sum: Each kernel contributes independently (additive decomposition)
/// - Product: Kernels interact multiplicatively (AND-like combination)
///
/// Example: RBF × Periodic means "similar if BOTH spatially close AND at similar phase"
/// (requires both conditions), while RBF + Periodic means "similar if EITHER close OR
/// at similar phase" (either condition helps).
/// </para>
/// </remarks>
public class ProductKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The list of kernels to multiply together.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This stores all the kernels that will be multiplied.
    /// The final similarity is only high when ALL component kernels give high values.
    /// </para>
    /// </remarks>
    private readonly IKernelFunction<T>[] _kernels;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new Product kernel from an array of kernels.
    /// </summary>
    /// <param name="kernels">The kernels to combine by multiplication.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a kernel that multiplies the outputs of all provided kernels.
    ///
    /// Example usage:
    /// var productKernel = new ProductKernel&lt;double&gt;(
    ///     new ConstantKernel&lt;double&gt;(scale: 2.0),
    ///     new GaussianKernel&lt;double&gt;(sigma: 1.0)
    /// );
    ///
    /// This creates a scaled RBF kernel with variance 2.0 × the base RBF variance.
    /// </para>
    /// </remarks>
    public ProductKernel(params IKernelFunction<T>[] kernels)
    {
        if (kernels is null || kernels.Length == 0)
            throw new ArgumentException("At least one kernel must be provided.", nameof(kernels));

        _kernels = kernels;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Calculates the Product kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The product of all component kernel values.</returns>
    /// <exception cref="ArgumentException">Thrown when vectors have different lengths.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes the similarity by multiplying the similarities
    /// from each component kernel.
    ///
    /// For example, with Constant(2) × RBF:
    /// - Constant returns 2.0
    /// - RBF might return 0.8
    /// - Total: 2.0 × 0.8 = 1.6
    ///
    /// The multiplicative structure means that if ANY kernel returns zero (or near-zero),
    /// the total similarity is also zero. This creates "AND" logic: points must be similar
    /// according to ALL component kernels.
    ///
    /// This is particularly useful for:
    /// - Scaling kernels (multiply by constant)
    /// - Creating kernels that require multiple conditions
    /// - Building complex covariance structures
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        if (x1.Length != x2.Length)
            throw new ArgumentException("Vectors must have the same length.");

        T product = _numOps.One;
        foreach (var kernel in _kernels)
        {
            product = _numOps.Multiply(product, kernel.Calculate(x1, x2));
        }

        return product;
    }

    /// <summary>
    /// Gets the component kernels in this product.
    /// </summary>
    /// <returns>Array of component kernels.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This lets you access the individual kernels that make up the product.
    /// Useful for inspecting or modifying the kernel structure.
    /// </para>
    /// </remarks>
    public IKernelFunction<T>[] GetKernels() => _kernels;
}
