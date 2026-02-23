namespace AiDotNet.Kernels;

/// <summary>
/// A wrapper kernel that scales another kernel by a constant factor (output scale/variance).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The ScaleKernel is a simple but essential building block for GPs.
/// It wraps any base kernel and multiplies its output by a constant scale factor.
///
/// In mathematical terms:
/// k_scaled(x, x') = σ² × k_base(x, x')
///
/// Where:
/// - σ² is the output scale (variance) parameter
/// - k_base is the underlying kernel function
///
/// Why is this useful?
/// 1. **Separates concerns**: The base kernel handles correlation structure,
///    while the scale controls the magnitude of variation
/// 2. **Better optimization**: Having an explicit scale parameter often helps
///    hyperparameter optimization converge more easily
/// 3. **Interpretability**: The scale directly relates to the variance of your output
///
/// Example: If your data varies between 0 and 100:
/// - Without scaling, the RBF kernel might not fit well
/// - With ScaleKernel(outputScale: 2500), the GP knows to expect variance of ~2500
/// </para>
/// <para>
/// Common usage patterns:
/// - ScaleKernel(rbfKernel, outputScale: variance_of_your_data)
/// - ScaleKernel(maternKernel, outputScale: 1.0) and optimize the scale
/// </para>
/// </remarks>
public class ScaleKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The base kernel being scaled.
    /// </summary>
    private readonly IKernelFunction<T> _baseKernel;

    /// <summary>
    /// The output scale factor (σ²).
    /// </summary>
    private double _outputScale;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new ScaleKernel with the specified base kernel and output scale.
    /// </summary>
    /// <param name="baseKernel">The kernel to scale.</param>
    /// <param name="outputScale">The output scale factor (variance). Default is 1.0.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a scaled version of any kernel.
    ///
    /// Example:
    /// var rbf = new GaussianKernel&lt;double&gt;(lengthScale: 1.0);
    /// var scaledRbf = new ScaleKernel&lt;double&gt;(rbf, outputScale: 2.0);
    /// // Now the kernel outputs values ~2x larger
    /// </para>
    /// </remarks>
    public ScaleKernel(IKernelFunction<T> baseKernel, double outputScale = 1.0)
    {
        if (baseKernel is null) throw new ArgumentNullException(nameof(baseKernel));
        if (outputScale <= 0)
            throw new ArgumentException("Output scale must be positive.", nameof(outputScale));

        _baseKernel = baseKernel;
        _outputScale = outputScale;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Gets the base kernel being scaled.
    /// </summary>
    public IKernelFunction<T> BaseKernel => _baseKernel;

    /// <summary>
    /// Gets or sets the output scale factor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the σ² parameter that multiplies the base kernel.
    /// Setting this to a larger value makes the GP predict larger variations.
    /// </para>
    /// </remarks>
    public double OutputScale
    {
        get => _outputScale;
        set
        {
            if (value <= 0)
                throw new ArgumentException("Output scale must be positive.");
            _outputScale = value;
        }
    }

    /// <summary>
    /// Calculates the scaled kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The scaled kernel value: outputScale × baseKernel(x1, x2).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Simply computes the base kernel value and multiplies by the scale.
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        T baseValue = _baseKernel.Calculate(x1, x2);
        return _numOps.FromDouble(_outputScale * _numOps.ToDouble(baseValue));
    }

    /// <summary>
    /// Computes the gradient of the kernel with respect to the scale parameter.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The gradient dk/d(outputScale).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The gradient with respect to the scale is simply the base kernel value.
    /// This is used during hyperparameter optimization.
    ///
    /// Since k = σ² × k_base, we have dk/dσ² = k_base
    /// </para>
    /// </remarks>
    public double CalculateScaleGradient(Vector<T> x1, Vector<T> x2)
    {
        return _numOps.ToDouble(_baseKernel.Calculate(x1, x2));
    }

    /// <summary>
    /// Creates a ScaleKernel from an RBF (Gaussian) kernel with specified parameters.
    /// </summary>
    /// <param name="lengthScale">The length scale for the RBF kernel.</param>
    /// <param name="outputScale">The output scale (variance).</param>
    /// <returns>A new ScaleKernel wrapping an RBF kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a convenience method for the most common kernel setup:
    /// a scaled RBF (Gaussian) kernel.
    ///
    /// The resulting kernel is: σ² × exp(-r²/(2l²))
    /// - σ² = outputScale (controls amplitude)
    /// - l = lengthScale (controls smoothness)
    /// </para>
    /// </remarks>
    public static ScaleKernel<T> WithRBF(double lengthScale = 1.0, double outputScale = 1.0)
    {
        var rbf = new GaussianKernel<T>(lengthScale);
        return new ScaleKernel<T>(rbf, outputScale);
    }

    /// <summary>
    /// Creates a ScaleKernel from a Matern kernel with specified parameters.
    /// </summary>
    /// <param name="nu">The smoothness parameter for the Matern kernel.</param>
    /// <param name="lengthScale">The length scale for the Matern kernel.</param>
    /// <param name="outputScale">The output scale (variance).</param>
    /// <returns>A new ScaleKernel wrapping a Matern kernel.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a scaled Matern kernel, which is often preferred
    /// over RBF for its more realistic smoothness assumptions.
    ///
    /// Common nu values:
    /// - 0.5: Ornstein-Uhlenbeck (rough)
    /// - 1.5: Once differentiable
    /// - 2.5: Twice differentiable (good default)
    /// </para>
    /// </remarks>
    public static ScaleKernel<T> WithMatern(double nu = 2.5, double lengthScale = 1.0, double outputScale = 1.0)
    {
        var matern = new MaternKernel<T>(nu, lengthScale);
        return new ScaleKernel<T>(matern, outputScale);
    }
}
