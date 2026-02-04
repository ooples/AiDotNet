namespace AiDotNet.Kernels;

/// <summary>
/// Kernel that incorporates gradient observations for GPs with derivative information.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Sometimes you have access not just to function values f(x),
/// but also to gradient information ∇f(x) (the rate of change in each direction).
///
/// This is common in:
/// - Bayesian optimization: When evaluating an expensive function, you might also
///   get gradient information "for free" (e.g., via automatic differentiation)
/// - Physics-informed ML: Physical laws often constrain derivatives
/// - Adjoint methods: Gradients are computed alongside function values
///
/// The GradientKernel extends a base kernel to model both values and gradients.
/// If f(x) is a GP with kernel k(x, x'), then:
/// - Cov(f(x), f(x')) = k(x, x')
/// - Cov(∂f/∂xᵢ, f(x')) = ∂k(x, x')/∂xᵢ
/// - Cov(∂f/∂xᵢ, ∂f/∂x'ⱼ) = ∂²k(x, x')/∂xᵢ∂x'ⱼ
///
/// This lets you use gradient observations to improve predictions, especially
/// when gradient observations are cheaper than function observations.
/// </para>
/// <para>
/// Usage: Create with a base kernel (RBF, Matern, etc.) that supports gradients.
/// The resulting kernel operates on extended vectors: [x; gradient_dim_flag; is_gradient_obs].
/// </para>
/// </remarks>
public class GradientKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The base kernel from which derivatives are computed.
    /// </summary>
    private readonly IKernelFunction<T> _baseKernel;

    /// <summary>
    /// The input dimensionality.
    /// </summary>
    private readonly int _inputDim;

    /// <summary>
    /// The length scale parameter (needed for gradient computation).
    /// </summary>
    private readonly double _lengthScale;

    /// <summary>
    /// The kernel type for gradient computation.
    /// </summary>
    private readonly GradientKernelType _kernelType;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Supported base kernel types for gradient computation.
    /// </summary>
    public enum GradientKernelType
    {
        /// <summary>RBF (Gaussian/Squared Exponential) kernel.</summary>
        RBF,
        /// <summary>Matern 5/2 kernel.</summary>
        Matern52,
        /// <summary>Matern 3/2 kernel.</summary>
        Matern32,
        /// <summary>Polynomial kernel.</summary>
        Polynomial
    }

    /// <summary>
    /// Initializes a new GradientKernel.
    /// </summary>
    /// <param name="inputDim">The input dimensionality.</param>
    /// <param name="kernelType">The type of base kernel.</param>
    /// <param name="lengthScale">The length scale parameter.</param>
    /// <param name="outputScale">The output scale (variance).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates a kernel that handles both function values and gradients.
    ///
    /// For RBF kernel: k(x, x') = σ² × exp(-r²/(2l²))
    /// The gradient formulas are analytically derived:
    /// - ∂k/∂xᵢ = -(xᵢ - x'ᵢ)/l² × k(x, x')
    /// - ∂²k/∂xᵢ∂x'ⱼ = (δᵢⱼ/l² - (xᵢ - x'ᵢ)(xⱼ - x'ⱼ)/l⁴) × k(x, x')
    ///
    /// Matern kernels have more complex but still tractable gradient formulas.
    /// </para>
    /// </remarks>
    public GradientKernel(
        int inputDim,
        GradientKernelType kernelType = GradientKernelType.RBF,
        double lengthScale = 1.0,
        double outputScale = 1.0)
    {
        if (inputDim < 1)
            throw new ArgumentException("Input dimension must be at least 1.", nameof(inputDim));
        if (lengthScale <= 0)
            throw new ArgumentException("Length scale must be positive.", nameof(lengthScale));
        if (outputScale <= 0)
            throw new ArgumentException("Output scale must be positive.", nameof(outputScale));
        if (kernelType == GradientKernelType.Polynomial)
            throw new NotSupportedException(
                "Polynomial kernel does not support gradient observations because its derivatives " +
                "are not implemented. Use RBF, Matern52, or Matern32 instead.");

        _inputDim = inputDim;
        _kernelType = kernelType;
        _lengthScale = lengthScale;
        _numOps = MathHelper.GetNumericOperations<T>();

        // Create the appropriate base kernel
        // Note: GaussianKernel uses sigma (length scale), MaternKernel uses (nu, lengthScale, variance)
        _baseKernel = kernelType switch
        {
            GradientKernelType.RBF => new GaussianKernel<T>(lengthScale),
            GradientKernelType.Matern52 => new MaternKernel<T>(2.5, lengthScale, outputScale),
            GradientKernelType.Matern32 => new MaternKernel<T>(1.5, lengthScale, outputScale),
            _ => new GaussianKernel<T>(lengthScale)
        };
    }

    /// <summary>
    /// Gets the input dimensionality.
    /// </summary>
    public int InputDim => _inputDim;

    /// <summary>
    /// Gets the kernel type.
    /// </summary>
    public GradientKernelType KernelType => _kernelType;

    /// <summary>
    /// Gets the length scale.
    /// </summary>
    public double LengthScale => _lengthScale;

    /// <summary>
    /// Gets the base kernel.
    /// </summary>
    public IKernelFunction<T> BaseKernel => _baseKernel;

    /// <summary>
    /// Calculates the kernel value between two points (value-value, value-gradient, or gradient-gradient).
    /// </summary>
    /// <param name="x1">First extended vector: [input..., gradientDim (or -1 for value)]</param>
    /// <param name="x2">Second extended vector: [input..., gradientDim (or -1 for value)]</param>
    /// <returns>The appropriate kernel value or gradient covariance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The extended vector format encodes whether this is a value or gradient observation.
    /// - gradientDim = -1: This is a function value observation
    /// - gradientDim = 0, 1, 2, ...: This is a gradient observation for dimension gradientDim
    ///
    /// The kernel returns:
    /// - k(x, x') if both are values
    /// - ∂k/∂xᵢ if one is value, one is gradient in dim i
    /// - ∂²k/∂xᵢ∂x'ⱼ if both are gradients
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        // Expected format: [input_0, input_1, ..., input_{d-1}, gradientDim]
        // gradientDim = -1 for value observation
        // gradientDim = 0, 1, ..., d-1 for gradient observation in that dimension

        int len1 = x1.Length;
        int len2 = x2.Length;
        int expectedLen = _inputDim + 1;

        if (len1 != expectedLen)
            throw new ArgumentException($"x1 must have length {expectedLen} (inputDim + 1), got {len1}.", nameof(x1));
        if (len2 != expectedLen)
            throw new ArgumentException($"x2 must have length {expectedLen} (inputDim + 1), got {len2}.", nameof(x2));

        int gradDim1 = (int)Math.Round(_numOps.ToDouble(x1[len1 - 1]));
        int gradDim2 = (int)Math.Round(_numOps.ToDouble(x2[len2 - 1]));

        // Validate gradient dimensions: -1 for value, or [0, inputDim-1] for gradient
        if (gradDim1 < -1 || gradDim1 >= _inputDim)
            throw new ArgumentOutOfRangeException(nameof(x1),
                $"Gradient dimension {gradDim1} in x1 must be -1 (value) or in [0, {_inputDim - 1}].");
        if (gradDim2 < -1 || gradDim2 >= _inputDim)
            throw new ArgumentOutOfRangeException(nameof(x2),
                $"Gradient dimension {gradDim2} in x2 must be -1 (value) or in [0, {_inputDim - 1}].");

        // Extract input parts
        var input1 = ExtractInput(x1);
        var input2 = ExtractInput(x2);

        // Determine which type of covariance to compute
        bool isValue1 = gradDim1 < 0;
        bool isValue2 = gradDim2 < 0;

        if (isValue1 && isValue2)
        {
            // Value-Value: k(x, x')
            return _baseKernel.Calculate(input1, input2);
        }
        else if (isValue1)
        {
            // Value-Gradient: Cov(f(x), ∂f/∂x'_j) = ∂k/∂x'_j
            return _numOps.FromDouble(ComputeFirstDerivative(input1, input2, gradDim2, false));
        }
        else if (isValue2)
        {
            // Gradient-Value: Cov(∂f/∂x_i, f(x')) = ∂k/∂x_i
            return _numOps.FromDouble(ComputeFirstDerivative(input1, input2, gradDim1, true));
        }
        else
        {
            // Gradient-Gradient: Cov(∂f/∂x_i, ∂f/∂x'_j) = ∂²k/∂x_i∂x'_j
            return _numOps.FromDouble(ComputeSecondDerivative(input1, input2, gradDim1, gradDim2));
        }
    }

    /// <summary>
    /// Extracts the input part from an extended vector (removes the gradient dimension flag).
    /// </summary>
    private Vector<T> ExtractInput(Vector<T> extended)
    {
        var result = new Vector<T>(extended.Length - 1);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = extended[i];
        }
        return result;
    }

    /// <summary>
    /// Computes the first derivative of the kernel with respect to one input.
    /// </summary>
    /// <param name="x1">First input.</param>
    /// <param name="x2">Second input.</param>
    /// <param name="dim">Dimension to differentiate.</param>
    /// <param name="wrtFirst">True to differentiate w.r.t. x1, false for x2.</param>
    /// <returns>The first derivative ∂k/∂x_dim.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes how the kernel changes as you move one of the inputs
    /// in a particular direction. For RBF kernel:
    /// ∂k/∂x_i = -(x_i - x'_i)/l² × k(x, x')
    /// </para>
    /// </remarks>
    private double ComputeFirstDerivative(Vector<T> x1, Vector<T> x2, int dim, bool wrtFirst)
    {
        double k = _numOps.ToDouble(_baseKernel.Calculate(x1, x2));
        double xi = _numOps.ToDouble(x1[dim]);
        double xi_prime = _numOps.ToDouble(x2[dim]);
        double diff = xi - xi_prime;
        double l2 = _lengthScale * _lengthScale;

        double result = _kernelType switch
        {
            GradientKernelType.RBF => ComputeRBFFirstDerivative(k, diff, l2, wrtFirst),
            GradientKernelType.Matern52 => ComputeMatern52FirstDerivative(x1, x2, dim, wrtFirst),
            GradientKernelType.Matern32 => ComputeMatern32FirstDerivative(x1, x2, dim, wrtFirst),
            _ => ComputeRBFFirstDerivative(k, diff, l2, wrtFirst)
        };

        return result;
    }

    /// <summary>
    /// Computes the second derivative of the kernel (mixed partial).
    /// </summary>
    /// <param name="x1">First input.</param>
    /// <param name="x2">Second input.</param>
    /// <param name="dim1">Dimension for x1.</param>
    /// <param name="dim2">Dimension for x2.</param>
    /// <returns>The second derivative ∂²k/∂x_dim1 ∂x'_dim2.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This computes the cross-covariance between gradient observations
    /// in different dimensions. For RBF kernel:
    /// ∂²k/∂x_i∂x'_j = (δ_ij/l² - (x_i - x'_i)(x_j - x'_j)/l⁴) × k(x, x')
    /// </para>
    /// </remarks>
    private double ComputeSecondDerivative(Vector<T> x1, Vector<T> x2, int dim1, int dim2)
    {
        double k = _numOps.ToDouble(_baseKernel.Calculate(x1, x2));
        double l2 = _lengthScale * _lengthScale;
        double l4 = l2 * l2;

        double diff_i = _numOps.ToDouble(x1[dim1]) - _numOps.ToDouble(x2[dim1]);
        double diff_j = _numOps.ToDouble(x1[dim2]) - _numOps.ToDouble(x2[dim2]);

        double result = _kernelType switch
        {
            GradientKernelType.RBF => ComputeRBFSecondDerivative(k, diff_i, diff_j, dim1, dim2, l2, l4),
            GradientKernelType.Matern52 => ComputeMatern52SecondDerivative(x1, x2, dim1, dim2),
            GradientKernelType.Matern32 => ComputeMatern32SecondDerivative(x1, x2, dim1, dim2),
            _ => ComputeRBFSecondDerivative(k, diff_i, diff_j, dim1, dim2, l2, l4)
        };

        return result;
    }

    /// <summary>
    /// RBF first derivative: ∂k/∂x_i = -(x_i - x'_i)/l² × k(x, x')
    /// </summary>
    private static double ComputeRBFFirstDerivative(double k, double diff, double l2, bool wrtFirst)
    {
        // Note: Sign depends on which input we differentiate w.r.t.
        // ∂k/∂x_i = -diff/l² × k (for x_i)
        // ∂k/∂x'_i = +diff/l² × k (for x'_i)
        return wrtFirst ? -diff / l2 * k : diff / l2 * k;
    }

    /// <summary>
    /// RBF second derivative: ∂²k/∂x_i∂x'_j = (δ_ij/l² - diff_i × diff_j/l⁴) × k(x, x')
    /// </summary>
    private static double ComputeRBFSecondDerivative(
        double k, double diff_i, double diff_j, int dim1, int dim2, double l2, double l4)
    {
        double delta = (dim1 == dim2) ? 1.0 : 0.0;
        return (delta / l2 - diff_i * diff_j / l4) * k;
    }

    /// <summary>
    /// Matern 5/2 first derivative.
    /// </summary>
    private double ComputeMatern52FirstDerivative(Vector<T> x1, Vector<T> x2, int dim, bool wrtFirst)
    {
        double r = ComputeDistance(x1, x2);
        double sqrt5 = Math.Sqrt(5);
        double l = _lengthScale;
        double l2 = l * l;
        double diff = _numOps.ToDouble(x1[dim]) - _numOps.ToDouble(x2[dim]);

        if (r < 1e-10)
        {
            return 0;
        }

        // For Matern 5/2: k = (1 + √5r/l + 5r²/(3l²)) × exp(-√5r/l)
        double z = sqrt5 * r / l;
        double expTerm = Math.Exp(-z);
        double polyTerm = 1 + z + z * z / 3;

        // dk/dr = -5r/(3l²) × (1 + √5r/l) × exp(-√5r/l)
        // ∂k/∂x_i = dk/dr × ∂r/∂x_i = dk/dr × (x_i - x'_i)/r
        double dkdr = -5 * r / (3 * l2) * (1 + z) * expTerm;
        double drdi = diff / r;

        return wrtFirst ? dkdr * drdi : -dkdr * drdi;
    }

    /// <summary>
    /// Matern 3/2 first derivative.
    /// </summary>
    private double ComputeMatern32FirstDerivative(Vector<T> x1, Vector<T> x2, int dim, bool wrtFirst)
    {
        double r = ComputeDistance(x1, x2);
        double sqrt3 = Math.Sqrt(3);
        double l = _lengthScale;
        double l2 = l * l;
        double diff = _numOps.ToDouble(x1[dim]) - _numOps.ToDouble(x2[dim]);

        if (r < 1e-10)
        {
            return 0;
        }

        // For Matern 3/2: k = (1 + √3r/l) × exp(-√3r/l)
        double z = sqrt3 * r / l;
        double expTerm = Math.Exp(-z);

        // dk/dr = -3r/l² × exp(-√3r/l)
        double dkdr = -3 * r / l2 * expTerm;
        double drdi = diff / r;

        return wrtFirst ? dkdr * drdi : -dkdr * drdi;
    }

    /// <summary>
    /// Matern 5/2 second derivative.
    /// </summary>
    private double ComputeMatern52SecondDerivative(Vector<T> x1, Vector<T> x2, int dim1, int dim2)
    {
        double r = ComputeDistance(x1, x2);
        double sqrt5 = Math.Sqrt(5);
        double l = _lengthScale;
        double l2 = l * l;
        double l4 = l2 * l2;

        double diff_i = _numOps.ToDouble(x1[dim1]) - _numOps.ToDouble(x2[dim1]);
        double diff_j = _numOps.ToDouble(x1[dim2]) - _numOps.ToDouble(x2[dim2]);

        if (r < 1e-10)
        {
            return (dim1 == dim2) ? 5.0 / (3 * l2) : 0;
        }

        double z = sqrt5 * r / l;
        double expTerm = Math.Exp(-z);
        double r2 = r * r;

        // Complex formula for ∂²k/∂x_i∂x'_j
        double delta = (dim1 == dim2) ? 1.0 : 0.0;
        double term1 = -5 / (3 * l2) * (1 + z) * expTerm * delta;
        double term2 = -5 / (3 * l2) * expTerm * sqrt5 / l * diff_i * diff_j / r;
        double term3 = 5 / (3 * l4) * (1 + z) * expTerm * diff_i * diff_j;
        double term4 = 5 / (3 * l2) * (1 + z) * expTerm * diff_i * diff_j / r2;

        return -(term1 + term2 + term3 - term4);
    }

    /// <summary>
    /// Matern 3/2 second derivative.
    /// </summary>
    private double ComputeMatern32SecondDerivative(Vector<T> x1, Vector<T> x2, int dim1, int dim2)
    {
        double r = ComputeDistance(x1, x2);
        double sqrt3 = Math.Sqrt(3);
        double l = _lengthScale;
        double l2 = l * l;
        double l3 = l2 * l;

        double diff_i = _numOps.ToDouble(x1[dim1]) - _numOps.ToDouble(x2[dim1]);
        double diff_j = _numOps.ToDouble(x1[dim2]) - _numOps.ToDouble(x2[dim2]);

        if (r < 1e-10)
        {
            return (dim1 == dim2) ? 3.0 / l2 : 0;
        }

        double z = sqrt3 * r / l;
        double expTerm = Math.Exp(-z);
        double r2 = r * r;

        double delta = (dim1 == dim2) ? 1.0 : 0.0;
        double term1 = -3 / l2 * expTerm * delta;
        double term2 = 3 * sqrt3 / l3 * expTerm * diff_i * diff_j / r;
        double term3 = 3 / l2 * expTerm * diff_i * diff_j / r2;

        return -(term1 + term2 - term3);
    }

    /// <summary>
    /// Computes Euclidean distance between two vectors.
    /// </summary>
    private double ComputeDistance(Vector<T> x1, Vector<T> x2)
    {
        double sum = 0;
        int d = Math.Min(x1.Length, x2.Length);
        for (int i = 0; i < d; i++)
        {
            double diff = _numOps.ToDouble(x1[i]) - _numOps.ToDouble(x2[i]);
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    /// <summary>
    /// Creates an extended vector for a value observation.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>Extended vector with gradient dimension flag = -1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to create the extended format for function value observations.
    /// </para>
    /// </remarks>
    public Vector<T> CreateValueObservation(Vector<T> input)
    {
        var extended = new Vector<T>(input.Length + 1);
        for (int i = 0; i < input.Length; i++)
        {
            extended[i] = input[i];
        }
        extended[input.Length] = _numOps.FromDouble(-1);
        return extended;
    }

    /// <summary>
    /// Creates an extended vector for a gradient observation.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <param name="gradientDim">The dimension of the gradient.</param>
    /// <returns>Extended vector with gradient dimension flag.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to create the extended format for gradient observations.
    /// gradientDim specifies which partial derivative this observation represents.
    /// </para>
    /// </remarks>
    public Vector<T> CreateGradientObservation(Vector<T> input, int gradientDim)
    {
        if (gradientDim < 0 || gradientDim >= _inputDim)
            throw new ArgumentOutOfRangeException(nameof(gradientDim));

        var extended = new Vector<T>(input.Length + 1);
        for (int i = 0; i < input.Length; i++)
        {
            extended[i] = input[i];
        }
        extended[input.Length] = _numOps.FromDouble(gradientDim);
        return extended;
    }
}
