namespace AiDotNet.Kernels;

using AiDotNet.Tensors.Helpers;

/// <summary>
/// Random Fourier Features (RFF) kernel for scalable approximation of shift-invariant kernels.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Random Fourier Features is a clever technique for making Gaussian Processes
/// scale to large datasets. Instead of computing the full N×N kernel matrix (which has O(N²) memory
/// and O(N³) computation), RFF approximates the kernel with D random features.
///
/// The key insight (Bochner's theorem): Any shift-invariant kernel can be written as the
/// expectation of random cosine features:
///
/// k(x, x') ≈ (1/D) Σᵢ cos(ωᵢ·x + bᵢ) × cos(ωᵢ·x' + bᵢ)
///          = φ(x)ᵀ × φ(x')
///
/// Where:
/// - ωᵢ are random frequencies drawn from the kernel's spectral density
/// - bᵢ are random phases drawn uniformly from [0, 2π]
/// - φ(x) is a D-dimensional feature map
///
/// This transforms the GP kernel computation into a linear model:
/// - Instead of O(N³) for exact GP, we get O(ND² + D³)
/// - Memory goes from O(N²) to O(ND)
///
/// For D ≈ 1000-10000, the approximation is usually very good.
/// </para>
/// <para>
/// Applications:
/// - Large-scale GP regression (N > 10,000 points)
/// - Online/streaming GP updates
/// - Deep kernel learning (combining with neural networks)
/// </para>
/// </remarks>
public class RFFKernel<T> : IKernelFunction<T>
{
    /// <summary>
    /// The random frequency vectors (ω).
    /// </summary>
    private readonly double[,] _frequencies;

    /// <summary>
    /// The random phase shifts (b).
    /// </summary>
    private readonly double[] _phases;

    /// <summary>
    /// Number of random features.
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// Input dimensionality.
    /// </summary>
    private readonly int _inputDim;

    /// <summary>
    /// The kernel type being approximated.
    /// </summary>
    private readonly RFFKernelType _kernelType;

    /// <summary>
    /// Length scale parameter.
    /// </summary>
    private readonly double _lengthScale;

    /// <summary>
    /// Output scale parameter.
    /// </summary>
    private readonly double _outputScale;

    /// <summary>
    /// Operations for performing numeric calculations with type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Types of kernels that can be approximated with RFF.
    /// </summary>
    public enum RFFKernelType
    {
        /// <summary>RBF (Gaussian/Squared Exponential) kernel.</summary>
        RBF,
        /// <summary>Laplacian (Exponential) kernel.</summary>
        Laplacian,
        /// <summary>Matern 1/2 kernel (same as Laplacian).</summary>
        Matern12,
        /// <summary>Matern 3/2 kernel.</summary>
        Matern32,
        /// <summary>Matern 5/2 kernel.</summary>
        Matern52
    }

    /// <summary>
    /// Initializes a new RFF kernel.
    /// </summary>
    /// <param name="inputDim">The dimensionality of input vectors.</param>
    /// <param name="numFeatures">Number of random features (D). More = better approximation but slower.</param>
    /// <param name="kernelType">The type of kernel to approximate.</param>
    /// <param name="lengthScale">The length scale parameter.</param>
    /// <param name="outputScale">The output scale (variance).</param>
    /// <param name="seed">Random seed for reproducibility. If null, uses secure random.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creates an RFF kernel that approximates the specified kernel type.
    ///
    /// Choosing numFeatures:
    /// - 100-500: Fast but rough approximation
    /// - 500-2000: Good balance of speed and accuracy
    /// - 2000-10000: High accuracy, still faster than exact for large N
    ///
    /// Rule of thumb: For approximation error ε, you need roughly D = O(1/ε²) features.
    ///
    /// Example:
    /// var rff = new RFFKernel&lt;double&gt;(
    ///     inputDim: 10,
    ///     numFeatures: 1000,
    ///     kernelType: RFFKernelType.RBF,
    ///     lengthScale: 1.0
    /// );
    /// </para>
    /// </remarks>
    public RFFKernel(
        int inputDim,
        int numFeatures,
        RFFKernelType kernelType = RFFKernelType.RBF,
        double lengthScale = 1.0,
        double outputScale = 1.0,
        int? seed = null)
    {
        if (inputDim < 1)
            throw new ArgumentException("Input dimension must be at least 1.", nameof(inputDim));
        if (numFeatures < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(numFeatures));
        if (lengthScale <= 0)
            throw new ArgumentException("Length scale must be positive.", nameof(lengthScale));
        if (outputScale <= 0)
            throw new ArgumentException("Output scale must be positive.", nameof(outputScale));

        _inputDim = inputDim;
        _numFeatures = numFeatures;
        _kernelType = kernelType;
        _lengthScale = lengthScale;
        _outputScale = outputScale;
        _numOps = MathHelper.GetNumericOperations<T>();

        // Initialize random frequencies and phases
        _frequencies = new double[numFeatures, inputDim];
        _phases = new double[numFeatures];

        var rand = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        SampleFrequencies(rand);
        SamplePhases(rand);
    }

    /// <summary>
    /// Samples random frequencies from the spectral density of the kernel.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Different kernels have different spectral densities (Fourier transforms).
    /// We sample frequencies according to these densities:
    ///
    /// - RBF: Gaussian with variance 1/l²
    /// - Laplacian: Cauchy distribution
    /// - Matern: t-distribution with degrees of freedom based on ν
    /// </para>
    /// </remarks>
    private void SampleFrequencies(Random rand)
    {
        double scale = 1.0 / _lengthScale;

        for (int i = 0; i < _numFeatures; i++)
        {
            for (int d = 0; d < _inputDim; d++)
            {
                double freq = _kernelType switch
                {
                    RFFKernelType.RBF => SampleGaussian(rand) * scale,
                    RFFKernelType.Laplacian or RFFKernelType.Matern12 => SampleCauchy(rand) * scale,
                    RFFKernelType.Matern32 => SampleStudentT(rand, 3) * scale,
                    RFFKernelType.Matern52 => SampleStudentT(rand, 5) * scale,
                    _ => SampleGaussian(rand) * scale
                };
                _frequencies[i, d] = freq;
            }
        }
    }

    /// <summary>
    /// Samples random phases uniformly from [0, 2π].
    /// </summary>
    private void SamplePhases(Random rand)
    {
        for (int i = 0; i < _numFeatures; i++)
        {
            _phases[i] = rand.NextDouble() * 2 * Math.PI;
        }
    }

    /// <summary>
    /// Samples from a standard Gaussian distribution using Box-Muller transform.
    /// </summary>
    private static double SampleGaussian(Random rand)
    {
        double u1 = rand.NextDouble();
        double u2 = rand.NextDouble();
        // Avoid log(0)
        u1 = Math.Max(u1, 1e-10);
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }

    /// <summary>
    /// Samples from a Cauchy distribution.
    /// </summary>
    private static double SampleCauchy(Random rand)
    {
        double u = rand.NextDouble();
        // Avoid tan(π/2)
        u = Math.Max(0.001, Math.Min(0.999, u));
        return Math.Tan(Math.PI * (u - 0.5));
    }

    /// <summary>
    /// Samples from a Student-t distribution with given degrees of freedom.
    /// </summary>
    private static double SampleStudentT(Random rand, double df)
    {
        // Use ratio of Gaussian to chi-squared
        double z = SampleGaussian(rand);
        double chi2 = 0;
        for (int i = 0; i < (int)df; i++)
        {
            double g = SampleGaussian(rand);
            chi2 += g * g;
        }
        return z / Math.Sqrt(chi2 / df);
    }

    /// <summary>
    /// Gets the number of random features.
    /// </summary>
    public int NumFeatures => _numFeatures;

    /// <summary>
    /// Gets the input dimensionality.
    /// </summary>
    public int InputDim => _inputDim;

    /// <summary>
    /// Gets the kernel type being approximated.
    /// </summary>
    public RFFKernelType KernelType => _kernelType;

    /// <summary>
    /// Gets the length scale.
    /// </summary>
    public double LengthScale => _lengthScale;

    /// <summary>
    /// Gets the output scale.
    /// </summary>
    public double OutputScale => _outputScale;

    /// <summary>
    /// Computes the random Fourier feature map for an input vector.
    /// </summary>
    /// <param name="x">The input vector.</param>
    /// <returns>The D-dimensional feature vector.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the key transformation that makes RFF work.
    /// It maps your input x to a D-dimensional feature space where the
    /// dot product approximates the kernel: φ(x)ᵀφ(x') ≈ k(x, x').
    ///
    /// The feature map is: φᵢ(x) = √(2σ²/D) × cos(ωᵢ·x + bᵢ)
    ///
    /// This is useful for:
    /// - Linear models on the features (equivalent to kernel methods)
    /// - Neural network layers (kernel approximation via random features)
    /// </para>
    /// </remarks>
    public double[] GetFeatures(Vector<T> x)
    {
        if (x.Length != _inputDim)
            throw new ArgumentException($"Expected input dimension {_inputDim}, got {x.Length}.");

        var features = new double[_numFeatures];
        double scale = Math.Sqrt(2.0 * _outputScale / _numFeatures);

        for (int i = 0; i < _numFeatures; i++)
        {
            // Compute ωᵢ·x
            double dot = 0;
            for (int d = 0; d < _inputDim; d++)
            {
                dot += _frequencies[i, d] * _numOps.ToDouble(x[d]);
            }

            // Compute √(2σ²/D) × cos(ωᵢ·x + bᵢ)
            features[i] = scale * Math.Cos(dot + _phases[i]);
        }

        return features;
    }

    /// <summary>
    /// Calculates the approximate kernel value between two vectors.
    /// </summary>
    /// <param name="x1">The first vector.</param>
    /// <param name="x2">The second vector.</param>
    /// <returns>The approximate kernel value: φ(x1)ᵀφ(x2).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Computes the kernel approximation as the dot product
    /// of the random feature maps.
    ///
    /// This is faster than exact kernel computation when you need to:
    /// - Compute many kernel values
    /// - Solve kernel systems (use features with linear solver instead)
    /// </para>
    /// </remarks>
    public T Calculate(Vector<T> x1, Vector<T> x2)
    {
        double[] f1 = GetFeatures(x1);
        double[] f2 = GetFeatures(x2);

        double result = 0;
        for (int i = 0; i < _numFeatures; i++)
        {
            result += f1[i] * f2[i];
        }

        return _numOps.FromDouble(result);
    }

    /// <summary>
    /// Computes the feature matrix for multiple input points.
    /// </summary>
    /// <param name="X">Input matrix where each row is a data point.</param>
    /// <returns>Feature matrix of shape (N, D).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main entry point for scalable GP regression.
    /// Instead of computing the N×N kernel matrix, compute the N×D feature matrix
    /// and use linear regression techniques.
    ///
    /// For GP regression with RFF:
    /// 1. Compute Φ = GetFeatureMatrix(X_train)
    /// 2. Solve (ΦᵀΦ + σ²I)w = Φᵀy using linear algebra (O(D³) instead of O(N³))
    /// 3. Predictions: mean = Φ_test × w
    /// </para>
    /// </remarks>
    public double[,] GetFeatureMatrix(Matrix<T> X)
    {
        int n = X.Rows;
        var features = new double[n, _numFeatures];
        double scale = Math.Sqrt(2.0 * _outputScale / _numFeatures);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < _numFeatures; j++)
            {
                // Compute ωⱼ·xᵢ
                double dot = 0;
                for (int d = 0; d < _inputDim; d++)
                {
                    dot += _frequencies[j, d] * _numOps.ToDouble(X[i, d]);
                }

                features[i, j] = scale * Math.Cos(dot + _phases[j]);
            }
        }

        return features;
    }

    /// <summary>
    /// Estimates the approximation quality by comparing to exact kernel on test points.
    /// </summary>
    /// <param name="X">Test points to evaluate approximation quality.</param>
    /// <param name="exactKernel">The exact kernel for comparison.</param>
    /// <returns>Mean absolute error between approximate and exact kernel values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this to verify that your RFF approximation is accurate enough.
    /// Lower error means better approximation.
    ///
    /// If the error is too high, increase numFeatures.
    /// </para>
    /// </remarks>
    public double EstimateApproximationError(Matrix<T> X, IKernelFunction<T> exactKernel)
    {
        int n = X.Rows;
        int numPairs = Math.Min(n * (n + 1) / 2, 1000); // Sample at most 1000 pairs

        double totalError = 0;
        int count = 0;
        var rand = RandomHelper.CreateSecureRandom();

        for (int trial = 0; trial < numPairs && count < numPairs; trial++)
        {
            int i = rand.Next(n);
            int j = rand.Next(n);

            var xi = GetRow(X, i);
            var xj = GetRow(X, j);

            double exact = _numOps.ToDouble(exactKernel.Calculate(xi, xj));
            double approx = _numOps.ToDouble(Calculate(xi, xj));

            totalError += Math.Abs(exact - approx);
            count++;
        }

        return count > 0 ? totalError / count : 0;
    }

    /// <summary>
    /// Extracts a row from a matrix as a vector.
    /// </summary>
    private Vector<T> GetRow(Matrix<T> matrix, int row)
    {
        var result = new Vector<T>(matrix.Columns);
        for (int j = 0; j < matrix.Columns; j++)
        {
            result[j] = matrix[row, j];
        }
        return result;
    }
}
