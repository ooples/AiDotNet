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
    private readonly Tensor<T> _frequencies;

    /// <summary>
    /// The random phase shifts (b).
    /// </summary>
    private readonly Tensor<T> _phases;
    // Frequencies and phase geometry retain the original double precision for narrow storage types.
    private readonly RFFKernel<double>? _wideKernel;


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
        /// <summary>Matern 1/2 kernel: exp(-||x - x'||₂ / l), the L2 exponential (Laplacian is the L1 form).</summary>
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

        if (typeof(T) != typeof(double))
        {
            _wideKernel = new RFFKernel<double>(inputDim, numFeatures,
                (RFFKernel<double>.RFFKernelType)(int)kernelType, lengthScale, outputScale, seed);
            _frequencies = new Tensor<T>(new[] { 0, 0 });
            _phases = new Tensor<T>(new[] { 0 });
            return;
        }

        var rand = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize random frequencies and phases from one seeded engine draw
        (_frequencies, _phases) = SampleFrequenciesAndPhases(rand.Next());
    }

    /// <summary>
    /// Samples the frequencies from the kernel's spectral density, scaled by 1/lengthScale, and the
    /// phases uniformly from [0, 2π), all from one seeded uniform tensor.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Spectral densities: RBF → Gaussian (Box-Muller); Laplacian (the L1 kernel) → independent Cauchy
    /// per coordinate (inverse CDF); Matérn-ν (L2) → multivariate Student-t with 2ν degrees of freedom:
    /// Matern12 → df 1, Matern32 → df 3, Matern52 → df 5. A multivariate t shares one chi-squared value
    /// across all coordinates of a frequency row (a Gaussian vector over the root of chi-squared / df).
    /// </para>
    /// </remarks>
    private (Tensor<T> Frequencies, Tensor<T> Phases) SampleFrequenciesAndPhases(int seed)
    {
        var engine = AiDotNetEngine.Current;
        var shape = new[] { _numFeatures, _inputDim };
        int perFrequency = _kernelType switch
        {
            RFFKernelType.Laplacian => 1,
            RFFKernelType.Matern12 => 2 * (1 + 1),
            RFFKernelType.Matern32 => 2 * (1 + 3),
            RFFKernelType.Matern52 => 2 * (1 + 5),
            _ => 2
        };

        // Uniform draws: perFrequency planes of [numFeatures, inputDim], then one plane for the phases
        var uniforms = engine.TensorRandomUniformRange(new[] { perFrequency * _numFeatures * _inputDim + _numFeatures },
            _numOps.Zero, _numOps.One, seed);
        int planeSize = _numFeatures * _inputDim;
        Tensor<T> Plane(int index) => engine.TensorSlice(uniforms, new[] { index * planeSize }, new[] { planeSize }).Reshape(shape);

        Tensor<T> frequencies = _kernelType switch
        {
            RFFKernelType.Laplacian => SampleCauchy(Plane(0)),
            RFFKernelType.Matern12 => SampleStudentT(Plane, 1),
            RFFKernelType.Matern32 => SampleStudentT(Plane, 3),
            RFFKernelType.Matern52 => SampleStudentT(Plane, 5),
            _ => SampleGaussian(Plane(0), Plane(1))
        };
        frequencies = engine.TensorMultiplyScalar(frequencies, _numOps.FromDouble(1.0 / _lengthScale));

        var phases = engine.TensorMultiplyScalar(
            engine.TensorSlice(uniforms, new[] { perFrequency * planeSize }, new[] { _numFeatures }),
            _numOps.FromDouble(2 * Math.PI));

        return (frequencies, phases);
    }

    /// <summary>
    /// Box-Muller: sqrt(-2 log(max(u1, 1e-10))) cos(2π u2).
    /// </summary>
    private Tensor<T> SampleGaussian(Tensor<T> u1, Tensor<T> u2)
    {
        var engine = AiDotNetEngine.Current;

        // Avoid log(0)
        var radius = engine.TensorSqrt(engine.TensorMultiplyScalar(
            engine.TensorLog(engine.TensorClampMin(u1, _numOps.FromDouble(1e-10))), _numOps.FromDouble(-2.0)));
        return engine.TensorMultiply(radius,
            engine.TensorCos(engine.TensorMultiplyScalar(u2, _numOps.FromDouble(2 * Math.PI))));
    }

    /// <summary>
    /// Inverse-CDF Cauchy: tan(π (u - 0.5)) with u clamped to [0.001, 0.999].
    /// </summary>
    private Tensor<T> SampleCauchy(Tensor<T> u)
    {
        var engine = AiDotNetEngine.Current;

        // Avoid tan(π/2)
        var angle = engine.TensorMultiplyScalar(
            engine.TensorAddScalar(engine.TensorClamp(u, _numOps.FromDouble(0.001), _numOps.FromDouble(0.999)),
                _numOps.FromDouble(-0.5)),
            _numOps.FromDouble(Math.PI));
        return engine.TensorDivide(engine.TensorSin(angle), engine.TensorCos(angle));
    }

    /// <summary>
    /// Multivariate Student-t with df degrees of freedom per frequency row: z / sqrt(chi2 / df), with z a
    /// Gaussian per coordinate and chi2 (the sum of df squared Gaussians) drawn once per row.
    /// </summary>
    private Tensor<T> SampleStudentT(Func<int, Tensor<T>> plane, int df)
    {
        var engine = AiDotNetEngine.Current;
        var z = SampleGaussian(plane(0), plane(1));
        int rows = z.Shape[0];
        int columns = z.Shape[1];
        var squares = Enumerable.Range(1, df).Select(k =>
        {
            // One Gaussian per row: the first coordinate's draw of this plane pair.
            var g = engine.TensorSlice(SampleGaussian(plane(2 * k), plane(2 * k + 1)), new[] { 0, 0 }, new[] { rows, 1 });
            return engine.TensorMultiply(g, g);
        });
        var chi2 = engine.TensorBroadcastTo(squares.Aggregate((sum, square) => engine.TensorAdd(sum, square)),
            new[] { rows, columns });

        return engine.TensorDivide(z, engine.TensorSqrt(engine.TensorMultiplyScalar(chi2, _numOps.FromDouble(1.0 / df))));
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
    public Vector<T> GetFeatures(Vector<T> x)
    {
        if (x.Length != _inputDim)
            throw new ArgumentException($"Expected input dimension {_inputDim}, got {x.Length}.");

        // √(2σ²/D) × cos(ωᵢ·x + bᵢ)
        return FeatureMap(x.AsTensor().Reshape(new[] { 1, _inputDim })).ToVector();
    }

    /// <summary>
    /// Computes √(2σ²/D) × cos(X Ωᵀ + b) for a [n, inputDim] batch.
    /// </summary>
    private Tensor<T> FeatureMap(Tensor<T> inputs)
    {
        if (_wideKernel is not null) return _wideKernel.FeatureMap(inputs.Cast<double>()).Cast<T>();
        var engine = AiDotNetEngine.Current;
        int n = inputs.Shape[0];
        var projections = engine.TensorMatMul(inputs, engine.TensorTranspose(_frequencies));
        var shifted = engine.TensorAdd(projections,
            engine.TensorBroadcastTo(_phases.Reshape(new[] { 1, _numFeatures }), new[] { n, _numFeatures }));

        return engine.TensorMultiplyScalar(engine.TensorCos(shifted),
            _numOps.FromDouble(Math.Sqrt(2.0 * _outputScale / _numFeatures)));
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
        if (_wideKernel is not null)
            return _numOps.FromDouble(_wideKernel.Calculate(
                x1.AsTensor().Cast<double>().ToVector(), x2.AsTensor().Cast<double>().ToVector()));
        return AiDotNetEngine.Current.DotProduct(GetFeatures(x1), GetFeatures(x2));
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
    public Matrix<T> GetFeatureMatrix(Matrix<T> X)
    {
        if (X is null)
            throw new ArgumentNullException(nameof(X));
        if (X.Columns != _inputDim)
            throw new ArgumentException($"Expected input dimension {_inputDim}, got {X.Columns}.", nameof(X));
        if (X.Rows == 0)
            return new Matrix<T>(0, _numFeatures);

        return FeatureMap(Tensor<T>.FromMatrix(X)).ToMatrix();
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
    public T EstimateApproximationError(Matrix<T> X, IKernelFunction<T> exactKernel)
    {
        int n = X.Rows;
        // Sample at most 1000 pairs; the pair count is formed in long so large n cannot overflow int.
        int numPairs = (int)Math.Min((long)n * (n + 1) / 2, 1000L);

        // Draw every pair first, in the same order as before (i then j per trial)
        var rand = RandomHelper.CreateSecureRandom();
        var pairs = new (int I, int J)[numPairs];
        for (int trial = 0; trial < numPairs; trial++)
        {
            int i = rand.Next(n);
            int j = rand.Next(n);
            pairs[trial] = (i, j);
        }

        if (numPairs == 0)
            return _numOps.Zero;

        var errors = new Vector<T>(pairs.Select(pair => PairError(X.GetRow(pair.I), X.GetRow(pair.J), exactKernel)));
        return _numOps.FromDouble(AiDotNetEngine.Current.TensorMean(errors.AsTensor().Cast<double>()));
    }

    /// <summary>
    /// |k_exact(xi, xj) - k_rff(xi, xj)|.
    /// </summary>
    private T PairError(Vector<T> xi, Vector<T> xj, IKernelFunction<T> exactKernel)
    {
        return _numOps.Abs(_numOps.Subtract(exactKernel.Calculate(xi, xj), Calculate(xi, xj)));
    }
}
