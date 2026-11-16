using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Fast Independent Component Analysis (FastICA) for blind source separation.
/// </summary>
/// <remarks>
/// <para>
/// FastICA separates a multivariate signal into additive, independent non-Gaussian
/// components. It is commonly used for blind source separation (e.g., separating
/// mixed audio signals) and feature extraction.
/// </para>
/// <para>
/// The algorithm:
/// 1. Centers and whitens the data
/// 2. Uses fixed-point iteration to find independent components
/// 3. Each component is found by maximizing non-Gaussianity
/// </para>
/// <para><b>For Beginners:</b> Imagine you have recordings from multiple microphones
/// in a room with multiple speakers. ICA can separate the individual speakers:
/// - It finds components that are statistically independent
/// - Unlike PCA which finds uncorrelated components, ICA finds truly independent ones
/// - Works best when source signals are non-Gaussian (most real-world signals are)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FastICA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly ICAAlgorithm _algorithm;
    private readonly ICAFunction _fun;
    private readonly int _maxIterations;
    private readonly double _tolerance;
    private const double TinyEigenvalue = 1e-30; // eigenvalue floor (sklearn clips at finfo.tiny), representable as float
    private readonly int _randomState;
    private readonly bool _whiten;

    // Fitted parameters
    private Vector<T>? _mean;
    private Matrix<T>? _whitening;
    private Matrix<T>? _components; // Unmixing matrix
    private Matrix<T>? _mixing; // Mixing matrix (pseudo-inverse of the full unmixing)
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of independent components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the algorithm type (parallel or deflation).
    /// </summary>
    public ICAAlgorithm Algorithm => _algorithm;

    /// <summary>
    /// Gets the non-linearity function used.
    /// </summary>
    public ICAFunction Fun => _fun;

    /// <summary>
    /// Gets the unmixing matrix (components).
    /// </summary>
    public Matrix<T>? Components => _components;

    /// <summary>
    /// Gets the mixing matrix.
    /// </summary>
    public Matrix<T>? Mixing => _mixing;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="FastICA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Number of components to extract. Defaults to 2.</param>
    /// <param name="algorithm">Algorithm: parallel or deflation. Defaults to Parallel.</param>
    /// <param name="fun">Non-linearity function: logcosh, exp, or cube. Defaults to LogCosh.</param>
    /// <param name="maxIterations">Maximum iterations. Defaults to 200.</param>
    /// <param name="tolerance">Convergence tolerance. Defaults to 1e-4.</param>
    /// <param name="whiten">Whether to whiten data before ICA. Defaults to true.</param>
    /// <param name="randomState">Random seed. Defaults to 0.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public FastICA(
        int nComponents = 2,
        ICAAlgorithm algorithm = ICAAlgorithm.Parallel,
        ICAFunction fun = ICAFunction.LogCosh,
        int maxIterations = 200,
        double tolerance = 1e-4,
        bool whiten = true,
        int randomState = 0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        _nComponents = nComponents;
        _algorithm = algorithm;
        _fun = fun;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _whiten = whiten;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits FastICA to extract independent components.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        if (n == 0)
        {
            throw new ArgumentException("Input data must have at least one sample.", nameof(data));
        }

        if (_whiten && n < p)
        {
            throw new ArgumentException(
                $"When whitening is enabled, number of samples ({n}) must be >= number of features ({p}).",
                nameof(data));
        }

        int k = Math.Min(_nComponents, Math.Min(n, p));

        // Center the data
        var x = Tensor<T>.FromMatrix(data);
        var mean = DimensionalityReductionOps.ColumnMeans(x);
        var centered = DimensionalityReductionOps.SubtractRow(x, mean);
        _mean = mean.ToVector();

        // Whiten the data
        Tensor<T> x1;
        Tensor<T>? whitening = null;
        if (_whiten)
        {
            (x1, whitening) = WhitenData(centered);
        }
        else
        {
            x1 = centered;
        }

        // Run FastICA
        var unmixing = _algorithm == ICAAlgorithm.Deflation
            ? FastICADeflation(x1, k)
            : FastICAParallel(x1, k);

        _components = unmixing.ToMatrix();
        _whitening = whitening?.ToMatrix();

        // Compute mixing matrix
        ComputeMixingMatrix(unmixing, whitening);
    }

    /// <summary>
    /// Whitening by the eigen-decomposition of the (1/n) covariance: rows of the whitening matrix are the
    /// eigenvectors scaled by 1/sqrt(eigenvalue) (0 for eigenvalues at most 1e-10), largest first.
    /// </summary>
    private (Tensor<T> Whitened, Tensor<T> Whitening) WhitenData(Tensor<T> x)
    {
        var engine = Engine;
        int n = x.Shape[0];

        // Compute covariance matrix
        var cov = engine.TensorDivideScalar(DimensionalityReductionOps.TransposeMatMul(x, x), NumOps.FromDouble(n));

        // Eigendecomposition
        var (eigenvalues, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(cov, descending: true);

        // Whitening matrix: D^(-1/2) * E^T
        var scale = engine.TensorWhere(
            engine.TensorGreaterThan(eigenvalues, DimensionalityReductionOps.Tolerance<T>(1e-10)),
            engine.TensorRsqrt(eigenvalues),
            engine.TensorZerosLike(eigenvalues));
        var whitening = engine.TensorTranspose(DimensionalityReductionOps.ScaleColumns(eigenvectors, scale));

        // Apply whitening
        return (DimensionalityReductionOps.MatMulTranspose(x, whitening), whitening);
    }

    private Tensor<T> FastICADeflation(Tensor<T> x, int nComponents)
    {
        var engine = Engine;
        int n = x.Shape[0];
        int p = x.Shape[1];
        var random = RandomHelper.CreateSeededRandom(_randomState);
        T half = NumOps.FromDouble(0.5);
        T sampleCount = NumOps.FromDouble(n);
        T tolerance = DimensionalityReductionOps.Tolerance<T>(_tolerance);
        T floor = DimensionalityReductionOps.Tolerance<T>(1e-10);
        var rows = new Tensor<T>[nComponents];

        for (int c = 0; c < nComponents; c++)
        {
            // Random initialization
            var w = UnitRow(engine.TensorSubtractScalar(DimensionalityReductionOps.Uniform<T>(random, 1, p), half));

            for (int iter = 0; iter < _maxIterations; iter++)
            {
                var wOld = w;

                // Fixed-point update: w = E[x g(w^T x)] - E[g'(w^T x)] w
                var wx = DimensionalityReductionOps.MatMulTranspose(x, w);                        // [n, 1]
                var (g, gPrime) = ApplyNonLinearity(wx);
                var expectedXg = engine.TensorDivideScalar(DimensionalityReductionOps.TransposeMatMul(g, x), sampleCount);
                w = engine.TensorSubtract(expectedXg, engine.TensorMultiplyScalar(wOld, engine.TensorMean(gPrime)));

                // Decorrelate from previous components (Gram-Schmidt)
                for (int prevC = 0; prevC < c; prevC++)
                {
                    var previous = rows[prevC];
                    w = engine.TensorSubtract(w, engine.TensorMatMul(DimensionalityReductionOps.MatMulTranspose(w, previous), previous));
                }

                // Normalize; a vanished direction restarts from a fresh random vector
                T norm = DimensionalityReductionOps.Norm(w);
                if (DimensionalityReductionOps.Below(norm, floor))
                {
                    w = UnitRow(engine.TensorSubtractScalar(DimensionalityReductionOps.Uniform<T>(random, 1, p), half));
                    break;
                }

                w = engine.TensorDivideScalar(w, norm);

                // Check convergence: sum | |w| - |w_old| |
                T change = engine.TensorSum(engine.TensorAbs(engine.TensorSubtract(engine.TensorAbs(w), engine.TensorAbs(wOld))));
                if (DimensionalityReductionOps.Below(change, tolerance))
                {
                    break;
                }
            }

            rows[c] = w;
        }

        return engine.TensorConcatenate(rows, 0);
    }

    private Tensor<T> FastICAParallel(Tensor<T> x, int nComponents)
    {
        var engine = Engine;
        int n = x.Shape[0];
        int p = x.Shape[1];
        var random = RandomHelper.CreateSeededRandom(_randomState);
        T sampleCount = NumOps.FromDouble(n);
        T tolerance = DimensionalityReductionOps.Tolerance<T>(_tolerance);
        var ones = DimensionalityReductionOps.Full(new[] { nComponents }, NumOps.One);

        // Random initialization, then symmetric decorrelation (sklearn _ica_par)
        var w = SymmetricDecorrelation(
            engine.TensorSubtractScalar(DimensionalityReductionOps.Uniform<T>(random, nComponents, p), NumOps.FromDouble(0.5)));

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var wOld = w;

            // Fixed-point update for every component at once: W = E[g(W x) x^T] - diag(E[g'(W x)]) W
            var wx = DimensionalityReductionOps.MatMulTranspose(x, wOld);                          // [n, k]
            var (g, gPrime) = ApplyNonLinearity(wx);
            var expectedXg = engine.TensorDivideScalar(DimensionalityReductionOps.TransposeMatMul(g, x), sampleCount);
            var expectedGPrime = DimensionalityReductionOps.ColumnMeans(gPrime);                   // [k]
            w = SymmetricDecorrelation(engine.TensorSubtract(expectedXg, DimensionalityReductionOps.ScaleRows(wOld, expectedGPrime)));

            // Convergence (sklearn): max | |diag(W W_old^T)| - 1 |
            var alignment = engine.TensorDiagonal(DimensionalityReductionOps.MatMulTranspose(w, wOld));
            T change = DimensionalityReductionOps.MaxAbsDifference(engine.TensorAbs(alignment), ones);
            if (DimensionalityReductionOps.Below(change, tolerance))
            {
                break;
            }
        }

        return w;
    }

    /// <summary>Divides a row by its Euclidean norm.</summary>
    private Tensor<T> UnitRow(Tensor<T> row) => Engine.TensorDivideScalar(row, DimensionalityReductionOps.Norm(row));

    /// <summary>Symmetric decorrelation W &lt;- (W W^T)^(-1/2) W (Hyvarinen 1999; sklearn FastICA).</summary>
    private Tensor<T> SymmetricDecorrelation(Tensor<T> w)
    {
        // S = W W^T = U diag(s) U^T; W <- U diag(1 / sqrt(s)) U^T W, s clipped below at a tiny positive floor
        // (sklearn _sym_decorrelation)
        var engine = Engine;
        var (values, vectors) = DimensionalityReductionOps.SymmetricEigen(
            DimensionalityReductionOps.MatMulTranspose(w, w), descending: true);
        var scale = engine.TensorRsqrt(engine.TensorClampMin(values, NumOps.FromDouble(TinyEigenvalue)));
        var inverseRoot = DimensionalityReductionOps.MatMulTranspose(
            DimensionalityReductionOps.ScaleColumns(vectors, scale), vectors);
        return engine.TensorMatMul(inverseRoot, w);
    }

    /// <summary>The contrast function g and its derivative g' applied to every projection.</summary>
    private (Tensor<T> G, Tensor<T> GPrime) ApplyNonLinearity(Tensor<T> u)
    {
        var engine = Engine;
        switch (_fun)
        {
            case ICAFunction.Exp:
            {
                // G(u) = -exp(-u^2/2), g(u) = u exp(-u^2/2), g'(u) = (1 - u^2) exp(-u^2/2)
                var uSquared = engine.TensorSquare(u);
                var expU2 = engine.TensorExp(engine.TensorMultiplyScalar(uSquared, NumOps.FromDouble(-0.5)));
                return (engine.TensorMultiply(u, expU2),
                    engine.TensorMultiply(engine.ScalarMinusTensor(NumOps.One, uSquared), expU2));
            }

            case ICAFunction.Cube:
            {
                // G(u) = u^4/4, g(u) = u^3, g'(u) = 3u^2
                var uSquared = engine.TensorSquare(u);
                return (engine.TensorMultiply(uSquared, u), engine.TensorMultiplyScalar(uSquared, NumOps.FromDouble(3.0)));
            }

            default:
            {
                // LogCosh: g(u) = tanh(u), g'(u) = 1 - tanh^2(u)
                var tanhU = engine.TensorTanh(u);
                return (tanhU, engine.ScalarMinusTensor(NumOps.One, engine.TensorSquare(tanhU)));
            }
        }
    }

    /// <summary>
    /// Mixing matrix A = pinv(W K) (K the whitening matrix, identity without whitening), the matrix that maps
    /// sources back to centered data, as scikit-learn's FastICA.mixing_.
    /// </summary>
    private void ComputeMixingMatrix(Tensor<T> unmixing, Tensor<T>? whitening)
    {
        var fullUnmixing = whitening is null ? unmixing : Engine.TensorMatMul(unmixing, whitening);
        _mixing = DimensionalityReductionOps.PseudoInverse(fullUnmixing).ToMatrix();
    }

    /// <summary>
    /// Transforms the data to independent components.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The independent components.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _components is null)
        {
            throw new InvalidOperationException("FastICA has not been fitted.");
        }

        if (data.Columns != _nFeaturesIn)
        {
            throw new ArgumentException(
                $"Input data has {data.Columns} features, but FastICA was fitted with {_nFeaturesIn} features.",
                nameof(data));
        }

        // Center and whiten
        var x = DimensionalityReductionOps.SubtractRow(Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(_mean));
        if (_whitening is not null)
        {
            x = DimensionalityReductionOps.MatMulTranspose(x, Tensor<T>.FromMatrix(_whitening));
        }

        // Apply unmixing
        return DimensionalityReductionOps.MatMulTranspose(x, Tensor<T>.FromMatrix(_components)).ToMatrix();
    }

    /// <summary>
    /// Transforms independent components back to original space.
    /// </summary>
    /// <param name="data">The independent components.</param>
    /// <returns>Reconstructed data in original space.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _mixing is null)
        {
            throw new InvalidOperationException("FastICA has not been fitted.");
        }

        int k = _mixing.Columns;

        if (data.Columns != k)
        {
            throw new ArgumentException(
                $"Input data has {data.Columns} columns, but expected {k} independent components.",
                nameof(data));
        }

        // X = S A^T + mean
        var reconstructed = DimensionalityReductionOps.MatMulTranspose(Tensor<T>.FromMatrix(data), Tensor<T>.FromMatrix(_mixing));
        return DimensionalityReductionOps.AddRow(reconstructed, Tensor<T>.FromVector(_mean)).ToMatrix();
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        int k = _components?.Rows ?? _nComponents;
        var names = new string[k];
        for (int i = 0; i < k; i++)
        {
            names[i] = $"IC{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the ICA algorithm type.
/// </summary>
public enum ICAAlgorithm
{
    /// <summary>
    /// Parallel (symmetric) algorithm - all components extracted simultaneously.
    /// </summary>
    Parallel,

    /// <summary>
    /// Deflation algorithm - components extracted one at a time.
    /// </summary>
    Deflation
}

/// <summary>
/// Specifies the non-linearity function for ICA.
/// </summary>
public enum ICAFunction
{
    /// <summary>
    /// Log cosh function: g(u) = tanh(u). Good general-purpose choice.
    /// </summary>
    LogCosh,

    /// <summary>
    /// Exponential function: g(u) = u * exp(-u²/2). Good for super-Gaussian sources.
    /// </summary>
    Exp,

    /// <summary>
    /// Cubic function: g(u) = u³. Fast but less robust.
    /// </summary>
    Cube
}
