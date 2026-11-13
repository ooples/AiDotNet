using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// t-Distributed Stochastic Neighbor Embedding for visualization.
/// </summary>
/// <remarks>
/// <para>
/// t-SNE is a nonlinear dimensionality reduction technique well-suited for
/// visualizing high-dimensional data in 2D or 3D space. It preserves local
/// structure by keeping similar points close together.
/// </para>
/// <para>
/// The algorithm converts similarities between data points to joint probabilities
/// and tries to minimize the divergence between probability distributions in
/// high and low dimensional spaces.
/// </para>
/// <para><b>For Beginners:</b> t-SNE creates beautiful 2D/3D visualizations:
/// - Points that are similar stay close together
/// - Points that are different move apart
/// - Great for exploring clusters in your data
/// - Warning: Not for preserving global distances, just local neighborhoods
/// - Warning: Results can vary with different random seeds
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TSNE<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly double _perplexity;
    private readonly double _learningRate;
    private readonly int _nIter;
    private readonly double _earlyExaggeration;
    private readonly int? _randomState;
    private readonly TSNEMetric _metric;
    private readonly TSNEInitialization _initialization;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the perplexity parameter.
    /// </summary>
    public double Perplexity => _perplexity;

    /// <summary>
    /// Gets the learning rate.
    /// </summary>
    public double LearningRate => _learningRate;

    /// <summary>
    /// Gets the distance metric.
    /// </summary>
    public TSNEMetric Metric => _metric;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="TSNE{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality (usually 2 or 3). Defaults to 2.</param>
    /// <param name="perplexity">Balance between local and global structure. Defaults to 30.</param>
    /// <param name="learningRate">Learning rate for optimization. Defaults to 200.</param>
    /// <param name="nIter">Number of optimization iterations. Defaults to 1000.</param>
    /// <param name="earlyExaggeration">Exaggeration factor for early iterations. Defaults to 12.</param>
    /// <param name="metric">Distance metric to use. Defaults to Euclidean.</param>
    /// <param name="initialization">Initialization method. Defaults to Random.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public TSNE(
        int nComponents = 2,
        double perplexity = 30.0,
        double learningRate = 200.0,
        int nIter = 1000,
        double earlyExaggeration = 12.0,
        TSNEMetric metric = TSNEMetric.Euclidean,
        TSNEInitialization initialization = TSNEInitialization.Random,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (perplexity <= 0)
        {
            throw new ArgumentException("Perplexity must be positive.", nameof(perplexity));
        }

        _nComponents = nComponents;
        _perplexity = perplexity;
        _learningRate = learningRate;
        _nIter = nIter;
        _earlyExaggeration = earlyExaggeration;
        _metric = metric;
        _initialization = initialization;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits t-SNE and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nSamples = data.Rows;
        int n = data.Rows;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        var x = Tensor<T>.FromMatrix(data);

        // Compute pairwise distances
        var distances = ComputeDistances(x);

        // Compute joint probabilities P and apply early exaggeration
        var p = engine.TensorMultiplyScalar(ComputeJointProbabilities(distances, n), NumOps.FromDouble(_earlyExaggeration));

        // Initialize embedding Y
        var y = InitializeEmbedding(x, n, random);

        // Gradient descent with momentum and adaptive gains
        var gains = DimensionalityReductionOps.Full(new[] { n, _nComponents }, NumOps.One);
        var velocity = new Tensor<T>(new[] { n, _nComponents });
        var zero = NumOps.Zero;
        var gainDecay = NumOps.FromDouble(0.8);
        var gainStep = NumOps.FromDouble(0.2);
        var minGain = NumOps.FromDouble(0.01);
        var learningRate = NumOps.FromDouble(_learningRate);
        var inverseExaggeration = NumOps.FromDouble(1.0 / _earlyExaggeration);
        var momentum = NumOps.FromDouble(0.5);
        var lateMomentum = NumOps.FromDouble(0.8);
        int earlyExaggerationEnd = 250;

        for (int iter = 0; iter < _nIter; iter++)
        {
            // Update momentum
            if (iter == 250)
            {
                momentum = lateMomentum;
            }

            // Remove early exaggeration
            if (iter == earlyExaggerationEnd)
            {
                p = engine.TensorMultiplyScalar(p, inverseExaggeration);
            }

            // Compute Q (student-t distribution) and gradients
            var (q, kernel) = ComputeQ(y, n);
            var gradient = ComputeGradient(p, q, kernel, y);

            // Adaptive gains: shrink where gradient and velocity share a sign, grow otherwise
            var sameSign = engine.TensorEquals(engine.TensorGreaterThan(gradient, zero), engine.TensorGreaterThan(velocity, zero));
            gains = engine.TensorWhere(
                sameSign,
                engine.TensorClampMin(engine.TensorMultiplyScalar(gains, gainDecay), minGain),
                engine.TensorAddScalar(gains, gainStep));

            // Update velocity and position
            velocity = engine.TensorSubtract(
                engine.TensorMultiplyScalar(velocity, momentum),
                engine.TensorMultiplyScalar(engine.TensorMultiply(gains, gradient), learningRate));
            y = engine.TensorAdd(y, velocity);

            // Re-center
            y = DimensionalityReductionOps.SubtractRow(y, DimensionalityReductionOps.ColumnMeans(y));
        }

        _embedding = y.ToMatrix();
    }

    private Tensor<T> ComputeDistances(Tensor<T> x)
    {
        var engine = Engine;
        switch (_metric)
        {
            case TSNEMetric.Manhattan:
                return engine.TensorCDist(x, x, 1.0);

            case TSNEMetric.Cosine:
                // 1 - x_i . x_j / (|x_i| |x_j|), or 1 when the norm product is at most 1e-10; zero diagonal
                var dot = DimensionalityReductionOps.MatMulTranspose(x, x);
                var squaredNorms = DimensionalityReductionOps.RowSums(engine.TensorSquare(x));
                var denom = engine.TensorSqrt(engine.TensorMatMul(
                    DimensionalityReductionOps.AsColumn(squaredNorms), DimensionalityReductionOps.AsRow(squaredNorms)));
                var valid = engine.TensorGreaterThan(denom, DimensionalityReductionOps.Tolerance<T>(1e-10));
                var cosine = engine.TensorWhere(
                    valid,
                    engine.TensorSubtract(engine.TensorOnesLike(dot), engine.TensorDivide(dot, engine.TensorWhere(valid, denom, engine.TensorOnesLike(denom)))),
                    engine.TensorOnesLike(dot));
                engine.TensorFillDiagonalInPlace(cosine, NumOps.Zero);
                return cosine;

            default:
                return DimensionalityReductionOps.EuclideanDistances(x, x);
        }
    }

    /// <summary>
    /// Conditional probabilities p_j|i = exp(-d_ij^2 / (2 sigma_i^2)) / sum, with every row's sigma found by the
    /// same 50-step bisection on the entropy (all rows advanced together; a row stops once |H - log perplexity|
    /// &lt; 1e-5), then P = max((P + P^T) / (2n), 1e-12).
    /// </summary>
    private Tensor<T> ComputeJointProbabilities(Tensor<T> distances, int n)
    {
        var engine = Engine;
        var one = NumOps.One;
        var squared = engine.TensorSquare(distances);
        var offDiagonal = engine.TensorSubtract(
            DimensionalityReductionOps.Full(new[] { n, n }, one), DimensionalityReductionOps.Identity<T>(n));

        var sigma = DimensionalityReductionOps.Full(new[] { n, 1 }, one);
        var sigmaLow = DimensionalityReductionOps.Full(new[] { n, 1 }, DimensionalityReductionOps.Tolerance<T>(1e-20));
        var sigmaHigh = DimensionalityReductionOps.Full(new[] { n, 1 }, NumOps.FromDouble(1e20));
        var active = DimensionalityReductionOps.Full(new[] { n, 1 }, one);
        var conditional = new Tensor<T>(new[] { n, n });

        var logPerplexity = NumOps.FromDouble(Math.Log(_perplexity));
        var floor = DimensionalityReductionOps.Tolerance<T>(1e-10);
        var tolerance = DimensionalityReductionOps.Tolerance<T>(1e-5);
        var unboundedColumn = DimensionalityReductionOps.Full(new[] { n, 1 }, NumOps.FromDouble(1e19));
        var allOnes = DimensionalityReductionOps.Full(new[] { n, n }, one);
        var half = NumOps.FromDouble(0.5);
        var two = NumOps.FromDouble(2.0);

        for (int iter = 0; iter < 50; iter++)
        {
            // P_i|j with the current sigma, normalized where the row sum exceeds 1e-10
            var scale = engine.TensorMultiplyScalar(engine.TensorSquare(sigma), two);
            var unnormalized = engine.TensorMultiply(
                engine.TensorExp(engine.TensorNegate(engine.TensorDivide(squared, scale))), offDiagonal);
            var sumP = engine.ReduceSum(unnormalized, new[] { 1 }, true);
            var normalizable = engine.TensorGreaterThan(sumP, floor);
            var rows = engine.TensorDivide(unnormalized, engine.TensorWhere(normalizable, sumP, engine.TensorOnesLike(sumP)));

            // Entropy over entries above 1e-10 (zero for rows that could not be normalized)
            var significant = engine.TensorGreaterThan(rows, floor);
            var logs = engine.TensorLog(engine.TensorWhere(significant, rows, engine.TensorOnesLike(rows)));
            var entropy = engine.TensorMultiply(
                engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(rows, logs), new[] { 1 }, true)), normalizable);

            // Rows still searching take the new probabilities
            conditional = engine.TensorWhere(engine.TensorMultiply(allOnes, active), rows, conditional);

            // Adjust sigma for rows that have not converged
            var perpDiff = engine.TensorSubtractScalar(entropy, logPerplexity);
            var converged = engine.TensorLessThan(engine.TensorAbs(perpDiff), tolerance);
            active = engine.TensorMultiply(active, engine.TensorSubtract(engine.TensorOnesLike(converged), converged));
            if (!DimensionalityReductionOps.Above(engine.TensorSum(active), NumOps.Zero))
            {
                break;
            }

            var tooWide = engine.TensorGreaterThan(perpDiff, NumOps.Zero);
            var lowered = engine.TensorMultiplyScalar(engine.TensorAdd(sigmaLow, sigma), half);
            var raised = engine.TensorWhere(
                engine.TensorGreaterEqual(sigmaHigh, unboundedColumn),
                engine.TensorMultiplyScalar(sigma, two),
                engine.TensorMultiplyScalar(engine.TensorAdd(sigma, sigmaHigh), half));

            var newHigh = engine.TensorWhere(tooWide, sigma, sigmaHigh);
            var newLow = engine.TensorWhere(tooWide, sigmaLow, sigma);
            var newSigma = engine.TensorWhere(tooWide, lowered, raised);
            sigmaHigh = engine.TensorWhere(active, newHigh, sigmaHigh);
            sigmaLow = engine.TensorWhere(active, newLow, sigmaLow);
            sigma = engine.TensorWhere(active, newSigma, sigma);
        }

        // Symmetrize and ensure minimum probability
        var symmetric = engine.TensorDivideScalar(
            engine.TensorAdd(conditional, engine.TensorTranspose(conditional)), NumOps.FromDouble(2.0 * n));
        return engine.TensorClampMin(symmetric, DimensionalityReductionOps.Tolerance<T>(1e-12));
    }

    private Tensor<T> InitializeEmbedding(Tensor<T> x, int n, Random random)
    {
        var engine = Engine;
        switch (_initialization)
        {
            case TSNEInitialization.PCA:
                // PCA projection scaled so the first component has standard deviation 1e-4 (sklearn init="pca")
                var centered = DimensionalityReductionOps.SubtractRow(x, DimensionalityReductionOps.ColumnMeans(x));
                var (_, vectors) = DimensionalityReductionOps.SymmetricEigen(
                    DimensionalityReductionOps.TransposeMatMul(centered, centered), descending: true);
                int available = Math.Min(_nComponents, x.Shape[1]);
                var projected = engine.TensorMatMul(centered, DimensionalityReductionOps.FirstColumns(vectors, available));
                if (available < _nComponents)
                {
                    projected = engine.TensorConcatenate(
                        new[] { projected, new Tensor<T>(new[] { n, _nComponents - available }) }, 1);
                }

                var firstColumn = engine.TensorNarrow(projected, 1, 0, 1);
                var std = engine.TensorSqrt(engine.ReduceMean(engine.TensorSquare(firstColumn), new[] { 0 }, true));
                var scale = engine.TensorDivide(
                    DimensionalityReductionOps.Full(new[] { 1, 1 }, DimensionalityReductionOps.Tolerance<T>(1e-4)),
                    engine.TensorWhere(
                        engine.TensorGreaterThan(std, NumOps.Zero), std, engine.TensorOnesLike(std)));
                return engine.TensorMultiply(projected, scale);

            case TSNEInitialization.Random:
            default:
                // Small random values 1e-4 (u - 0.5)
                return engine.TensorMultiplyScalar(
                    engine.TensorSubtractScalar(DimensionalityReductionOps.Uniform<T>(random, n, _nComponents), NumOps.FromDouble(0.5)),
                    NumOps.FromDouble(0.0001));
        }
    }

    /// <summary>Returns (Q, kernel) with kernel = 1 / (1 + |y_i - y_j|^2) off the diagonal (zero on it).</summary>
    private (Tensor<T> Q, Tensor<T> Kernel) ComputeQ(Tensor<T> y, int n)
    {
        var engine = Engine;
        var kernel = engine.TensorReciprocal(engine.TensorAddScalar(DimensionalityReductionOps.SquaredDistances(y, y), NumOps.One));
        engine.TensorFillDiagonalInPlace(kernel, NumOps.Zero);

        // Normalize by the total (when positive) with a 1e-12 floor; the diagonal only enters through the kernel (zero)
        var sumQ = engine.ReduceSum(engine.Reshape(kernel, new[] { 1, n * n }), new[] { 1 }, true);
        var positive = engine.TensorGreaterThan(sumQ, NumOps.Zero);
        var q = engine.TensorClampMin(
            engine.TensorDivide(kernel, engine.TensorWhere(positive, sumQ, engine.TensorOnesLike(sumQ))), DimensionalityReductionOps.Tolerance<T>(1e-12));
        return (q, kernel);
    }

    /// <summary>dC/dy_i = sum_j 4 (p_ij - q_ij) (y_i - y_j) / (1 + |y_i - y_j|^2).</summary>
    private Tensor<T> ComputeGradient(Tensor<T> p, Tensor<T> q, Tensor<T> kernel, Tensor<T> y)
    {
        var engine = Engine;
        var weights = engine.TensorMultiplyScalar(engine.TensorMultiply(engine.TensorSubtract(p, q), kernel), NumOps.FromDouble(4.0));
        return engine.TensorSubtract(
            DimensionalityReductionOps.ScaleRows(y, DimensionalityReductionOps.RowSums(weights)),
            engine.TensorMatMul(weights, y));
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("TSNE has not been fitted.");
        }

        // t-SNE doesn't support out-of-sample transformation
        // Return the fitted embedding if it's the same data
        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "t-SNE does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TSNE does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"TSNE{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies the distance metric for t-SNE.
/// </summary>
public enum TSNEMetric
{
    /// <summary>
    /// Euclidean (L2) distance.
    /// </summary>
    Euclidean,

    /// <summary>
    /// Manhattan (L1) distance.
    /// </summary>
    Manhattan,

    /// <summary>
    /// Cosine distance (1 - cosine similarity).
    /// </summary>
    Cosine
}

/// <summary>
/// Specifies the initialization method for t-SNE.
/// </summary>
public enum TSNEInitialization
{
    /// <summary>
    /// Random initialization from a small normal distribution.
    /// </summary>
    Random,

    /// <summary>
    /// Initialize using PCA projection.
    /// </summary>
    PCA
}
