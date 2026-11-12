using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Linear;

/// <summary>
/// Implements One-Class SVM for novelty/outlier detection using the RBF kernel.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> One-Class SVM learns a "boundary" around your normal data points.
/// New points that fall outside this boundary are considered outliers. It's like drawing
/// a flexible shape around your data - anything outside the shape is unusual.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Mapping data to a high-dimensional space using the RBF (Radial Basis Function) kernel
/// 2. Finding a hyperplane that separates most data points from the origin
/// 3. Points far from this hyperplane (on the wrong side) are outliers
/// </para>
/// <para>
/// <b>When to use:</b> One-Class SVM is particularly effective for:
/// - Novelty detection (training only on "normal" data)
/// - When you have a clear notion of what "normal" looks like
/// - Data with complex, non-linear boundaries
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Nu: 0.1 (upper bound on outlier fraction)
/// - Gamma: Auto-detect (1/n_features)
/// - Max iterations: 1000
/// - Tolerance: 1e-3
/// </para>
/// <para>
/// Reference: Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., and Williamson, R. C. (2001).
/// "Estimating the Support of a High-Dimensional Distribution." Neural Computation.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.SVM)]
[ModelCategory(ModelCategory.Kernel)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Estimating the Support of a High-Dimensional Distribution", "https://doi.org/10.1162/089976601750264965", Year = 2001, Authors = "Bernhard Scholkopf, John C. Platt, John Shawe-Taylor, Alex J. Smola, Robert C. Williamson")]
public partial class OneClassSVM<T> : AnomalyDetectorBase<T>
{
    private readonly double _nu;
    private readonly double _gamma;
    private readonly int _maxIterations;
    private readonly double _tolerance;

    [Buffer]
    private Matrix<T>? _supportVectors;
    private Vector<T>? _alphas;
    private T _rho;

    /// <summary>
    /// Gets the nu parameter (upper bound on outlier fraction).
    /// </summary>
    public double Nu => _nu;

    /// <summary>
    /// Gets the gamma parameter for the RBF kernel.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Gets the number of support vectors after fitting.
    /// </summary>
    public int NumSupportVectors => _supportVectors?.Rows ?? 0;

    /// <summary>
    /// Creates a new One-Class SVM anomaly detector.
    /// </summary>
    /// <param name="nu">
    /// An upper bound on the fraction of training errors and a lower bound on the fraction
    /// of support vectors. Should be in (0, 1]. Default is 0.1.
    /// Roughly corresponds to the expected proportion of outliers.
    /// </param>
    /// <param name="gamma">
    /// Kernel coefficient for RBF. Default is 0 which uses 1/(n_features) auto-detection.
    /// Larger values mean each point has a smaller "influence radius".
    /// </param>
    /// <param name="maxIterations">
    /// Maximum number of iterations for the solver. Default is 1000.
    /// </param>
    /// <param name="tolerance">
    /// Tolerance for stopping criterion. Default is 1e-3.
    /// </param>
    /// <param name="contamination">
    /// The expected proportion of anomalies in the data. Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The most important parameter is nu:
    /// - nu controls how "tight" the boundary is around your data
    /// - Lower nu = tighter boundary, fewer outliers
    /// - Higher nu = looser boundary, more outliers
    ///
    /// Start with nu equal to your expected outlier proportion (e.g., 0.1 for 10%).
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when nu is not in (0, 1] or maxIterations is less than 1.
    /// </exception>
    public OneClassSVM(
        double nu = 0.1,
        double gamma = 0,
        int maxIterations = 1000,
        double tolerance = 1e-3,
        double contamination = 0.1,
        int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nu <= 0 || nu > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nu),
                "Nu must be in the range (0, 1]. " +
                "It roughly represents the expected proportion of outliers.");
        }

        if (gamma < 0 || double.IsNaN(gamma) || double.IsInfinity(gamma))
        {
            throw new ArgumentOutOfRangeException(nameof(gamma),
                "Gamma must be a non-negative, finite number. Use 0 for auto-detection.");
        }

        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "Max iterations must be at least 1.");
        }

        if (tolerance <= 0 || double.IsNaN(tolerance) || double.IsInfinity(tolerance))
        {
            throw new ArgumentOutOfRangeException(nameof(tolerance),
                "Tolerance must be a positive, finite value.");
        }

        _nu = nu;
        _gamma = gamma;
        _maxIterations = maxIterations;
        _tolerance = tolerance;
        _rho = NumOps.Zero;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int numFeatures = X.Columns;

        // One-class SVM requires at least 2 samples for meaningful optimization
        if (n < 2)
        {
            throw new ArgumentException(
                $"One-class SVM requires at least 2 samples, but got {n}.",
                nameof(X));
        }

        // Auto-detect gamma if not specified
        double effectiveGamma = _gamma > 0 ? _gamma : 1.0 / numFeatures;

        // Compute kernel matrix
        var K = ComputeKernelMatrix(X, X, effectiveGamma);

        // Solve the dual problem using simplified SMO
        var alphas = SolveDual(K, n);

        // Extract support vectors (points with alpha > 1e-7)
        var svIndices = AnomalyTensorOps.NonzeroIndices(Engine.TensorGreaterThan(alphas.AsTensor(), NumOps.FromDouble(1e-7)));
        _supportVectors = AnomalyTensorOps.SelectRows(AnomalyTensorOps.ToTensor(X), svIndices).ToMatrix();
        _alphas = AnomalyTensorOps.SelectElements(alphas, svIndices);

        // Compute rho (threshold)
        _rho = ComputeRho(X, alphas, K, effectiveGamma);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X, effectiveGamma);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();

        var supportVectors = _supportVectors;
        if (supportVectors == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        double effectiveGamma = _gamma > 0 ? _gamma : 1.0 / supportVectors.Columns;
        return ScoreAnomaliesInternal(X, effectiveGamma);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X, double gamma)
    {
        ValidateInput(X);

        var supportVectors = _supportVectors;
        var alphas = _alphas;
        if (supportVectors == null || alphas == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        if (X.Columns != supportVectors.Columns)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {supportVectors.Columns} features.",
                nameof(X));
        }

        // Decision function: sign(sum(alpha_i * K(x, x_i)) - rho)
        // We return the signed distance, not just the sign
        // Negate so higher values = more anomalous (consistent with other detectors)
        // Query rows are scored in bounded blocks: only a [block, nSupport] kernel is held at a time
        var engine = Engine;
        var support = AnomalyTensorOps.ToTensor(supportVectors);
        var alphaColumn = engine.Reshape(alphas.AsTensor(), new[] { alphas.Length, 1 });
        T negGamma = NumOps.FromDouble(-gamma);
        var decision = AnomalyTensorOps.MapRowBlocks(AnomalyTensorOps.ToTensor(X), supportVectors.Rows,
            block => engine.Reshape(engine.TensorMatMul(engine.TensorExp(engine.TensorMultiplyScalar(
                AnomalyTensorOps.PairwiseSquaredDistances(block, support), negGamma)), alphaColumn),
                new[] { block.Shape[0] }));
        return engine.TensorNegate(engine.TensorSubtractScalar(decision, _rho)).ToVector();
    }

    // RBF kernel exp(-gamma * ||x1_i - x2_j||^2) between the rows of X1 and X2.
    private Matrix<T> ComputeKernelMatrix(Matrix<T> X1, Matrix<T> X2, double gamma)
    {
        var engine = Engine;
        var squared = AnomalyTensorOps.PairwiseSquaredDistances(AnomalyTensorOps.ToTensor(X1), AnomalyTensorOps.ToTensor(X2));
        return engine.TensorExp(engine.TensorNegate(engine.TensorMultiplyScalar(squared, NumOps.FromDouble(gamma)))).ToMatrix();
    }

    private Vector<T> SolveDual(Matrix<T> K, int n)
    {
        // Simplified SMO for one-class SVM
        // Constraint: sum(alpha) = 1, 0 <= alpha_i <= 1/(n*nu)

        T upperBound = NumOps.FromDouble(1.0 / (n * _nu));
        // Initialize alphas uniformly
        T initialAlpha = NumOps.FromDouble(1.0 / n);
        var alphas = Engine.Fill(n, initialAlpha);

        // SMO iterations
        // SMO requires at least 2 samples to optimize pairs
        if (n < 2)
        {
            return alphas;
        }

        var kernelRows = AnomalyTensorOps.RowVectors(AnomalyTensorOps.ToTensor(K));
        T two = NumOps.FromDouble(2);
        T tolerance = NumOps.FromDouble(_tolerance);
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            bool changed = false;

            for (int i = 0; i < n; i++)
            {
                // Select a random j != i using modular arithmetic to avoid infinite loop
                int offset = 1 + _random.Next(n - 1);
                int j = (i + offset) % n;

                // Compute gradient components
                T gi = Engine.DotProduct(alphas, kernelRows[i]);
                T gj = Engine.DotProduct(alphas, kernelRows[j]);

                // Compute optimal step
                T kii = K[i, i];
                T kjj = K[j, j];
                T kij = K[i, j];
                T eta = NumOps.Subtract(NumOps.Add(kii, kjj), NumOps.Multiply(two, kij));

                if (NumOps.LessThanOrEquals(eta, NumOps.Zero))
                {
                    continue;
                }

                // Update direction
                // The dual objective 1/2 a'Ka changes by delta * (g_i - g_j) + delta^2 * eta / 2 when
                // a_i += delta and a_j -= delta, so its minimum along that line is at (g_j - g_i) / eta
                T delta = NumOps.Divide(NumOps.Subtract(gj, gi), eta);

                // Clip delta to keep alphas in bounds
                T oldAlphaI = alphas[i];
                T oldAlphaJ = alphas[j];

                // alphas[i] += delta, alphas[j] -= delta (maintaining sum = 1)
                T newAlphaI = NumOps.Add(oldAlphaI, delta);
                T newAlphaJ = NumOps.Subtract(oldAlphaJ, delta);

                // Clip to [0, upperBound]
                if (NumOps.LessThan(newAlphaI, NumOps.Zero))
                {
                    newAlphaI = NumOps.Zero;
                    newAlphaJ = NumOps.Add(oldAlphaI, oldAlphaJ);
                }
                else if (NumOps.GreaterThan(newAlphaI, upperBound))
                {
                    newAlphaI = upperBound;
                    newAlphaJ = NumOps.Subtract(NumOps.Add(oldAlphaI, oldAlphaJ), upperBound);
                }

                if (NumOps.LessThan(newAlphaJ, NumOps.Zero))
                {
                    newAlphaJ = NumOps.Zero;
                    newAlphaI = NumOps.Add(oldAlphaI, oldAlphaJ);
                }
                else if (NumOps.GreaterThan(newAlphaJ, upperBound))
                {
                    newAlphaJ = upperBound;
                    newAlphaI = NumOps.Subtract(NumOps.Add(oldAlphaI, oldAlphaJ), upperBound);
                }

                // Final clipping
                if (NumOps.LessThan(newAlphaI, NumOps.Zero)) newAlphaI = NumOps.Zero;
                if (NumOps.GreaterThan(newAlphaI, upperBound)) newAlphaI = upperBound;
                if (NumOps.LessThan(newAlphaJ, NumOps.Zero)) newAlphaJ = NumOps.Zero;
                if (NumOps.GreaterThan(newAlphaJ, upperBound)) newAlphaJ = upperBound;

                T change = NumOps.Abs(NumOps.Subtract(newAlphaI, oldAlphaI));
                if (NumOps.GreaterThan(change, tolerance))
                {
                    alphas[i] = newAlphaI;
                    alphas[j] = newAlphaJ;
                    changed = true;
                }
            }

            if (!changed)
            {
                break;
            }
        }

        return alphas;
    }

    private T ComputeRho(Matrix<T> X, Vector<T> alphas, Matrix<T> K, double gamma)
    {
        // Rho is the mean decision value sum_j alpha_j K[i, j] over the free support vectors
        // (1e-7 < alpha < upperBound - 1e-7), or over all points when there are none
        var engine = Engine;
        int n = X.Rows;
        T upperBound = NumOps.FromDouble(1.0 / (n * _nu));
        T lowerThreshold = NumOps.FromDouble(1e-7);
        T upperThreshold = NumOps.Subtract(upperBound, lowerThreshold);
        var alphaTensor = alphas.AsTensor();
        var scores = engine.Reshape(engine.TensorMatMul(AnomalyTensorOps.ToTensor(K),
            engine.Reshape(alphaTensor, new[] { n, 1 })), new[] { n });
        var free = engine.TensorMultiply(engine.TensorGreaterThan(alphaTensor, lowerThreshold),
            engine.TensorLessThan(alphaTensor, upperThreshold));
        T count = engine.TensorSum(free);
        if (NumOps.GreaterThan(count, NumOps.Zero))
            return NumOps.Divide(engine.TensorSum(engine.TensorWhere(free, scores, Tensor<T>.CreateZeros(new[] { n }))), count);

        // Fallback: use mean of all scores
        return NumOps.Divide(engine.TensorSum(scores), NumOps.FromDouble(n));
    }
}
