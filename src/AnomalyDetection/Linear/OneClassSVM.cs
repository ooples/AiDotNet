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
public class OneClassSVM<T> : AnomalyDetectorBase<T>
{
    private readonly double _nu;
    private readonly double _gamma;
    private readonly int _maxIterations;
    private readonly double _tolerance;

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

        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "Max iterations must be at least 1.");
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

        // Auto-detect gamma if not specified
        double effectiveGamma = _gamma > 0 ? _gamma : 1.0 / numFeatures;

        // Compute kernel matrix
        var K = ComputeKernelMatrix(X, X, effectiveGamma);

        // Solve the dual problem using simplified SMO
        var alphas = SolveDual(K, n);

        // Extract support vectors (points with alpha > 0)
        var svIndices = new List<int>();
        var svAlphas = new List<T>();

        T alphaThreshold = NumOps.FromDouble(1e-7);

        for (int i = 0; i < n; i++)
        {
            if (NumOps.GreaterThan(alphas[i], alphaThreshold))
            {
                svIndices.Add(i);
                svAlphas.Add(alphas[i]);
            }
        }

        // Store support vectors
        _supportVectors = new Matrix<T>(svIndices.Count, numFeatures);
        _alphas = new Vector<T>(svIndices.Count);

        for (int i = 0; i < svIndices.Count; i++)
        {
            _supportVectors.SetRow(i, X.GetRow(svIndices[i]));
            _alphas[i] = svAlphas[i];
        }

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

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = X.GetRow(i);
            T score = NumOps.Zero;

            for (int j = 0; j < supportVectors.Rows; j++)
            {
                T kernelVal = RbfKernel(point, supportVectors.GetRow(j), gamma);
                score = NumOps.Add(score, NumOps.Multiply(alphas[j], kernelVal));
            }

            // Decision function: sign(sum(alpha_i * K(x, x_i)) - rho)
            // We return the signed distance, not just the sign
            // Negate so higher values = more anomalous (consistent with other detectors)
            scores[i] = NumOps.Negate(NumOps.Subtract(score, _rho));
        }

        return scores;
    }

    private Matrix<T> ComputeKernelMatrix(Matrix<T> X1, Matrix<T> X2, double gamma)
    {
        var K = new Matrix<T>(X1.Rows, X2.Rows);

        for (int i = 0; i < X1.Rows; i++)
        {
            for (int j = 0; j < X2.Rows; j++)
            {
                K[i, j] = RbfKernel(X1.GetRow(i), X2.GetRow(j), gamma);
            }
        }

        return K;
    }

    private T RbfKernel(Vector<T> x1, Vector<T> x2, double gamma)
    {
        // Use StatisticsHelper for distance calculation
        T dist = StatisticsHelper<T>.EuclideanDistance(x1, x2);
        double squaredDist = Math.Pow(NumOps.ToDouble(dist), 2);
        return NumOps.FromDouble(Math.Exp(-gamma * squaredDist));
    }

    private Vector<T> SolveDual(Matrix<T> K, int n)
    {
        // Simplified SMO for one-class SVM
        // Constraint: sum(alpha) = 1, 0 <= alpha_i <= 1/(n*nu)

        T upperBound = NumOps.FromDouble(1.0 / (n * _nu));
        var alphas = new Vector<T>(n);

        // Initialize alphas uniformly
        T initialAlpha = NumOps.FromDouble(1.0 / n);
        for (int i = 0; i < n; i++)
        {
            alphas[i] = initialAlpha;
        }

        // SMO iterations
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            bool changed = false;

            for (int i = 0; i < n; i++)
            {
                // Select a random j != i
                int j = _random.Next(n);
                while (j == i)
                {
                    j = _random.Next(n);
                }

                // Compute gradient components
                T gi = NumOps.Zero;
                T gj = NumOps.Zero;

                for (int k = 0; k < n; k++)
                {
                    gi = NumOps.Add(gi, NumOps.Multiply(alphas[k], K[i, k]));
                    gj = NumOps.Add(gj, NumOps.Multiply(alphas[k], K[j, k]));
                }

                // Compute optimal step
                T kii = K[i, i];
                T kjj = K[j, j];
                T kij = K[i, j];
                T eta = NumOps.Subtract(NumOps.Add(kii, kjj), NumOps.Multiply(NumOps.FromDouble(2), kij));

                if (NumOps.LessThanOrEquals(eta, NumOps.Zero))
                {
                    continue;
                }

                // Update direction
                T delta = NumOps.Divide(NumOps.Subtract(gi, gj), eta);

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
                if (NumOps.GreaterThan(change, NumOps.FromDouble(_tolerance)))
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
        // Rho is computed from support vectors (0 < alpha < upperBound)
        T upperBound = NumOps.FromDouble(1.0 / (X.Rows * _nu));
        T lowerThreshold = NumOps.FromDouble(1e-7);
        T upperThreshold = NumOps.Subtract(upperBound, lowerThreshold);

        T rhoSum = NumOps.Zero;
        int count = 0;

        for (int i = 0; i < X.Rows; i++)
        {
            if (NumOps.GreaterThan(alphas[i], lowerThreshold) &&
                NumOps.LessThan(alphas[i], upperThreshold))
            {
                T score = NumOps.Zero;
                for (int j = 0; j < X.Rows; j++)
                {
                    score = NumOps.Add(score, NumOps.Multiply(alphas[j], K[i, j]));
                }

                rhoSum = NumOps.Add(rhoSum, score);
                count++;
            }
        }

        if (count > 0)
        {
            return NumOps.Divide(rhoSum, NumOps.FromDouble(count));
        }

        // Fallback: use mean of all scores
        for (int i = 0; i < X.Rows; i++)
        {
            T score = NumOps.Zero;
            for (int j = 0; j < X.Rows; j++)
            {
                score = NumOps.Add(score, NumOps.Multiply(alphas[j], K[i, j]));
            }

            rhoSum = NumOps.Add(rhoSum, score);
        }

        return NumOps.Divide(rhoSum, NumOps.FromDouble(X.Rows));
    }
}
