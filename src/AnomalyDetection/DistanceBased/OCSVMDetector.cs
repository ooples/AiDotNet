using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.DistanceBased;

/// <summary>
/// Detects anomalies using One-Class SVM with simplified SGD training.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> One-Class SVM finds a boundary that encompasses the normal data.
/// It learns the shape of normal data in a high-dimensional kernel space and flags points
/// outside this region as anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Map data to kernel space using RBF kernel
/// 2. Find hyperplane that separates data from origin with maximum margin
/// 3. Points on the negative side of the hyperplane are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you have only normal examples
/// - High-dimensional data
/// - When you need a decision boundary
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Nu: 0.1 (roughly proportion of outliers)
/// - Gamma: auto (1/n_features)
/// - Kernel: RBF
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Schölkopf, B., et al. (2001). "Estimating the Support of a High-Dimensional Distribution."
/// </para>
/// </remarks>
public class OCSVMDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _nu;
    private readonly double _gamma;
    private readonly int _maxIterations;
    private double[][]? _supportVectors;
    private double[]? _alphas;
    private double _rho;

    /// <summary>
    /// Gets the nu parameter.
    /// </summary>
    public double Nu => _nu;

    /// <summary>
    /// Gets the gamma parameter.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Creates a new OCSVM anomaly detector.
    /// </summary>
    /// <param name="nu">Upper bound on outlier fraction. Default is 0.1.</param>
    /// <param name="gamma">RBF kernel parameter. -1 means auto (1/n_features). Default is -1.</param>
    /// <param name="maxIterations">Maximum training iterations. Default is 1000.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public OCSVMDetector(double nu = 0.1, double gamma = -1, int maxIterations = 1000,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nu <= 0 || nu > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nu),
                "Nu must be between 0 (exclusive) and 1 (inclusive). Recommended is 0.1.");
        }

        if (maxIterations < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(maxIterations),
                "MaxIterations must be at least 1.");
        }

        _nu = nu;
        _gamma = gamma;
        _maxIterations = maxIterations;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int d = X.Columns;

        // Convert to double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[d];
            for (int j = 0; j < d; j++)
            {
                data[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        double effectiveGamma = _gamma > 0 ? _gamma : 1.0 / d;

        // Train using simplified SMO-like algorithm
        TrainOCSVM(data, effectiveGamma);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private void TrainOCSVM(double[][] data, double gamma)
    {
        int n = data.Length;

        // Initialize alphas
        _alphas = new double[n];
        double sumAlpha = 1.0 / (_nu * n);

        // Start with uniform alphas bounded by 1/(nu*n)
        for (int i = 0; i < n; i++)
        {
            _alphas[i] = 1.0 / n;
        }

        // Compute kernel matrix (cache for efficiency)
        var K = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                K[i, j] = RBFKernel(data[i], data[j], gamma);
                K[j, i] = K[i, j];
            }
        }

        // SMO-like optimization
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            bool changed = false;

            for (int i = 0; i < n; i++)
            {
                // Compute decision function value
                double fi = 0;
                for (int k = 0; k < n; k++)
                {
                    fi += _alphas[k] * K[k, i];
                }

                // Check KKT conditions (simplified)
                double maxAlpha = 1.0 / (_nu * n);

                // If violates KKT, update
                if ((_alphas[i] < maxAlpha && fi < 0) || (_alphas[i] > 0 && fi > 0))
                {
                    // Select random j != i
                    int j = _random.Next(n);
                    while (j == i) j = _random.Next(n);

                    // Compute bounds
                    double fj = 0;
                    for (int k = 0; k < n; k++)
                    {
                        fj += _alphas[k] * K[k, j];
                    }

                    // Compute step
                    double eta = K[i, i] + K[j, j] - 2 * K[i, j];
                    if (eta > 1e-10)
                    {
                        double oldAlphaI = _alphas[i];
                        double oldAlphaJ = _alphas[j];

                        // Update (simplified)
                        double delta = (fi - fj) / eta;
                        _alphas[i] -= delta * 0.1;
                        _alphas[j] += delta * 0.1;

                        // Clip
                        _alphas[i] = Math.Max(0, Math.Min(maxAlpha, _alphas[i]));
                        _alphas[j] = Math.Max(0, Math.Min(maxAlpha, _alphas[j]));

                        if (Math.Abs(_alphas[i] - oldAlphaI) > 1e-8)
                        {
                            changed = true;
                        }
                    }
                }
            }

            if (!changed) break;
        }

        // Extract support vectors
        var svIndices = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (_alphas[i] > 1e-8)
            {
                svIndices.Add(i);
            }
        }

        _supportVectors = svIndices.Select(i => data[i]).ToArray();
        var svAlphas = svIndices.Select(i => _alphas[i]).ToArray();
        _alphas = svAlphas;

        // Compute rho (threshold)
        _rho = 0;
        int count = 0;
        for (int i = 0; i < svIndices.Count; i++)
        {
            if (_alphas[i] > 1e-8 && _alphas[i] < 1.0 / (_nu * n) - 1e-8)
            {
                double f = 0;
                for (int j = 0; j < svIndices.Count; j++)
                {
                    f += _alphas[j] * RBFKernel(_supportVectors[j], _supportVectors[i], gamma);
                }
                _rho += f;
                count++;
            }
        }

        if (count > 0) _rho /= count;
    }

    private static double RBFKernel(double[] a, double[] b, double gamma)
    {
        double sqDist = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sqDist += diff * diff;
        }
        return Math.Exp(-gamma * sqDist);
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        var supportVectors = _supportVectors;
        var alphas = _alphas;

        if (supportVectors == null || alphas == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        double effectiveGamma = _gamma > 0 ? _gamma : 1.0 / X.Columns;
        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = NumOps.ToDouble(X[i, j]);
            }

            // Decision function: sum(alpha_i * K(x_i, x)) - rho
            double f = 0;
            for (int k = 0; k < supportVectors.Length; k++)
            {
                f += alphas[k] * RBFKernel(supportVectors[k], point, effectiveGamma);
            }
            f -= _rho;

            // Higher (more positive) = more normal
            // We want higher score = more anomalous, so negate
            scores[i] = NumOps.FromDouble(-f);
        }

        return scores;
    }
}
