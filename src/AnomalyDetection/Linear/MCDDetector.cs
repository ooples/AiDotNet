using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Linear;

/// <summary>
/// Detects anomalies using Minimum Covariance Determinant (MCD).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> MCD is a robust method for estimating the center and spread of data.
/// Unlike standard mean and covariance, MCD is resistant to outliers by finding the subset
/// of points that minimizes the covariance determinant. Points far from this robust estimate
/// are anomalies.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Find the subset of h points with minimum covariance determinant
/// 2. Compute robust mean and covariance from this subset
/// 3. Compute Mahalanobis distances using robust estimates
/// 4. Points with large distances are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Data with outliers that corrupt standard statistics
/// - When you need robust location/scatter estimates
/// - Multivariate anomaly detection
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Support fraction: 0.5 (use 50% of data for robust estimate)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Rousseeuw, P.J. (1984). "Least Median of Squares Regression."
/// Rousseeuw, P.J., Driessen, K.V. (1999). "A Fast Algorithm for the Minimum Covariance Determinant Estimator."
/// </para>
/// </remarks>
public class MCDDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _supportFraction;
    private Vector<T>? _robustMean;
    private Matrix<T>? _robustPrecision;
    private int _nFeatures;

    /// <summary>
    /// Gets the support fraction.
    /// </summary>
    public double SupportFraction => _supportFraction;

    /// <summary>
    /// Creates a new MCD anomaly detector.
    /// </summary>
    /// <param name="supportFraction">
    /// Fraction of data to use for robust estimate (0.5-1.0). Default is 0.5.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public MCDDetector(double supportFraction = 0.5, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (supportFraction < 0.5 || supportFraction > 1.0)
        {
            throw new ArgumentOutOfRangeException(nameof(supportFraction),
                "SupportFraction must be between 0.5 and 1.0. Recommended is 0.5.");
        }

        _supportFraction = supportFraction;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _nFeatures = X.Columns;

        // Validate that we have more samples than features for covariance estimation
        if (n <= _nFeatures)
        {
            throw new ArgumentException(
                $"MCD requires more samples ({n}) than features ({_nFeatures}). " +
                "Consider reducing dimensions or adding more samples.",
                nameof(X));
        }

        // Compute h and ensure it's at least d + 1 for valid covariance estimation
        int h = Math.Max(_nFeatures + 1, (int)(n * _supportFraction));

        // Convert to double array
        var data = new double[n][];
        for (int i = 0; i < n; i++)
        {
            data[i] = new double[_nFeatures];
            for (int j = 0; j < _nFeatures; j++)
            {
                data[i][j] = NumOps.ToDouble(X[i, j]);
            }
        }

        // Run Fast-MCD algorithm (simplified)
        var (robustMeanDouble, robustCov) = FastMCD(data, h);

        // Compute precision matrix (inverse of covariance)
        var robustPrecisionDouble = InvertMatrix(robustCov);

        // Convert to generic types for storage
        _robustMean = new Vector<T>(_nFeatures);
        for (int j = 0; j < _nFeatures; j++)
        {
            _robustMean[j] = NumOps.FromDouble(robustMeanDouble[j]);
        }

        _robustPrecision = new Matrix<T>(_nFeatures, _nFeatures);
        for (int j1 = 0; j1 < _nFeatures; j1++)
        {
            for (int j2 = 0; j2 < _nFeatures; j2++)
            {
                _robustPrecision[j1, j2] = NumOps.FromDouble(robustPrecisionDouble[j1, j2]);
            }
        }

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private (double[] mean, double[,] covariance) FastMCD(double[][] data, int h)
    {
        int n = data.Length;
        int nTrials = 10;

        double[]? bestMean = null;
        double[,]? bestCov = null;
        double bestDet = double.MaxValue;

        for (int trial = 0; trial < nTrials; trial++)
        {
            // Random initial subset
            var indices = Enumerable.Range(0, n)
                .OrderBy(_ => _random.NextDouble())
                .Take(h)
                .ToArray();

            // Compute mean and covariance of subset
            var (mean, cov) = ComputeMeanCovariance(data, indices);

            // C-step iterations
            for (int iter = 0; iter < 10; iter++)
            {
                // Compute Mahalanobis distances
                var precision = InvertMatrix(cov);
                var distances = new (double dist, int idx)[n];

                for (int i = 0; i < n; i++)
                {
                    distances[i] = (MahalanobisDistanceDouble(data[i], mean, precision), i);
                }

                // Select h points with smallest distances
                indices = distances
                    .OrderBy(d => d.dist)
                    .Take(h)
                    .Select(d => d.idx)
                    .ToArray();

                // Recompute mean and covariance
                (mean, cov) = ComputeMeanCovariance(data, indices);
            }

            // Check if this is the best solution
            double det = Determinant(cov);
            if (det < bestDet && det > 1e-10)
            {
                bestDet = det;
                bestMean = mean;
                bestCov = cov;
            }
        }

        // Fallback to sample mean/cov if MCD failed
        if (bestMean == null || bestCov == null)
        {
            var allIndices = Enumerable.Range(0, n).ToArray();
            (bestMean, bestCov) = ComputeMeanCovariance(data, allIndices);
        }

        return (bestMean, bestCov);
    }

    private (double[] mean, double[,] covariance) ComputeMeanCovariance(double[][] data, int[] indices)
    {
        int h = indices.Length;
        var mean = new double[_nFeatures];

        // Compute mean
        foreach (int i in indices)
        {
            for (int j = 0; j < _nFeatures; j++)
            {
                mean[j] += data[i][j];
            }
        }
        for (int j = 0; j < _nFeatures; j++)
        {
            mean[j] /= h;
        }

        // Compute covariance
        var cov = new double[_nFeatures, _nFeatures];
        foreach (int i in indices)
        {
            for (int j1 = 0; j1 < _nFeatures; j1++)
            {
                for (int j2 = 0; j2 < _nFeatures; j2++)
                {
                    cov[j1, j2] += (data[i][j1] - mean[j1]) * (data[i][j2] - mean[j2]);
                }
            }
        }

        for (int j1 = 0; j1 < _nFeatures; j1++)
        {
            for (int j2 = 0; j2 < _nFeatures; j2++)
            {
                cov[j1, j2] /= (h - 1);
            }
            // Add small regularization
            cov[j1, j1] += 1e-6;
        }

        return (mean, cov);
    }

    private double MahalanobisDistanceDouble(double[] point, double[] mean, double[,] precision)
    {
        double dist = 0;
        for (int i = 0; i < _nFeatures; i++)
        {
            double diff1 = point[i] - mean[i];
            for (int j = 0; j < _nFeatures; j++)
            {
                double diff2 = point[j] - mean[j];
                dist += diff1 * precision[i, j] * diff2;
            }
        }
        return Math.Sqrt(Math.Max(0, dist));
    }

    private T MahalanobisDistance(double[] point, Vector<T> mean, Matrix<T> precision)
    {
        T dist = NumOps.Zero;
        for (int i = 0; i < _nFeatures; i++)
        {
            T diff1 = NumOps.Subtract(NumOps.FromDouble(point[i]), mean[i]);
            for (int j = 0; j < _nFeatures; j++)
            {
                T diff2 = NumOps.Subtract(NumOps.FromDouble(point[j]), mean[j]);
                T contrib = NumOps.Multiply(diff1, NumOps.Multiply(precision[i, j], diff2));
                dist = NumOps.Add(dist, contrib);
            }
        }
        double distDouble = NumOps.ToDouble(dist);
        return NumOps.FromDouble(Math.Sqrt(Math.Max(0, distDouble)));
    }

    private double[,] InvertMatrix(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var result = new double[n, n];
        var temp = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                temp[i, j] = matrix[i, j];
                result[i, j] = (i == j) ? 1.0 : 0.0;
            }
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(temp[row, col]) > Math.Abs(temp[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            if (maxRow != col)
            {
                for (int j = 0; j < n; j++)
                {
                    double tmpVal = temp[col, j];
                    temp[col, j] = temp[maxRow, j];
                    temp[maxRow, j] = tmpVal;

                    tmpVal = result[col, j];
                    result[col, j] = result[maxRow, j];
                    result[maxRow, j] = tmpVal;
                }
            }

            double pivot = temp[col, col];
            if (Math.Abs(pivot) < 1e-10) pivot = 1e-10;

            for (int j = 0; j < n; j++)
            {
                temp[col, j] /= pivot;
                result[col, j] /= pivot;
            }

            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    double factor = temp[row, col];
                    for (int j = 0; j < n; j++)
                    {
                        temp[row, j] -= factor * temp[col, j];
                        result[row, j] -= factor * result[col, j];
                    }
                }
            }
        }

        return result;
    }

    private double Determinant(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        var temp = new double[n, n];

        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                temp[i, j] = matrix[i, j];

        double det = 1;
        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(temp[row, col]) > Math.Abs(temp[maxRow, col]))
                    maxRow = row;
            }

            if (maxRow != col)
            {
                det *= -1;
                for (int j = col; j < n; j++)
                {
                    double tmp = temp[col, j];
                    temp[col, j] = temp[maxRow, j];
                    temp[maxRow, j] = tmp;
                }
            }

            if (Math.Abs(temp[col, col]) < 1e-10) return 0;

            det *= temp[col, col];

            for (int row = col + 1; row < n; row++)
            {
                double factor = temp[row, col] / temp[col, col];
                for (int j = col + 1; j < n; j++)
                {
                    temp[row, j] -= factor * temp[col, j];
                }
            }
        }

        return det;
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

        if (X.Columns != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nFeatures} features.",
                nameof(X));
        }

        var robustMean = _robustMean;
        var robustPrecision = _robustPrecision;

        if (robustMean == null || robustPrecision == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        var scores = new Vector<T>(X.Rows);

        for (int i = 0; i < X.Rows; i++)
        {
            var point = new double[X.Columns];
            for (int j = 0; j < X.Columns; j++)
            {
                point[j] = NumOps.ToDouble(X[i, j]);
            }

            scores[i] = MahalanobisDistance(point, robustMean, robustPrecision);
        }

        return scores;
    }
}
