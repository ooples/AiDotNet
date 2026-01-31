using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Linear;

/// <summary>
/// Detects anomalies using Elliptic Envelope (robust covariance estimation).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Elliptic Envelope fits an ellipse (in 2D) or ellipsoid (in higher dimensions)
/// around the data using robust estimation. Points far from this envelope are anomalies.
/// It's like drawing the smallest ellipse that contains most of the data.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Estimate robust mean and covariance using Minimum Covariance Determinant (MCD)
/// 2. Compute Mahalanobis distance for each point
/// 3. Points with large distances are anomalies
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Data is approximately Gaussian/elliptical
/// - You need robustness against outliers in the training data
/// - Multivariate anomaly detection
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Support fraction: 0.5 (use 50% of data for robust estimation)
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Rousseeuw, P.J., Van Driessen, K. (1999). "A Fast Algorithm for the Minimum
/// Covariance Determinant Estimator." Technometrics.
/// </para>
/// </remarks>
public class EllipticEnvelopeDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _supportFraction;
    private Vector<T>? _location;
    private Matrix<T>? _precisionMatrix;

    /// <summary>
    /// Gets the support fraction (proportion of data for robust estimation).
    /// </summary>
    public double SupportFraction => _supportFraction;

    /// <summary>
    /// Creates a new Elliptic Envelope anomaly detector.
    /// </summary>
    /// <param name="supportFraction">
    /// Proportion of data to use for robust estimation. Default is 0.5 (50%).
    /// Should be between 0.5 and 1.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public EllipticEnvelopeDetector(double supportFraction = 0.5, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (supportFraction < 0.5 || supportFraction > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(supportFraction),
                "SupportFraction must be between 0.5 and 1. Recommended is 0.5.");
        }

        _supportFraction = supportFraction;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        int d = X.Columns;

        if (n <= d)
        {
            throw new ArgumentException(
                $"Number of samples ({n}) must be greater than number of features ({d}).",
                nameof(X));
        }

        // Compute robust location and covariance using simplified MCD
        (_location, var covariance) = ComputeRobustEstimates(X);

        // Compute precision matrix (inverse of covariance)
        _precisionMatrix = ComputeInverse(covariance, d);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
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

        var scores = new Vector<T>(X.Rows);
        int d = X.Columns;

        for (int i = 0; i < X.Rows; i++)
        {
            // Compute Mahalanobis distance: sqrt((x - mu)' * Precision * (x - mu))
            var diff = new Vector<T>(d);
            for (int j = 0; j < d; j++)
            {
                diff[j] = NumOps.Subtract(X[i, j], _location![j]);
            }

            // temp = Precision * diff
            var temp = new Vector<T>(d);
            for (int j = 0; j < d; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < d; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_precisionMatrix![j, k], diff[k]));
                }
                temp[j] = sum;
            }

            // distance^2 = diff' * temp
            T distSquared = NumOps.Zero;
            for (int j = 0; j < d; j++)
            {
                distSquared = NumOps.Add(distSquared, NumOps.Multiply(diff[j], temp[j]));
            }

            scores[i] = NumOps.Sqrt(distSquared);
        }

        return scores;
    }

    private (Vector<T> location, Matrix<T> covariance) ComputeRobustEstimates(Matrix<T> X)
    {
        int n = X.Rows;
        int d = X.Columns;
        var random = new Random(_randomSeed);

        int h = (int)(n * _supportFraction);
        h = Math.Max(h, d + 1);

        // Start with median-based initial estimates
        var location = ComputeMedian(X);
        var covariance = ComputeCovariance(X, location);

        // Iterative refinement (C-steps)
        for (int iter = 0; iter < 10; iter++)
        {
            // Compute Mahalanobis distances
            var precision = ComputeInverse(covariance, d);
            var distances = new double[n];

            for (int i = 0; i < n; i++)
            {
                var diff = new Vector<T>(d);
                for (int j = 0; j < d; j++)
                {
                    diff[j] = NumOps.Subtract(X[i, j], location[j]);
                }

                var temp = new Vector<T>(d);
                for (int j = 0; j < d; j++)
                {
                    T sum = NumOps.Zero;
                    for (int k = 0; k < d; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(precision[j, k], diff[k]));
                    }
                    temp[j] = sum;
                }

                T distSquared = NumOps.Zero;
                for (int j = 0; j < d; j++)
                {
                    distSquared = NumOps.Add(distSquared, NumOps.Multiply(diff[j], temp[j]));
                }

                distances[i] = NumOps.ToDouble(distSquared);
            }

            // Select h points with smallest distances
            var indices = Enumerable.Range(0, n)
                .OrderBy(i => distances[i])
                .Take(h)
                .ToArray();

            // Recompute location and covariance from subset
            location = ComputeSubsetMean(X, indices);
            covariance = ComputeSubsetCovariance(X, location, indices);
        }

        return (location, covariance);
    }

    private Vector<T> ComputeMedian(Matrix<T> X)
    {
        int d = X.Columns;
        var median = new Vector<T>(d);

        for (int j = 0; j < d; j++)
        {
            var values = new List<double>();
            for (int i = 0; i < X.Rows; i++)
            {
                values.Add(NumOps.ToDouble(X[i, j]));
            }
            values.Sort();

            int mid = values.Count / 2;
            double medianValue = values.Count % 2 == 0
                ? (values[mid - 1] + values[mid]) / 2
                : values[mid];

            median[j] = NumOps.FromDouble(medianValue);
        }

        return median;
    }

    private Vector<T> ComputeSubsetMean(Matrix<T> X, int[] indices)
    {
        int d = X.Columns;
        var mean = new Vector<T>(d);

        foreach (int i in indices)
        {
            for (int j = 0; j < d; j++)
            {
                mean[j] = NumOps.Add(mean[j], X[i, j]);
            }
        }

        for (int j = 0; j < d; j++)
        {
            mean[j] = NumOps.Divide(mean[j], NumOps.FromDouble(indices.Length));
        }

        return mean;
    }

    private Matrix<T> ComputeCovariance(Matrix<T> X, Vector<T> mean)
    {
        int n = X.Rows;
        int d = X.Columns;
        var cov = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = i; j < d; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < n; k++)
                {
                    T diffI = NumOps.Subtract(X[k, i], mean[i]);
                    T diffJ = NumOps.Subtract(X[k, j], mean[j]);
                    sum = NumOps.Add(sum, NumOps.Multiply(diffI, diffJ));
                }
                cov[i, j] = NumOps.Divide(sum, NumOps.FromDouble(n - 1));
                cov[j, i] = cov[i, j];
            }
        }

        // Add regularization
        double epsilon = 1e-6;
        for (int i = 0; i < d; i++)
        {
            cov[i, i] = NumOps.Add(cov[i, i], NumOps.FromDouble(epsilon));
        }

        return cov;
    }

    private Matrix<T> ComputeSubsetCovariance(Matrix<T> X, Vector<T> mean, int[] indices)
    {
        int d = X.Columns;
        var cov = new Matrix<T>(d, d);

        for (int i = 0; i < d; i++)
        {
            for (int j = i; j < d; j++)
            {
                T sum = NumOps.Zero;
                foreach (int k in indices)
                {
                    T diffI = NumOps.Subtract(X[k, i], mean[i]);
                    T diffJ = NumOps.Subtract(X[k, j], mean[j]);
                    sum = NumOps.Add(sum, NumOps.Multiply(diffI, diffJ));
                }
                cov[i, j] = NumOps.Divide(sum, NumOps.FromDouble(indices.Length - 1));
                cov[j, i] = cov[i, j];
            }
        }

        // Add regularization
        double epsilon = 1e-6;
        for (int i = 0; i < d; i++)
        {
            cov[i, i] = NumOps.Add(cov[i, i], NumOps.FromDouble(epsilon));
        }

        return cov;
    }

    private Matrix<T> ComputeInverse(Matrix<T> matrix, int d)
    {
        var result = new Matrix<T>(d, d);
        var augmented = new Matrix<T>(d, 2 * d);

        // Create augmented matrix [A | I]
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                augmented[i, j] = matrix[i, j];
                augmented[i, j + d] = i == j ? NumOps.FromDouble(1) : NumOps.Zero;
            }
        }

        // Gauss-Jordan elimination
        for (int col = 0; col < d; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < d; row++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(augmented[row, col]), NumOps.Abs(augmented[maxRow, col])))
                {
                    maxRow = row;
                }
            }

            // Swap rows
            if (maxRow != col)
            {
                for (int j = 0; j < 2 * d; j++)
                {
                    T temp = augmented[col, j];
                    augmented[col, j] = augmented[maxRow, j];
                    augmented[maxRow, j] = temp;
                }
            }

            // Scale pivot row
            T pivot = augmented[col, col];
            if (NumOps.Equals(pivot, NumOps.Zero))
            {
                pivot = NumOps.FromDouble(1e-6);
            }

            for (int j = 0; j < 2 * d; j++)
            {
                augmented[col, j] = NumOps.Divide(augmented[col, j], pivot);
            }

            // Eliminate column
            for (int row = 0; row < d; row++)
            {
                if (row != col)
                {
                    T factor = augmented[row, col];
                    for (int j = 0; j < 2 * d; j++)
                    {
                        augmented[row, j] = NumOps.Subtract(augmented[row, j],
                            NumOps.Multiply(factor, augmented[col, j]));
                    }
                }
            }
        }

        // Extract inverse
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                result[i, j] = augmented[i, j + d];
            }
        }

        return result;
    }
}
