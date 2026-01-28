using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Statistical;

/// <summary>
/// Detects anomalies using the Chi-Square test for multivariate data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The Chi-Square (Mahalanobis distance) detector identifies outliers
/// by measuring how far a point is from the center of the data, accounting for correlations
/// between features. Points far from the center in Mahalanobis distance are anomalies.
/// </para>
/// <para>
/// The algorithm computes:
/// D^2 = (x - mean)' * Cov^(-1) * (x - mean)
/// This squared Mahalanobis distance follows a Chi-Square distribution with p degrees of freedom
/// (where p = number of features) under the assumption of multivariate normality.
/// </para>
/// <para>
/// <b>When to use:</b>
/// - Multivariate data (multiple correlated features)
/// - Data is approximately multivariate normal
/// - You want to account for correlations between features
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - Alpha (significance level): 0.05 (5%)
/// </para>
/// </remarks>
public class ChiSquareDetector<T> : AnomalyDetectorBase<T>
{
    private readonly double _alpha;
    private Vector<T>? _mean;
    private Matrix<T>? _covarianceInverse;
    private double _chiSquareCritical;

    /// <summary>
    /// Gets the significance level (alpha) for the test.
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// Creates a new Chi-Square anomaly detector.
    /// </summary>
    /// <param name="alpha">
    /// The significance level for the test. Default is 0.05 (5%).
    /// </param>
    /// <param name="contamination">
    /// The expected proportion of anomalies in the data. Default is 0.1 (10%).
    /// </param>
    /// <param name="randomSeed">
    /// Random seed for reproducibility. Default is 42.
    /// </param>
    public ChiSquareDetector(double alpha = 0.05, double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (alpha <= 0 || alpha >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha),
                "Alpha must be between 0 and 1. Recommended value is 0.05.");
        }

        _alpha = alpha;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Rows <= X.Columns)
        {
            throw new ArgumentException(
                $"Number of samples ({X.Rows}) must be greater than number of features ({X.Columns}) for covariance estimation.",
                nameof(X));
        }

        int n = X.Rows;
        int p = X.Columns;

        // Calculate mean
        _mean = new Vector<T>(p);
        for (int j = 0; j < p; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < n; i++)
            {
                sum = NumOps.Add(sum, X[i, j]);
            }
            _mean[j] = NumOps.Divide(sum, NumOps.FromDouble(n));
        }

        // Calculate covariance matrix
        var covariance = new Matrix<T>(p, p);
        for (int j1 = 0; j1 < p; j1++)
        {
            for (int j2 = j1; j2 < p; j2++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < n; i++)
                {
                    T diff1 = NumOps.Subtract(X[i, j1], _mean[j1]);
                    T diff2 = NumOps.Subtract(X[i, j2], _mean[j2]);
                    sum = NumOps.Add(sum, NumOps.Multiply(diff1, diff2));
                }
                T cov = NumOps.Divide(sum, NumOps.FromDouble(n - 1));
                covariance[j1, j2] = cov;
                covariance[j2, j1] = cov; // Symmetric
            }
        }

        // Compute inverse of covariance matrix (with regularization for stability)
        _covarianceInverse = ComputeInverse(covariance);

        // Get Chi-Square critical value (approximation)
        _chiSquareCritical = GetChiSquareCritical(p, 1 - _alpha);

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
        int p = X.Columns;

        for (int i = 0; i < X.Rows; i++)
        {
            // Compute Mahalanobis distance squared
            // D^2 = (x - mean)' * Cov^(-1) * (x - mean)
            var diff = new Vector<T>(p);
            for (int j = 0; j < p; j++)
            {
                diff[j] = NumOps.Subtract(X[i, j], _mean![j]);
            }

            // temp = Cov^(-1) * diff
            var temp = new Vector<T>(p);
            for (int j = 0; j < p; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < p; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(_covarianceInverse![j, k], diff[k]));
                }
                temp[j] = sum;
            }

            // D^2 = diff' * temp
            T mahalanobisSquared = NumOps.Zero;
            for (int j = 0; j < p; j++)
            {
                mahalanobisSquared = NumOps.Add(mahalanobisSquared, NumOps.Multiply(diff[j], temp[j]));
            }

            scores[i] = mahalanobisSquared;
        }

        return scores;
    }

    private Matrix<T> ComputeInverse(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var result = new Matrix<T>(n, n);
        var augmented = new Matrix<T>(n, 2 * n);

        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = matrix[i, j];
                augmented[i, j + n] = i == j ? NumOps.FromDouble(1) : NumOps.Zero;
            }
        }

        // Add small regularization for numerical stability
        double epsilon = 1e-6;
        for (int i = 0; i < n; i++)
        {
            augmented[i, i] = NumOps.Add(augmented[i, i], NumOps.FromDouble(epsilon));
        }

        // Gauss-Jordan elimination
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(augmented[row, col]), NumOps.Abs(augmented[maxRow, col])))
                {
                    maxRow = row;
                }
            }

            // Swap rows
            if (maxRow != col)
            {
                for (int j = 0; j < 2 * n; j++)
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
                // Matrix is singular, use pseudo-inverse approximation
                pivot = NumOps.FromDouble(epsilon);
            }

            for (int j = 0; j < 2 * n; j++)
            {
                augmented[col, j] = NumOps.Divide(augmented[col, j], pivot);
            }

            // Eliminate column
            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    T factor = augmented[row, col];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        augmented[row, j] = NumOps.Subtract(augmented[row, j],
                            NumOps.Multiply(factor, augmented[col, j]));
                    }
                }
            }
        }

        // Extract inverse
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = augmented[i, j + n];
            }
        }

        return result;
    }

    private double GetChiSquareCritical(int df, double p)
    {
        // Wilson-Hilferty approximation for Chi-Square quantile
        // X^2 ≈ df * (1 - 2/(9*df) + z * sqrt(2/(9*df)))^3
        double z = GetZCritical(1 - p);
        double term = 2.0 / (9 * df);
        double result = df * Math.Pow(1 - term + z * Math.Sqrt(term), 3);
        return Math.Max(0, result);
    }

    private double GetZCritical(double alpha)
    {
        // Approximation of inverse standard normal CDF
        double p = alpha;
        if (p > 0.5) p = 1 - p;

        double t = Math.Sqrt(-2 * Math.Log(p));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        double z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);

        return alpha > 0.5 ? -z : z;
    }
}
