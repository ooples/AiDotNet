namespace AiDotNet.Clustering.DistanceMetrics;

/// <summary>
/// Computes Mahalanobis distance between vectors.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Mahalanobis distance accounts for correlations between variables and scales
/// by variance. It's the standard metric used in Gaussian Mixture Models (GMM).
/// When the covariance matrix is the identity, it reduces to Euclidean distance.
/// </para>
/// <para>
/// Formula: d(a, b) = sqrt((a - b)^T × Σ^(-1) × (a - b))
/// where Σ is the covariance matrix.
/// </para>
/// <para><b>For Beginners:</b> Mahalanobis distance is "smart" about how features
/// relate to each other.
///
/// Example: If height and weight are correlated (tall people tend to weigh more),
/// Euclidean distance treats them as independent, but Mahalanobis distance
/// accounts for this relationship.
///
/// It's like asking "how many standard deviations away is this point?"
/// taking into account that the data might be stretched or tilted.
///
/// Best for:
/// - Detecting outliers
/// - Gaussian Mixture Models
/// - When features have different scales or correlations
/// </para>
/// </remarks>
public class MahalanobisDistance<T> : DistanceMetricBase<T>
{
    private Matrix<T>? _inverseCovarianceMatrix;

    /// <summary>
    /// Initializes a new instance without a covariance matrix.
    /// The covariance matrix must be set before computing distances.
    /// </summary>
    public MahalanobisDistance()
    {
    }

    /// <summary>
    /// Initializes a new instance with a precomputed inverse covariance matrix.
    /// </summary>
    /// <param name="inverseCovarianceMatrix">The inverse of the covariance matrix.</param>
    public MahalanobisDistance(Matrix<T> inverseCovarianceMatrix)
    {
        _inverseCovarianceMatrix = inverseCovarianceMatrix ??
            throw new ArgumentNullException(nameof(inverseCovarianceMatrix));
    }

    /// <summary>
    /// Gets or sets the inverse covariance matrix used for distance computation.
    /// </summary>
    public Matrix<T>? InverseCovarianceMatrix
    {
        get => _inverseCovarianceMatrix;
        set => _inverseCovarianceMatrix = value;
    }

    /// <inheritdoc />
    public override string Name => "Mahalanobis";

    /// <inheritdoc />
    public override T Compute(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException(
                $"Vectors must have the same length. Got {a.Length} and {b.Length}.");
        }

        if (_inverseCovarianceMatrix is null)
        {
            // Fall back to Euclidean (identity covariance)
            return ComputeEuclidean(a, b);
        }

        if (_inverseCovarianceMatrix.Rows != a.Length || _inverseCovarianceMatrix.Columns != a.Length)
        {
            throw new ArgumentException(
                $"Inverse covariance matrix dimensions ({_inverseCovarianceMatrix.Rows}x{_inverseCovarianceMatrix.Columns}) " +
                $"must match vector length ({a.Length}).");
        }

        // Compute difference vector: d = a - b
        var diff = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            diff[i] = NumOps.Subtract(a[i], b[i]);
        }

        // Compute Σ^(-1) × (a - b)
        var temp = MultiplyMatrixVector(_inverseCovarianceMatrix, diff);

        // Compute (a - b)^T × temp
        T result = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            result = NumOps.Add(result, NumOps.Multiply(diff[i], temp[i]));
        }

        return Sqrt(result);
    }

    /// <summary>
    /// Computes the Mahalanobis distance from data, estimating the covariance matrix.
    /// </summary>
    /// <param name="data">The data matrix to estimate covariance from.</param>
    /// <exception cref="ArgumentException">Thrown if data has fewer samples than features.</exception>
    public void FitFromData(Matrix<T> data)
    {
        if (data.Rows < data.Columns)
        {
            throw new ArgumentException(
                $"Need at least as many samples ({data.Rows}) as features ({data.Columns}) " +
                "to estimate covariance matrix.");
        }

        // Compute mean
        var mean = new Vector<T>(data.Columns);
        for (int j = 0; j < data.Columns; j++)
        {
            T sum = NumOps.Zero;
            for (int i = 0; i < data.Rows; i++)
            {
                sum = NumOps.Add(sum, data[i, j]);
            }
            mean[j] = NumOps.Divide(sum, NumOps.FromDouble(data.Rows));
        }

        // Compute covariance matrix
        var covariance = new Matrix<T>(data.Columns, data.Columns);
        for (int j = 0; j < data.Columns; j++)
        {
            for (int k = j; k < data.Columns; k++)
            {
                T sum = NumOps.Zero;
                for (int i = 0; i < data.Rows; i++)
                {
                    T diffJ = NumOps.Subtract(data[i, j], mean[j]);
                    T diffK = NumOps.Subtract(data[i, k], mean[k]);
                    sum = NumOps.Add(sum, NumOps.Multiply(diffJ, diffK));
                }
                T cov = NumOps.Divide(sum, NumOps.FromDouble(data.Rows - 1));
                covariance[j, k] = cov;
                covariance[k, j] = cov; // Symmetric
            }
        }

        // Add small regularization for numerical stability
        for (int i = 0; i < data.Columns; i++)
        {
            covariance[i, i] = NumOps.Add(covariance[i, i], NumOps.FromDouble(1e-6));
        }

        // Compute inverse using Cholesky decomposition for numerical stability
        _inverseCovarianceMatrix = InvertMatrix(covariance);
    }

    private T ComputeEuclidean(Vector<T> a, Vector<T> b)
    {
        T sumSquared = NumOps.Zero;
        for (int i = 0; i < a.Length; i++)
        {
            T diff = NumOps.Subtract(a[i], b[i]);
            sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(diff, diff));
        }
        return Sqrt(sumSquared);
    }

    private Vector<T> MultiplyMatrixVector(Matrix<T> matrix, Vector<T> vector)
    {
        var result = new Vector<T>(matrix.Rows);
        for (int i = 0; i < matrix.Rows; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < matrix.Columns; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(matrix[i, j], vector[j]));
            }
            result[i] = sum;
        }
        return result;
    }

    private Matrix<T> InvertMatrix(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var augmented = new Matrix<T>(n, 2 * n);

        // Create augmented matrix [A | I]
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = matrix[i, j];
                augmented[i, j + n] = i == j ? NumOps.One : NumOps.Zero;
            }
        }

        // Gauss-Jordan elimination with partial pivoting
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            double maxVal = Math.Abs(NumOps.ToDouble(augmented[col, col]));
            for (int row = col + 1; row < n; row++)
            {
                double val = Math.Abs(NumOps.ToDouble(augmented[row, col]));
                if (val > maxVal)
                {
                    maxVal = val;
                    maxRow = row;
                }
            }

            // Swap rows if necessary
            if (maxRow != col)
            {
                for (int j = 0; j < 2 * n; j++)
                {
                    T temp = augmented[col, j];
                    augmented[col, j] = augmented[maxRow, j];
                    augmented[maxRow, j] = temp;
                }
            }

            // Check for singular matrix
            if (Math.Abs(NumOps.ToDouble(augmented[col, col])) < 1e-10)
            {
                throw new InvalidOperationException("Matrix is singular and cannot be inverted.");
            }

            // Scale pivot row
            T pivot = augmented[col, col];
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
                        augmented[row, j] = NumOps.Subtract(
                            augmented[row, j],
                            NumOps.Multiply(factor, augmented[col, j]));
                    }
                }
            }
        }

        // Extract inverse matrix
        var inverse = new Matrix<T>(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inverse[i, j] = augmented[i, j + n];
            }
        }

        return inverse;
    }
}
