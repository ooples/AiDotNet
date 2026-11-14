using AiDotNet.Attributes;
using AiDotNet.Clustering.Base;
using AiDotNet.Enums;
using AiDotNet.Helpers;

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
[ComponentType(ComponentType.Evaluator)]
[PipelineStage(PipelineStage.Evaluation)]
public class MahalanobisDistance<T> : DistanceMetricBase<T>
{
    [AiDotNet.Attributes.FittedParameter]
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
        var diff = Engine.Subtract(a, b);

        // Compute Σ^(-1) × (a - b)
        var temp = Engine.MatrixVectorMultiply(_inverseCovarianceMatrix, diff);

        // Compute (a - b)^T × temp
        T result = Engine.DotProduct(diff, temp);

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

        // Compute mean and the (unbiased) covariance of the centered data: C = X_c^T X_c / (n - 1)
        var centered = ClusteringTensorOps.SubtractRow(data, ClusteringTensorOps.ColumnMeans(data));
        var scatter = Engine.TensorMatMul(Engine.TensorTranspose(centered), centered);
        var covariance = Engine.TensorDivideScalar(scatter, NumOps.FromDouble(data.Rows - 1));

        // Add small regularization for numerical stability
        var ridge = Engine.TensorMultiplyScalar(Engine.TensorEye<T>(data.Columns), NumOps.FromDouble(1e-6));

        _inverseCovarianceMatrix = InvertMatrix(Engine.TensorAdd(covariance, ridge).ToMatrix());
    }

    /// <inheritdoc />
    protected override Tensor<T> ComputeRowPairs(Tensor<T> a, Tensor<T> b)
    {
        if (_inverseCovarianceMatrix is null)
        {
            return ClusteringTensorOps.RowEuclidean(a, b);
        }

        int d = a.Shape[1];
        if (_inverseCovarianceMatrix.Rows != d || _inverseCovarianceMatrix.Columns != d)
        {
            throw new ArgumentException(
                $"Inverse covariance matrix dimensions ({_inverseCovarianceMatrix.Rows}x{_inverseCovarianceMatrix.Columns}) " +
                $"must match vector length ({d}).");
        }

        // Row i of diff * (S^-1)^T is S^-1 (a_i - b_i); its dot with (a_i - b_i) is the squared distance.
        var diff = Engine.TensorSubtract(a, b);
        var inverseT = Engine.TensorTranspose(Tensor<T>.FromMatrix(_inverseCovarianceMatrix));
        var quadratic = Engine.ReduceSum(Engine.TensorMultiply(Engine.TensorMatMul(diff, inverseT), diff), new[] { 1 }, false);
        return Engine.TensorSqrt(quadratic);
    }

    private T ComputeEuclidean(Vector<T> a, Vector<T> b)
    {
        return VectorHelper.EuclideanDistance(a, b);
    }

    private Matrix<T> InvertMatrix(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        var engine = Engine;

        // Augmented matrix [A | I]
        var augmented = engine.TensorConcatenate(
            new[] { Tensor<T>.FromMatrix(matrix), engine.TensorEye<T>(n) }, 1).ToMatrix();
        var ones = engine.Fill(n, NumOps.One);
        var tolerance = NumOps.FromDouble(1e-10);

        // Gauss-Jordan elimination with partial pivoting, one whole-matrix update per column
        for (int col = 0; col < n; col++)
        {
            // Pivot: the largest |entry| of this column at or below the diagonal (first one on ties)
            var below = engine.Abs(augmented.GetColumn(col).GetSubVector(col, n - col));
            int maxRow = col + below.IndexOfMax();
            if (maxRow != col)
            {
                engine.SwapRows(augmented, col, maxRow);
            }

            // Every candidate below the tolerance means the matrix is singular
            if (engine.TensorCountNonzero(engine.TensorLessThan(below.AsTensor(), tolerance)) == below.Length)
            {
                throw new InvalidOperationException("Matrix is singular and cannot be inverted.");
            }

            // Scale pivot row
            var pivotRow = augmented.GetRow(col);
            var scaledRow = engine.Divide(pivotRow, engine.Sum(pivotRow.GetSubVector(col, 1)));
            engine.SetRow(augmented, col, scaledRow);

            // Eliminate the column from every other row: A -= f * scaledRow^T with f[col] = 0
            var factors = engine.Multiply(
                augmented.GetColumn(col), engine.Subtract(ones, Vector<T>.CreateStandardBasis(n, col)));
            augmented = engine.MatrixSubtract(augmented, engine.OuterProduct(factors, scaledRow));
        }

        // Extract inverse matrix
        return augmented.GetSubMatrix(0, n, n, n);
    }
}
