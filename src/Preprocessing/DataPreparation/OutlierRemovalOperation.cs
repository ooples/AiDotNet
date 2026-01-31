using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation;

/// <summary>
/// Specifies how outliers should be handled during data preparation.
/// </summary>
public enum OutlierHandlingMode
{
    /// <summary>
    /// Remove rows identified as outliers. Reduces dataset size.
    /// </summary>
    Remove,

    /// <summary>
    /// Replace outlier values with the column median.
    /// Preserves dataset size while reducing outlier impact.
    /// </summary>
    ReplaceWithMedian,

    /// <summary>
    /// Replace outlier values with the column mean.
    /// Preserves dataset size while reducing outlier impact.
    /// </summary>
    ReplaceWithMean
}

/// <summary>
/// A row operation that removes or handles outliers using any anomaly detector.
/// </summary>
/// <remarks>
/// <para>
/// This operation wraps any <see cref="IAnomalyDetector{T}"/> to identify outliers and
/// either remove them or replace their values. When using Remove mode, both features (X)
/// and labels (y) are modified together to maintain alignment.
/// </para>
/// <para>
/// <b>For Beginners:</b> Outliers are unusual data points that don't follow the pattern
/// of most of your data. They can confuse machine learning models and lead to poor
/// predictions. This operation identifies outliers using statistical methods and either:
/// - Removes them entirely (reduces your dataset size)
/// - Replaces their values with typical values (median or mean)
/// </para>
/// <para>
/// <b>When to Use Each Mode:</b>
/// - <b>Remove:</b> When you have plenty of data and outliers are likely errors
/// - <b>ReplaceWithMedian:</b> When outliers are extreme but you need to preserve sample count
/// - <b>ReplaceWithMean:</b> Similar to median, but more affected by other outliers
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class OutlierRemovalOperation<T> : IRowOperation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IAnomalyDetector<T> _detector;
    private readonly OutlierHandlingMode _mode;

    // Fitted statistics for replacement modes
    private Vector<T>? _columnMedians;
    private Vector<T>? _columnMeans;
    private bool _isFitted;

    /// <inheritdoc/>
    public bool IsFitted => _isFitted;

    /// <inheritdoc/>
    public string Description => $"Outlier handling ({_mode}) using {_detector.GetType().Name}";

    /// <summary>
    /// Gets the underlying anomaly detector.
    /// </summary>
    public IAnomalyDetector<T> Detector => _detector;

    /// <summary>
    /// Gets the outlier handling mode.
    /// </summary>
    public OutlierHandlingMode Mode => _mode;

    /// <summary>
    /// Creates a new outlier removal operation.
    /// </summary>
    /// <param name="detector">
    /// The anomaly detector to use for identifying outliers.
    /// Can be any implementation such as IsolationForest, ZScoreDetector, IQRDetector, etc.
    /// </param>
    /// <param name="mode">
    /// How to handle detected outliers. Default is Remove.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when detector is null.</exception>
    public OutlierRemovalOperation(
        IAnomalyDetector<T> detector,
        OutlierHandlingMode mode = OutlierHandlingMode.Remove)
    {
        _detector = detector ?? throw new ArgumentNullException(nameof(detector));
        _mode = mode;
    }

    /// <inheritdoc/>
    public (Matrix<T> X, Vector<T> y) FitResample(Matrix<T> X, Vector<T> y)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (y is null) throw new ArgumentNullException(nameof(y));

        if (X.Rows != y.Length)
        {
            throw new ArgumentException(
                $"X has {X.Rows} rows but y has {y.Length} elements. They must match.",
                nameof(y));
        }

        // Fit the detector if not already fitted
        if (!_detector.IsFitted)
        {
            _detector.Fit(X);
        }

        // Get predictions: 1 = normal, -1 = anomaly
        var predictions = _detector.Predict(X);

        // Calculate statistics for replacement modes
        if (_mode == OutlierHandlingMode.ReplaceWithMedian || _mode == OutlierHandlingMode.ReplaceWithMean)
        {
            CalculateStatistics(X);
        }

        // Apply the appropriate handling mode
        var result = _mode switch
        {
            OutlierHandlingMode.Remove => RemoveOutliers(X, y, predictions),
            OutlierHandlingMode.ReplaceWithMedian => ReplaceOutliers(X, y, predictions, _columnMedians!),
            OutlierHandlingMode.ReplaceWithMean => ReplaceOutliers(X, y, predictions, _columnMeans!),
            _ => (X, y)
        };

        _isFitted = true;
        return result;
    }

    /// <inheritdoc/>
    public (Tensor<T> X, Tensor<T> y) FitResampleTensor(Tensor<T> X, Tensor<T> y)
    {
        if (X is null) throw new ArgumentNullException(nameof(X));
        if (y is null) throw new ArgumentNullException(nameof(y));

        if (X.Shape[0] != y.Shape[0])
        {
            throw new ArgumentException(
                $"X has {X.Shape[0]} samples but y has {y.Shape[0]} samples. They must match.",
                nameof(y));
        }

        // Convert tensor to matrix for outlier detection (flatten non-batch dimensions)
        int nSamples = X.Shape[0];
        int nFeatures = 1;
        for (int i = 1; i < X.Rank; i++)
        {
            nFeatures *= X.Shape[i];
        }

        var matrixX = new Matrix<T>(nSamples, nFeatures);
        for (int i = 0; i < nSamples; i++)
        {
            int flatIdx = 0;
            FlattenSampleToRow(X, i, matrixX, i, ref flatIdx);
        }

        // Fit the detector if not already fitted
        if (!_detector.IsFitted)
        {
            _detector.Fit(matrixX);
        }

        // Get predictions: 1 = normal, -1 = anomaly
        var predictions = _detector.Predict(matrixX);

        // For tensor data, only Remove mode is supported (replacement doesn't make sense for complex structures)
        if (_mode != OutlierHandlingMode.Remove)
        {
            throw new NotSupportedException(
                $"Outlier handling mode '{_mode}' is not supported for tensor data. Only 'Remove' mode is available.");
        }

        // Count inliers
        T one = NumOps.FromDouble(1);
        int inlierCount = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            if (NumOps.Equals(predictions[i], one))
            {
                inlierCount++;
            }
        }

        // If no outliers, return original data
        if (inlierCount == nSamples)
        {
            _isFitted = true;
            return (X, y);
        }

        // Create new tensors with only inliers
        int[] newXShape = (int[])X.Shape.Clone();
        newXShape[0] = inlierCount;
        var newX = new Tensor<T>(newXShape);

        int[] newYShape = (int[])y.Shape.Clone();
        newYShape[0] = inlierCount;
        var newY = new Tensor<T>(newYShape);

        int destIdx = 0;
        for (int i = 0; i < nSamples; i++)
        {
            if (NumOps.Equals(predictions[i], one))
            {
                CopyTensorSample(X, newX, i, destIdx);
                CopyTensorSample(y, newY, i, destIdx);
                destIdx++;
            }
        }

        _isFitted = true;
        return (newX, newY);
    }

    private void FlattenSampleToRow(Tensor<T> tensor, int sampleIdx, Matrix<T> matrix, int rowIdx, ref int flatIdx)
    {
        if (tensor.Rank == 1)
        {
            matrix[rowIdx, flatIdx++] = tensor[sampleIdx];
            return;
        }

        int[] indices = new int[tensor.Rank];
        indices[0] = sampleIdx;
        FlattenRecursive(tensor, matrix, rowIdx, indices, 1, ref flatIdx);
    }

    private void FlattenRecursive(Tensor<T> tensor, Matrix<T> matrix, int rowIdx, int[] indices, int dim, ref int flatIdx)
    {
        if (dim == tensor.Rank)
        {
            matrix[rowIdx, flatIdx++] = tensor[indices];
            return;
        }

        for (int i = 0; i < tensor.Shape[dim]; i++)
        {
            indices[dim] = i;
            FlattenRecursive(tensor, matrix, rowIdx, indices, dim + 1, ref flatIdx);
        }
    }

    private void CopyTensorSample(Tensor<T> source, Tensor<T> dest, int srcIdx, int destIdx)
    {
        if (source.Rank == 1)
        {
            dest[destIdx] = source[srcIdx];
            return;
        }

        int[] srcIndices = new int[source.Rank];
        int[] destIndices = new int[dest.Rank];
        srcIndices[0] = srcIdx;
        destIndices[0] = destIdx;
        CopyRecursive(source, dest, srcIndices, destIndices, 1);
    }

    private void CopyRecursive(Tensor<T> source, Tensor<T> dest, int[] srcIndices, int[] destIndices, int dim)
    {
        if (dim == source.Rank)
        {
            dest[destIndices] = source[srcIndices];
            return;
        }

        for (int i = 0; i < source.Shape[dim]; i++)
        {
            srcIndices[dim] = i;
            destIndices[dim] = i;
            CopyRecursive(source, dest, srcIndices, destIndices, dim + 1);
        }
    }

    private void CalculateStatistics(Matrix<T> X)
    {
        int nFeatures = X.Columns;
        _columnMedians = new Vector<T>(nFeatures);
        _columnMeans = new Vector<T>(nFeatures);

        for (int j = 0; j < nFeatures; j++)
        {
            var column = X.GetColumn(j);
            var (mean, _) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(column);
            _columnMeans[j] = mean;
            _columnMedians[j] = CalculateMedian(column);
        }
    }

    private T CalculateMedian(Vector<T> values)
    {
        var sorted = values.ToArray();
        Array.Sort(sorted, (a, b) => NumOps.LessThan(a, b) ? -1 : (NumOps.GreaterThan(a, b) ? 1 : 0));

        int n = sorted.Length;
        if (n == 0) return NumOps.Zero;

        if (n % 2 == 0)
        {
            return NumOps.Divide(
                NumOps.Add(sorted[n / 2 - 1], sorted[n / 2]),
                NumOps.FromDouble(2));
        }
        return sorted[n / 2];
    }

    private (Matrix<T> X, Vector<T> y) RemoveOutliers(Matrix<T> X, Vector<T> y, Vector<T> predictions)
    {
        T one = NumOps.FromDouble(1);

        // Count inliers (predictions == 1)
        int inlierCount = 0;
        for (int i = 0; i < predictions.Length; i++)
        {
            if (NumOps.Equals(predictions[i], one))
            {
                inlierCount++;
            }
        }

        // If no outliers, return original data
        if (inlierCount == X.Rows)
        {
            return (X, y);
        }

        // Create new arrays with only inliers
        var newX = new T[inlierCount, X.Columns];
        var newY = new T[inlierCount];
        int rowIdx = 0;

        for (int i = 0; i < X.Rows; i++)
        {
            if (NumOps.Equals(predictions[i], one))
            {
                for (int j = 0; j < X.Columns; j++)
                {
                    newX[rowIdx, j] = X[i, j];
                }
                newY[rowIdx] = y[i];
                rowIdx++;
            }
        }

        return (new Matrix<T>(newX), new Vector<T>(newY));
    }

    private (Matrix<T> X, Vector<T> y) ReplaceOutliers(
        Matrix<T> X, Vector<T> y, Vector<T> predictions, Vector<T> replacementValues)
    {
        T minusOne = NumOps.FromDouble(-1);

        var newX = new T[X.Rows, X.Columns];

        for (int i = 0; i < X.Rows; i++)
        {
            bool isOutlier = NumOps.Equals(predictions[i], minusOne);

            for (int j = 0; j < X.Columns; j++)
            {
                newX[i, j] = isOutlier ? replacementValues[j] : X[i, j];
            }
        }

        // y is unchanged for replacement modes
        return (new Matrix<T>(newX), y);
    }

    /// <summary>
    /// Gets a mask indicating which rows were identified as outliers.
    /// </summary>
    /// <param name="X">The data to check.</param>
    /// <returns>A boolean array where true indicates an outlier.</returns>
    public bool[] GetOutlierMask(Matrix<T> X)
    {
        if (!_detector.IsFitted)
        {
            throw new InvalidOperationException("Detector has not been fitted. Call FitResample first.");
        }

        var predictions = _detector.Predict(X);
        var mask = new bool[X.Rows];
        T minusOne = NumOps.FromDouble(-1);

        for (int i = 0; i < predictions.Length; i++)
        {
            mask[i] = NumOps.Equals(predictions[i], minusOne);
        }

        return mask;
    }
}
