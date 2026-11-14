using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
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
        Guard.NotNull(detector);
        _detector = detector;
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
            OutlierHandlingMode.ReplaceWithMedian => ReplaceOutliers(X, y, predictions, (_columnMedians ?? throw new InvalidOperationException("_columnMedians is null where it is required."))),
            OutlierHandlingMode.ReplaceWithMean => ReplaceOutliers(X, y, predictions, (_columnMeans ?? throw new InvalidOperationException("_columnMeans is null where it is required."))),
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

        // Inliers are the samples predicted as 1
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var inliers = PreprocessingTensorOps.MaskPositions(
            engine.TensorEquals(Tensor<T>.FromVector(predictions), NumOps.FromDouble(1)));
        int inlierCount = inliers.Length;

        // If no outliers, return original data
        if (inlierCount == nSamples)
        {
            _isFitted = true;
            return (X, y);
        }

        // Create new tensors with only inliers
        Tensor<T> newX;
        Tensor<T> newY;
        if (inlierCount == 0)
        {
            int[] newXShape = (int[])X._shape;
            newXShape[0] = 0;
            newX = new Tensor<T>(newXShape);
            int[] newYShape = (int[])y._shape;
            newYShape[0] = 0;
            newY = new Tensor<T>(newYShape);
        }
        else
        {
            var inlierIndex = new Tensor<int>(inliers, new[] { inlierCount });
            newX = engine.TensorIndexSelect(X, inlierIndex, 0);
            newY = engine.TensorIndexSelect(y, inlierIndex, 0);
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

    private void CalculateStatistics(Matrix<T> X)
    {
        int n = X.Rows;
        int nFeatures = X.Columns;
        if (n == 0)
        {
            _columnMeans = new Vector<T>(nFeatures);
            _columnMedians = new Vector<T>(nFeatures);
            return;
        }

        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var x = Tensor<T>.FromMatrix(X);
        _columnMeans = engine.ReduceMean(x, new[] { 0 }, false).ToVector();

        // Column medians: the middle order statistic, or the mean of the two middle ones for even counts
        var sorted = engine.TensorSort(x, 0).Item1;
        var upperMiddle = engine.TensorSlice(sorted, new[] { n / 2, 0 }, new[] { 1, nFeatures });
        var median = n % 2 == 0
            ? engine.TensorDivideScalar(
                engine.TensorAdd(engine.TensorSlice(sorted, new[] { n / 2 - 1, 0 }, new[] { 1, nFeatures }), upperMiddle),
                NumOps.FromDouble(2))
            : upperMiddle;
        _columnMedians = engine.Reshape(median, new[] { nFeatures }).ToVector();
    }

    private (Matrix<T> X, Vector<T> y) RemoveOutliers(Matrix<T> X, Vector<T> y, Vector<T> predictions)
    {
        // Inliers are the samples predicted as 1
        var inliers = PreprocessingTensorOps.MaskPositions(
            AiDotNet.Tensors.Engines.AiDotNetEngine.Current.TensorEquals(Tensor<T>.FromVector(predictions), NumOps.FromDouble(1)));

        // If no outliers, return original data
        if (inliers.Length == X.Rows)
        {
            return (X, y);
        }

        // Keep only inliers
        return (PreprocessingTensorOps.SelectRows(X, inliers), PreprocessingTensorOps.SelectElements(y, inliers));
    }

    private (Matrix<T> X, Vector<T> y) ReplaceOutliers(
        Matrix<T> X, Vector<T> y, Vector<T> predictions, Vector<T> replacementValues)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        int rows = X.Rows;
        int columns = X.Columns;
        if (rows == 0 || columns == 0)
            return (new Matrix<T>(rows, columns), y);

        // Outlier rows (predicted -1) take the replacement values
        var shape = new[] { rows, columns };
        var isOutlier = engine.TensorEquals(Tensor<T>.FromVector(predictions), NumOps.FromDouble(-1));
        var newX = engine.TensorWhere(
            engine.TensorBroadcastTo(engine.Reshape(isOutlier, new[] { rows, 1 }), shape),
            engine.TensorBroadcastTo(engine.Reshape(Tensor<T>.FromVector(replacementValues), new[] { 1, columns }), shape),
            Tensor<T>.FromMatrix(X));

        // y is unchanged for replacement modes
        return (newX.ToMatrix(), y);
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
        var isOutlier = AiDotNet.Tensors.Engines.AiDotNetEngine.Current
            .TensorEquals(Tensor<T>.FromVector(predictions), NumOps.FromDouble(-1)).Cast<int>().AsSpan().ToArray();

        for (int i = 0; i < isOutlier.Length; i++)
        {
            mask[i] = isOutlier[i] != 0;
        }

        return mask;
    }
}
