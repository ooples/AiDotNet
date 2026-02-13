using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Specifies how anomalies should be handled during preprocessing.
/// </summary>
public enum FilterMode
{
    /// <summary>
    /// Remove rows identified as anomalies. Reduces dataset size.
    /// </summary>
    Remove,

    /// <summary>
    /// Add a binary column indicating anomaly status (1 = anomaly, 0 = normal).
    /// Preserves all data and adds an indicator feature.
    /// </summary>
    Flag,

    /// <summary>
    /// Replace anomalous values with the column median.
    /// Preserves dataset size while reducing outlier impact.
    /// </summary>
    ReplaceWithMedian,

    /// <summary>
    /// Replace anomalous values with the column mean.
    /// Preserves dataset size while reducing outlier impact.
    /// </summary>
    ReplaceWithMean
}

/// <summary>
/// Wraps any <see cref="IAnomalyDetector{T}"/> for use in a preprocessing pipeline.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class bridges the gap between anomaly detection algorithms
/// and data preprocessing pipelines. It lets you use any anomaly detector to clean
/// your data before training a model.
/// </para>
/// <para>
/// <b>Usage Examples:</b>
/// <code>
/// // Remove outliers using Isolation Forest
/// var filter = new DetectorBasedFilter&lt;double&gt;(
///     new IsolationForest&lt;double&gt;(),
///     FilterMode.Remove);
///
/// // Flag outliers using Z-Score (adds a column)
/// var filter = new DetectorBasedFilter&lt;double&gt;(
///     new ZScoreDetector&lt;double&gt;(),
///     FilterMode.Flag);
///
/// // Replace outliers with median using IQR detector
/// var filter = new DetectorBasedFilter&lt;double&gt;(
///     new IQRDetector&lt;double&gt;(),
///     FilterMode.ReplaceWithMedian);
/// </code>
/// </para>
/// <para>
/// <b>Integration with PreprocessingPipeline:</b>
/// <code>
/// var pipeline = new PreprocessingPipeline&lt;double&gt;()
///     .Add(new DetectorBasedFilter&lt;double&gt;(new IsolationForest&lt;double&gt;(), FilterMode.Remove))
///     .Add(new StandardScaler&lt;double&gt;());
/// </code>
/// </para>
/// </remarks>
public class DetectorBasedFilter<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly IAnomalyDetector<T> _detector;
    private readonly FilterMode _mode;

    // Fitted parameters for replacement modes
    private Vector<T>? _columnMedians;
    private Vector<T>? _columnMeans;

    /// <summary>
    /// Gets the underlying anomaly detector.
    /// </summary>
    public IAnomalyDetector<T> Detector => _detector;

    /// <summary>
    /// Gets the filter mode (how anomalies are handled).
    /// </summary>
    public FilterMode Mode => _mode;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// Inverse transformation is only supported for Flag mode (by removing the flag column).
    /// Remove and Replace modes cannot be inverted.
    /// </remarks>
    public override bool SupportsInverseTransform => _mode == FilterMode.Flag;

    /// <summary>
    /// Creates a new detector-based filter for preprocessing.
    /// </summary>
    /// <param name="detector">
    /// The anomaly detector to use. Can be any implementation of <see cref="IAnomalyDetector{T}"/>
    /// such as IsolationForest, LocalOutlierFactor, ZScoreDetector, IQRDetector, etc.
    /// </param>
    /// <param name="mode">
    /// How to handle detected anomalies. Default is Remove (removes anomalous rows).
    /// </param>
    /// <param name="columnIndices">
    /// The column indices to consider for anomaly detection, or null for all columns.
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when detector is null.</exception>
    public DetectorBasedFilter(
        IAnomalyDetector<T> detector,
        FilterMode mode = FilterMode.Remove,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        Guard.NotNull(detector);
        _detector = detector;
        _mode = mode;
    }

    /// <summary>
    /// Fits the anomaly detector to the training data.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        // Always refit the detector to avoid stale models when data changes
        // The detector's Fit method handles re-initialization
        _detector.Fit(data);

        // Calculate statistics for replacement modes
        if (_mode == FilterMode.ReplaceWithMedian || _mode == FilterMode.ReplaceWithMean)
        {
            int nFeatures = data.Columns;
            _columnMedians = new Vector<T>(nFeatures);
            _columnMeans = new Vector<T>(nFeatures);

            for (int j = 0; j < nFeatures; j++)
            {
                var column = data.GetColumn(j);
                var (mean, _) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(column);
                _columnMeans[j] = mean;
                _columnMedians[j] = CalculateMedian(column);
            }
        }
    }

    private T CalculateMedian(Vector<T> values)
    {
        var sorted = values.ToArray();
        Array.Sort(sorted, (a, b) => NumOps.LessThan(a, b) ? -1 : (NumOps.GreaterThan(a, b) ? 1 : 0));

        int n = sorted.Length;
        if (n == 0)
        {
            return NumOps.Zero;
        }

        if (n % 2 == 0)
        {
            return NumOps.Divide(
                NumOps.Add(sorted[n / 2 - 1], sorted[n / 2]),
                NumOps.FromDouble(2));
        }
        else
        {
            return sorted[n / 2];
        }
    }

    /// <summary>
    /// Transforms the data by handling anomalies according to the filter mode.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data with anomalies handled.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        // Get predictions from the detector (-1 = anomaly, 1 = normal)
        var predictions = _detector.Predict(data);

        return _mode switch
        {
            FilterMode.Remove => RemoveAnomalies(data, predictions),
            FilterMode.Flag => FlagAnomalies(data, predictions),
            FilterMode.ReplaceWithMedian => ReplaceAnomalies(data, predictions, _columnMedians!),
            FilterMode.ReplaceWithMean => ReplaceAnomalies(data, predictions, _columnMeans!),
            _ => data
        };
    }

    private Matrix<T> RemoveAnomalies(Matrix<T> data, Vector<T> predictions)
    {
        // Count inliers
        int inlierCount = 0;
        T one = NumOps.FromDouble(1);
        for (int i = 0; i < predictions.Length; i++)
        {
            if (NumOps.Equals(predictions[i], one))
            {
                inlierCount++;
            }
        }

        if (inlierCount == data.Rows)
        {
            return data; // No anomalies, return original
        }

        // Create new matrix with only inliers
        var result = new T[inlierCount, data.Columns];
        int rowIdx = 0;

        for (int i = 0; i < data.Rows; i++)
        {
            if (NumOps.Equals(predictions[i], one))
            {
                for (int j = 0; j < data.Columns; j++)
                {
                    result[rowIdx, j] = data[i, j];
                }
                rowIdx++;
            }
        }

        return new Matrix<T>(result);
    }

    private Matrix<T> FlagAnomalies(Matrix<T> data, Vector<T> predictions)
    {
        // Add a new column with anomaly flag (1 = anomaly, 0 = normal)
        var result = new T[data.Rows, data.Columns + 1];
        T minusOne = NumOps.FromDouble(-1);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns; j++)
            {
                result[i, j] = data[i, j];
            }

            // Add flag column: 1 if anomaly (-1 prediction), 0 if normal (1 prediction)
            result[i, data.Columns] = NumOps.Equals(predictions[i], minusOne)
                ? NumOps.FromDouble(1)
                : NumOps.FromDouble(0);
        }

        return new Matrix<T>(result);
    }

    private Matrix<T> ReplaceAnomalies(Matrix<T> data, Vector<T> predictions, Vector<T> replacementValues)
    {
        var result = new T[data.Rows, data.Columns];
        T minusOne = NumOps.FromDouble(-1);

        for (int i = 0; i < data.Rows; i++)
        {
            bool isAnomaly = NumOps.Equals(predictions[i], minusOne);

            for (int j = 0; j < data.Columns; j++)
            {
                if (isAnomaly)
                {
                    // Replace with the specified value (median or mean)
                    result[i, j] = replacementValues[j];
                }
                else
                {
                    result[i, j] = data[i, j];
                }
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transforms flagged data by removing the flag column.
    /// Only supported when Mode is Flag.
    /// </summary>
    /// <param name="data">The flagged data.</param>
    /// <returns>The data without the flag column.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mode != FilterMode.Flag)
        {
            throw new NotSupportedException(
                $"Inverse transformation is only supported for FilterMode.Flag, not {_mode}.");
        }

        // Remove the last column (the flag column)
        var result = new T[data.Rows, data.Columns - 1];

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < data.Columns - 1; j++)
            {
                result[i, j] = data[i, j];
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The output feature names (with optional anomaly flag column).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_mode == FilterMode.Flag)
        {
            // Add anomaly flag column name
            var names = inputFeatureNames ?? Array.Empty<string>();
            var result = new string[names.Length + 1];
            names.CopyTo(result, 0);
            result[names.Length] = "is_anomaly";
            return result;
        }

        return inputFeatureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets a mask indicating which rows were identified as anomalies.
    /// </summary>
    /// <param name="data">The data to check.</param>
    /// <returns>A boolean array where true indicates an anomaly.</returns>
    public bool[] GetAnomalyMask(Matrix<T> data)
    {
        EnsureFitted();

        var predictions = _detector.Predict(data);
        var mask = new bool[data.Rows];
        T minusOne = NumOps.FromDouble(-1);

        for (int i = 0; i < predictions.Length; i++)
        {
            mask[i] = NumOps.Equals(predictions[i], minusOne);
        }

        return mask;
    }

    /// <summary>
    /// Gets the anomaly scores for each row in the data.
    /// </summary>
    /// <param name="data">The data to score.</param>
    /// <returns>Anomaly scores where higher values indicate more anomalous rows.</returns>
    public Vector<T> GetAnomalyScores(Matrix<T> data)
    {
        EnsureFitted();
        return _detector.ScoreAnomalies(data);
    }
}
