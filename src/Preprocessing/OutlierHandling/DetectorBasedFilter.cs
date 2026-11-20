using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.Engines;

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
/// var filter2 = new DetectorBasedFilter&lt;double&gt;(
///     new ZScoreDetector&lt;double&gt;(),
///     FilterMode.Flag);
///
/// // Replace outliers with median using IQR detector
/// var filter3 = new DetectorBasedFilter&lt;double&gt;(
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
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DetectorBasedFilter<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly IAnomalyDetector<T> _detector;
    private readonly FilterMode _mode;

    // Fitted parameters for replacement modes
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _columnMedians;
    [AiDotNet.Attributes.FittedParameter]
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
        // The detector's Fit method handles re-initialization. It sees only the columns selected for
        // anomaly detection (all columns when none are given).
        _detector.Fit(DetectionInput(data));

        // Calculate statistics for replacement modes
        if (_mode == FilterMode.ReplaceWithMedian || _mode == FilterMode.ReplaceWithMean)
        {
            int nFeatures = data.Columns;
            if (data.Rows == 0 || nFeatures == 0)
            {
                _columnMedians = new Vector<T>(nFeatures);
                _columnMeans = new Vector<T>(nFeatures);
                return;
            }

            // Median of an even count is the average of the two middle values (linear interpolation).
            var x = Tensor<T>.FromMatrix(data);
            _columnMeans = ScalerTensorOps.ColumnMean(x).ToVector();
            _columnMedians = Engine.TensorQuantile(x, 0.5, 0, false, QuantileInterpolation.Linear).ToVector();
        }
    }

    /// <summary>The columns the detector works on: the selected columns, or all of them.</summary>
    private Matrix<T> DetectionInput(Matrix<T> data) =>
        ColumnIndices is null || ColumnIndices.Length == 0
            ? data
            : PreprocessingTensorOps.SelectColumns(data, GetColumnsToProcess(data.Columns));

    /// <summary>
    /// Transforms the data by handling anomalies according to the filter mode.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data with anomalies handled.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        // Get predictions from the detector (-1 = anomaly, 1 = normal)
        var predictions = _detector.Predict(DetectionInput(data));

        return _mode switch
        {
            FilterMode.Remove => RemoveAnomalies(data, predictions),
            FilterMode.Flag => FlagAnomalies(data, predictions),
            FilterMode.ReplaceWithMedian => ReplaceAnomalies(data, predictions, (_columnMedians ?? throw new InvalidOperationException("_columnMedians is null where it is required."))),
            FilterMode.ReplaceWithMean => ReplaceAnomalies(data, predictions, (_columnMeans ?? throw new InvalidOperationException("_columnMeans is null where it is required."))),
            _ => data
        };
    }

    private Matrix<T> RemoveAnomalies(Matrix<T> data, Vector<T> predictions)
    {
        // Keep the inlier rows (prediction 1) in order
        var inliers = PreprocessingTensorOps.MaskPositions(
            Engine.TensorEquals(Tensor<T>.FromVector(predictions), NumOps.One));
        if (inliers.Length == data.Rows)
        {
            return data; // No anomalies, return original
        }

        return PreprocessingTensorOps.SelectRows(data, inliers);
    }

    private Matrix<T> FlagAnomalies(Matrix<T> data, Vector<T> predictions)
    {
        // Add a new column with anomaly flag (1 = anomaly, 0 = normal)
        int rows = data.Rows;
        var flag = Engine.Reshape(AnomalyFlags(predictions), new[] { rows, 1 });
        if (data.Columns == 0)
        {
            return flag.ToMatrix();
        }

        return Engine.TensorConcatenate(new[] { Tensor<T>.FromMatrix(data), flag }, 1).ToMatrix();
    }

    /// <summary>1 where the prediction is -1 (anomaly), 0 elsewhere.</summary>
    private Tensor<T> AnomalyFlags(Vector<T> predictions) =>
        Engine.TensorEquals(Tensor<T>.FromVector(predictions), NumOps.FromDouble(-1));

    private Matrix<T> ReplaceAnomalies(Matrix<T> data, Vector<T> predictions, Vector<T> replacementValues)
    {
        int rows = data.Rows;
        int columns = data.Columns;
        if (rows == 0 || columns == 0)
        {
            return new Matrix<T>(rows, columns);
        }

        // Replace anomalous rows with the specified values (median or mean)
        var anomalous = Engine.TensorBroadcastTo(
            Engine.Reshape(AnomalyFlags(predictions), new[] { rows, 1 }), new[] { rows, columns });
        return Engine.TensorWhere(
            anomalous, ScalerTensorOps.BroadcastRow(replacementValues, rows), Tensor<T>.FromMatrix(data)).ToMatrix();
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

        if (data.Columns == 0)
        {
            throw new ArgumentException("Flagged data must contain the flag column.", nameof(data));
        }

        // Remove the last column (the flag column)
        if (data.Rows == 0 || data.Columns == 1)
        {
            return new Matrix<T>(data.Rows, data.Columns - 1);
        }

        return Engine.TensorSlice(Tensor<T>.FromMatrix(data), new[] { 0, 0 }, new[] { data.Rows, data.Columns - 1 })
            .ToMatrix();
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

        var predictions = _detector.Predict(DetectionInput(data));
        var mask = new bool[data.Rows];
        if (predictions.Length == 0)
        {
            return mask;
        }

        var flags = AnomalyFlags(predictions).Cast<int>().AsSpan();
        for (int i = 0; i < flags.Length && i < mask.Length; i++)
        {
            mask[i] = flags[i] != 0;
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
        return _detector.ScoreAnomalies(DetectionInput(data));
    }
}
