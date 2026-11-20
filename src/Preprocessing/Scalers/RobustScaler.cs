using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales features using statistics that are robust to outliers.
/// </summary>
/// <remarks>
/// <para>
/// Robust scaling removes the median and scales data according to the interquartile range (IQR).
/// The IQR is the range between the 25th percentile (Q1) and 75th percentile (Q3).
/// Unlike StandardScaler, RobustScaler uses statistics that are less affected by outliers.
/// </para>
/// <para><b>For Beginners:</b> This scaler is like StandardScaler but better handles outliers:
/// - Uses median (middle value) instead of mean (average)
/// - Uses IQR (spread of middle 50%) instead of standard deviation
///
/// Why this matters:
/// - Mean and std are heavily influenced by extreme values
/// - Median and IQR ignore extreme values
///
/// Example: If most house prices are $100K-$500K but a few are $10M,
/// RobustScaler won't let those mansions distort the scaling.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Scaler)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RobustScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _median;
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _iqr;
    [JsonProperty]
    private readonly double _quantileRangeMin;
    [JsonProperty]
    private readonly double _quantileRangeMax;
    [JsonProperty]
    private readonly bool _withCentering;
    [JsonProperty]
    private readonly bool _withScaling;

    /// <summary>
    /// Gets the median of each feature computed during fitting.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? Median => _median;

    /// <summary>
    /// Gets the interquartile range (IQR) of each feature computed during fitting.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? InterquartileRange => _iqr;

    /// <summary>
    /// Gets whether this scaler centers the data (subtracts median).
    /// </summary>
    [JsonIgnore]
    public bool WithCentering => _withCentering;

    /// <summary>
    /// Gets whether this scaler scales the data (divides by IQR).
    /// </summary>
    [JsonIgnore]
    public bool WithScaling => _withScaling;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="RobustScaler{T}"/> with default settings.
    /// </summary>
    /// <param name="withCentering">If true, center the data by subtracting the median. Default is true.</param>
    /// <param name="withScaling">If true, scale the data by dividing by the IQR. Default is true.</param>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    [JsonConstructor]
    public RobustScaler(bool withCentering = true, bool withScaling = true, int[]? columnIndices = null)
        : this(25.0, 75.0, withCentering, withScaling, columnIndices)
    {
    }

    /// <summary>
    /// Creates a new instance of <see cref="RobustScaler{T}"/> with custom quantile range.
    /// </summary>
    /// <param name="quantileRangeMin">The lower quantile (0-100). Default is 25 (Q1).</param>
    /// <param name="quantileRangeMax">The upper quantile (0-100). Default is 75 (Q3).</param>
    /// <param name="withCentering">If true, center the data by subtracting the median.</param>
    /// <param name="withScaling">If true, scale the data by dividing by the IQR.</param>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public RobustScaler(
        double quantileRangeMin,
        double quantileRangeMax,
        bool withCentering = true,
        bool withScaling = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (quantileRangeMin < 0 || quantileRangeMin > 100)
        {
            throw new ArgumentOutOfRangeException(
                nameof(quantileRangeMin), "Quantile must be between 0 and 100.");
        }

        if (quantileRangeMax < 0 || quantileRangeMax > 100)
        {
            throw new ArgumentOutOfRangeException(
                nameof(quantileRangeMax), "Quantile must be between 0 and 100.");
        }

        if (quantileRangeMin >= quantileRangeMax)
        {
            throw new ArgumentException(
                $"Quantile range minimum ({quantileRangeMin}) must be less than maximum ({quantileRangeMax}).");
        }

        _quantileRangeMin = quantileRangeMin;
        _quantileRangeMax = quantileRangeMax;
        _withCentering = withCentering;
        _withScaling = withScaling;
    }

    /// <summary>
    /// Computes the median and IQR of each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        _median = ScalerTensorOps.FilledVector(numColumns, NumOps.Zero);
        _iqr = ScalerTensorOps.FilledVector(numColumns, NumOps.One);
        if (data.Rows == 0 || numColumns == 0)
        {
            return;
        }

        // scikit-learn: center_ = nanmedian, scale_ = nanpercentile(q_max) - nanpercentile(q_min) with linear
        // interpolation, and a zero (or sub-10-eps) range is treated as 1.
        var x = Tensor<T>.FromMatrix(data);
        if (_withCentering)
        {
            _median = ScalerTensorOps.MaskedParameters(mask, ColumnQuantile(x, 50.0), NumOps.Zero);
        }

        if (_withScaling)
        {
            var range = Engine.TensorSubtract(ColumnQuantile(x, _quantileRangeMax), ColumnQuantile(x, _quantileRangeMin));
            _iqr = ScalerTensorOps.MaskedParameters(mask, ScalerTensorOps.HandleZerosInScale(range), NumOps.One);
        }
    }

    /// <summary>
    /// Transforms the data by applying robust scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The scaled data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_median is null || _iqr is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        // Unprocessed columns have center 0 and scale 1, so (x - 0) / 1 leaves them exactly unchanged.
        var center = _withCentering ? _median : ScalerTensorOps.FilledVector(_median.Length, NumOps.Zero);
        var scale = _withScaling ? _iqr : ScalerTensorOps.FilledVector(_iqr.Length, NumOps.One);
        return ScalerTensorOps.ShiftThenScale(data, center, scale);
    }

    /// <summary>
    /// Reverses the robust scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_median is null || _iqr is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        var center = _withCentering ? _median : ScalerTensorOps.FilledVector(_median.Length, NumOps.Zero);
        var scale = _withScaling ? _iqr : ScalerTensorOps.FilledVector(_iqr.Length, NumOps.One);
        return ScalerTensorOps.ScaleThenShift(data, scale, center);
    }

    /// <summary>
    /// The <paramref name="percentile"/>-th percentile (0-100) of every column, ignoring NaN, with linear
    /// interpolation between order statistics (numpy <c>nanpercentile</c>), as [p].
    /// </summary>
    private Tensor<T> ColumnQuantile(Tensor<T> x, double percentile) =>
        Engine.TensorNanQuantile(x, percentile / 100.0, 0, false, QuantileInterpolation.Linear);

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (RobustScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
