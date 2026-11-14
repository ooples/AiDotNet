using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales features to a given range, typically [0, 1].
/// </summary>
/// <remarks>
/// <para>
/// Min-max scaling transforms features by scaling each feature to a given range.
/// The default range is [0, 1]. The transformation is: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
/// </para>
/// <para><b>For Beginners:</b> This scaler squishes all your data into a specific range:
/// - The smallest value becomes the minimum of the range (default 0)
/// - The largest value becomes the maximum of the range (default 1)
/// - Everything else is proportionally scaled in between
///
/// This is useful when:
/// - Your algorithm requires data in a specific range
/// - You want to preserve the relationships between values
/// - You don't want outliers to heavily influence the scaling
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Scaler)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MinMaxScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _dataMin;
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _dataMax;
    [JsonProperty]
    private readonly T _featureRangeMin;
    [JsonProperty]
    private readonly T _featureRangeMax;

    /// <summary>
    /// Gets the minimum value of each feature computed during fitting.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? DataMin => _dataMin;

    /// <summary>
    /// Gets the maximum value of each feature computed during fitting.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? DataMax => _dataMax;

    /// <summary>
    /// Gets the minimum value of the target feature range.
    /// </summary>
    [JsonIgnore]
    public T FeatureRangeMin => _featureRangeMin;

    /// <summary>
    /// Gets the maximum value of the target feature range.
    /// </summary>
    [JsonIgnore]
    public T FeatureRangeMax => _featureRangeMax;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="MinMaxScaler{T}"/> with default range [0, 1].
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    [JsonConstructor]
    public MinMaxScaler(int[]? columnIndices = null)
        : this(0.0, 1.0, columnIndices)
    {
    }

    /// <summary>
    /// Creates a new instance of <see cref="MinMaxScaler{T}"/> with a custom range.
    /// </summary>
    /// <param name="featureRangeMin">The minimum value of the target range.</param>
    /// <param name="featureRangeMax">The maximum value of the target range.</param>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public MinMaxScaler(double featureRangeMin, double featureRangeMax, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (featureRangeMin >= featureRangeMax)
        {
            throw new ArgumentException(
                $"Feature range minimum ({featureRangeMin}) must be less than maximum ({featureRangeMax}).");
        }

        _featureRangeMin = NumOps.FromDouble(featureRangeMin);
        _featureRangeMax = NumOps.FromDouble(featureRangeMax);
    }

    /// <summary>
    /// Computes the minimum and maximum of each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        if (data.Rows == 0 || numColumns == 0)
        {
            _dataMin = ScalerTensorOps.FilledVector(numColumns, NumOps.Zero);
            _dataMax = ScalerTensorOps.FilledVector(numColumns, NumOps.One);
            return;
        }

        // scikit-learn: data_min_ / data_max_ are nanmin / nanmax (NaN is treated as missing).
        var x = Tensor<T>.FromMatrix(data);
        _dataMin = ScalerTensorOps.MaskedParameters(mask, ScalerTensorOps.NanColumnMin(x), NumOps.Zero);
        _dataMax = ScalerTensorOps.MaskedParameters(mask, ScalerTensorOps.NanColumnMax(x), NumOps.One);
    }

    /// <summary>
    /// scikit-learn's per-column affine map: <c>scale_ = (max - min) / data_range</c> and
    /// <c>min_ = min - data_min * scale_</c>, where a zero data range counts as 1, so a constant column maps
    /// to the bottom of the feature range.
    /// </summary>
    private (Vector<T> Scale, Vector<T> Offset) ComputeAffine(Vector<T> dataMin, Vector<T> dataMax)
    {
        var low = Tensor<T>.FromVector(dataMin);
        var range = ScalerTensorOps.HandleZerosInScale(Engine.TensorSubtract(Tensor<T>.FromVector(dataMax), low));
        var scale = Engine.TensorDivide(
            PreprocessingTensorOps.Filled(range.Shape.ToArray(), NumOps.Subtract(_featureRangeMax, _featureRangeMin)),
            range);
        var offset = Engine.TensorSubtract(
            PreprocessingTensorOps.Filled(range.Shape.ToArray(), _featureRangeMin), Engine.TensorMultiply(low, scale));
        return (scale.ToVector(), offset.ToVector());
    }

    /// <summary>
    /// Transforms the data by applying min-max scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The scaled data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_dataMin is null || _dataMax is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        // X_scaled = X * scale_ + min_ on the processed columns; the other columns pass through.
        var (scale, offset) = ComputeAffine(_dataMin, _dataMax);
        var scaled = ScalerTensorOps.ScaleThenShift(data, scale, offset);
        var mask = ScalerTensorOps.ColumnMask<T>(data.Columns, GetColumnsToProcess(data.Columns));
        return ScalerTensorOps.MergeColumns(mask, Tensor<T>.FromMatrix(scaled), data);
    }

    /// <summary>
    /// Reverses the min-max scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_dataMin is null || _dataMax is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        // X = (X_scaled - min_) / scale_ on the processed columns; the other columns pass through.
        var (scale, offset) = ComputeAffine(_dataMin, _dataMax);
        var restored = ScalerTensorOps.ShiftThenScale(data, offset, scale);
        var mask = ScalerTensorOps.ColumnMask<T>(data.Columns, GetColumnsToProcess(data.Columns));
        return ScalerTensorOps.MergeColumns(mask, Tensor<T>.FromMatrix(restored), data);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (MinMaxScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
