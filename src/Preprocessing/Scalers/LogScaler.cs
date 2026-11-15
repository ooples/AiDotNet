using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Applies logarithmic transformation to features, useful for data spanning multiple orders of magnitude.
/// </summary>
/// <remarks>
/// <para>
/// Log scaling transforms data using natural logarithm, which compresses the range of values.
/// It shifts negative values to ensure all inputs are positive, then applies log scaling.
/// This is particularly useful for exponentially distributed data.
/// </para>
/// <para><b>For Beginners:</b> Log normalization measures percentages rather than absolute amounts:
/// - With regular measurement, going from 1 to 10 and from 10 to 100 look very different
/// - With logarithmic measurement, both represent a "10× increase" and appear as equal steps
///
/// Example: [1,000, 10,000, 100,000, 1,000,000] becomes evenly spaced values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Scaler)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LogScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _shift;
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _logMin;
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _logRange;

    /// <summary>
    /// Gets the shift applied to each feature to ensure positive values.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? Shift => _shift;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="LogScaler{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public LogScaler(int[]? columnIndices = null)
        : base(columnIndices)
    {
    }

    /// <summary>
    /// Computes the shift and log range for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        _shift = ScalerTensorOps.FilledVector(numColumns, NumOps.Zero);
        _logMin = ScalerTensorOps.FilledVector(numColumns, NumOps.Zero);
        _logRange = ScalerTensorOps.FilledVector(numColumns, NumOps.One);
        if (data.Rows == 0 || numColumns == 0)
        {
            return;
        }

        // Column min / max (NaN is treated as missing), shifted so the minimum is at least 1 when it is not
        // already positive.
        var x = Tensor<T>.FromMatrix(data);
        var min = ScalerTensorOps.NanColumnMin(x);
        var max = ScalerTensorOps.NanColumnMax(x);
        var zeros = new Tensor<T>(min.Shape.ToArray());
        var shift = Engine.TensorWhere(
            Engine.TensorGreaterThan(min, NumOps.Zero), zeros, Engine.TensorAddScalar(Engine.TensorNegate(min), NumOps.One));

        var logMin = Engine.TensorLog(Engine.TensorAdd(min, shift));
        var logRange = Engine.TensorSubtract(Engine.TensorLog(Engine.TensorAdd(max, shift)), logMin);

        // Prevent division by zero
        var flat = Engine.TensorEquals(logRange, NumOps.Zero);
        logRange = Engine.TensorWhere(flat, PreprocessingTensorOps.Filled(logRange.Shape.ToArray(), NumOps.One), logRange);

        _shift = ScalerTensorOps.MaskedParameters(mask, shift, NumOps.Zero);
        _logMin = ScalerTensorOps.MaskedParameters(mask, logMin, NumOps.Zero);
        _logRange = ScalerTensorOps.MaskedParameters(mask, logRange, NumOps.One);
    }

    /// <summary>
    /// Transforms the data by applying log scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The log-scaled data normalized to [0, 1].</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_shift is null || _logMin is null || _logRange is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _shift.Length);
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new Matrix<T>(numRows, numColumns);
        }

        // (log(x + shift) - logMin) / logRange, and 0 where x + shift is not positive (log undefined) or NaN.
        var shifted = Engine.TensorAdd(Tensor<T>.FromMatrix(data), ScalerTensorOps.BroadcastRow(_shift, numRows));
        var scaled = Engine.TensorDivide(
            Engine.TensorSubtract(Engine.TensorLog(shifted), ScalerTensorOps.BroadcastRow(_logMin, numRows)),
            ScalerTensorOps.BroadcastRow(_logRange, numRows));
        var zeros = new Tensor<T>(shifted.Shape.ToArray());
        scaled = Engine.TensorWhere(Engine.TensorLessEqual(shifted, zeros), zeros, scaled);
        scaled = Engine.TensorWhere(Engine.TensorIsNan(shifted), zeros, scaled);

        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        return ScalerTensorOps.MergeColumns(mask, scaled, data);
    }

    /// <summary>
    /// Reverses the log scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_shift is null || _logMin is null || _logRange is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _shift.Length);
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new Matrix<T>(numRows, numColumns);
        }

        // Reverse: exp(value * logRange + logMin) - shift
        var logValue = Engine.TensorAdd(
            Engine.TensorMultiply(Tensor<T>.FromMatrix(data), ScalerTensorOps.BroadcastRow(_logRange, numRows)),
            ScalerTensorOps.BroadcastRow(_logMin, numRows));
        var restored = Engine.TensorSubtract(Engine.TensorExp(logValue), ScalerTensorOps.BroadcastRow(_shift, numRows));

        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        return ScalerTensorOps.MergeColumns(mask, restored, data);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (LogScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
