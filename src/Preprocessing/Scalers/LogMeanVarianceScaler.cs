using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Applies logarithmic transformation followed by mean-variance standardization.
/// </summary>
/// <remarks>
/// <para>
/// Log-mean-variance scaling combines logarithmic transformation with z-score standardization.
/// It first applies log transformation (with shift for negative values), then standardizes
/// using mean and standard deviation. This is ideal for data spanning multiple orders of magnitude.
/// </para>
/// <para><b>For Beginners:</b> This scaler is perfect for highly skewed or exponentially distributed data:
/// - First, it takes the log of values (compressing large differences)
/// - Then, it standardizes to zero mean and unit variance
///
/// Example: [1000, 10000, 100000, 1000000] → after log: [6.9, 9.2, 11.5, 13.8] → standardized
/// This makes patterns in exponential data much easier to detect.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Scaler)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LogMeanVarianceScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _shift;
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _logMean;
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _logStdDev;

    [JsonProperty]
    private readonly T _epsilon;

    /// <summary>
    /// Gets the shift applied to each feature to ensure positive values.
    /// </summary>
    public Vector<T>? Shift => _shift;

    /// <summary>
    /// Gets the mean of log-transformed values for each feature.
    /// </summary>
    public Vector<T>? LogMean => _logMean;

    /// <summary>
    /// Gets the standard deviation of log-transformed values for each feature.
    /// </summary>
    public Vector<T>? LogStdDev => _logStdDev;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="LogMeanVarianceScaler{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public LogMeanVarianceScaler(int[]? columnIndices = null)
        : base(columnIndices)
    {
        _epsilon = NumOps.FromDouble(1e-10);
    }

    /// <summary>
    /// Computes the shift, log mean, and log standard deviation for each feature.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        _shift = ScalerTensorOps.FilledVector(numColumns, NumOps.Zero);
        _logMean = ScalerTensorOps.FilledVector(numColumns, NumOps.Zero);
        _logStdDev = ScalerTensorOps.FilledVector(numColumns, NumOps.One);
        if (data.Rows == 0 || numColumns == 0)
        {
            return;
        }

        // Shift each column so its minimum (NaN is treated as missing) is at least 1 + epsilon when it is not
        // already positive, then take the mean and population standard deviation of log(x + shift).
        var x = Tensor<T>.FromMatrix(data);
        var min = ScalerTensorOps.NanColumnMin(x);
        var zeros = new Tensor<T>(min.Shape.ToArray());
        var shift = Engine.TensorWhere(
            Engine.TensorGreaterThan(min, NumOps.Zero),
            zeros,
            Engine.TensorAddScalar(Engine.TensorNegate(min), NumOps.Add(NumOps.One, _epsilon)));

        var logValues = Engine.TensorLog(Engine.TensorAdd(x, ScalerTensorOps.BroadcastRow(shift, data.Rows)));
        var logMean = ScalerTensorOps.NanColumnMean(logValues);
        var variance = ScalerTensorOps.NanColumnVariance(logValues, logMean);
        var floor = PreprocessingTensorOps.Filled(variance.Shape.ToArray(), _epsilon);
        var logStdDev = Engine.TensorSqrt(
            Engine.TensorWhere(Engine.TensorGreaterThan(variance, _epsilon), variance, floor));

        _shift = ScalerTensorOps.MaskedParameters(mask, shift, NumOps.Zero);
        _logMean = ScalerTensorOps.MaskedParameters(mask, logMean, NumOps.Zero);
        _logStdDev = ScalerTensorOps.MaskedParameters(mask, logStdDev, NumOps.One);
    }

    /// <summary>
    /// Transforms the data by applying log transformation and standardization.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The log-standardized data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_shift is null || _logMean is null || _logStdDev is null)
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

        // Apply shift, log, then standardize; 0 where x + shift is not positive or the result is NaN.
        var shifted = Engine.TensorAdd(Tensor<T>.FromMatrix(data), ScalerTensorOps.BroadcastRow(_shift, numRows));
        var scaled = Engine.TensorDivide(
            Engine.TensorSubtract(Engine.TensorLog(shifted), ScalerTensorOps.BroadcastRow(_logMean, numRows)),
            ScalerTensorOps.BroadcastRow(_logStdDev, numRows));
        var zeros = new Tensor<T>(shifted.Shape.ToArray());
        scaled = Engine.TensorWhere(Engine.TensorLessEqual(shifted, zeros), zeros, scaled);
        scaled = Engine.TensorWhere(Engine.TensorIsNan(scaled), zeros, scaled);

        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        return ScalerTensorOps.MergeColumns(mask, scaled, data);
    }

    /// <summary>
    /// Reverses the log-mean-variance scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_shift is null || _logMean is null || _logStdDev is null)
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

        // Reverse: un-standardize, exp, then remove shift
        var logValue = Engine.TensorAdd(
            Engine.TensorMultiply(Tensor<T>.FromMatrix(data), ScalerTensorOps.BroadcastRow(_logStdDev, numRows)),
            ScalerTensorOps.BroadcastRow(_logMean, numRows));
        var restored = Engine.TensorSubtract(Engine.TensorExp(logValue), ScalerTensorOps.BroadcastRow(_shift, numRows));

        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        return ScalerTensorOps.MergeColumns(mask, restored, data);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (LogMeanVarianceScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
