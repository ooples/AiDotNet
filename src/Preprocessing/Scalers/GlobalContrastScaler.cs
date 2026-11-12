using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales features by adjusting contrast based on mean and standard deviation.
/// </summary>
/// <remarks>
/// <para>
/// Global contrast scaling transforms data using the formula: (x - mean) / (2 * stdDev) + 0.5
/// This centers values around 0.5 and typically results in values between 0 and 1.
/// </para>
/// <para><b>For Beginners:</b> This scaler improves the "contrast" of your data:
/// - It centers values around 0.5 (the new average)
/// - Values above average become > 0.5, below average become &lt; 0.5
/// - Most values end up between 0 and 1
///
/// Example: [68, 70, 71, 69, 72] → [0.3, 0.5, 0.6, 0.4, 0.7]
/// Now the differences between values are more visible.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Scaler)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GlobalContrastScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _mean;
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _stdDev;

    /// <summary>
    /// Gets the mean for each feature computed during fitting.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? Mean => _mean;

    /// <summary>
    /// Gets the standard deviation for each feature computed during fitting.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? StdDev => _stdDev;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="GlobalContrastScaler{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public GlobalContrastScaler(int[]? columnIndices = null)
        : base(columnIndices)
    {
    }

    /// <summary>
    /// Computes the mean and standard deviation for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        if (data.Rows == 0 || numColumns == 0)
        {
            _mean = ScalerTensorOps.FilledVector(numColumns, NumOps.Zero);
            _stdDev = ScalerTensorOps.FilledVector(numColumns, NumOps.One);
            return;
        }

        // Column mean and population standard deviation (NaN is treated as missing); a zero std becomes 1.
        var x = Tensor<T>.FromMatrix(data);
        var mean = ScalerTensorOps.NanColumnMean(x);
        var std = Engine.TensorSqrt(ScalerTensorOps.NanColumnVariance(x, mean));
        var zeroStd = Engine.TensorEquals(std, NumOps.Zero);
        std = Engine.TensorWhere(zeroStd, PreprocessingTensorOps.Filled(std.Shape.ToArray(), NumOps.One), std);

        _mean = ScalerTensorOps.MaskedParameters(mask, mean, NumOps.Zero);
        _stdDev = ScalerTensorOps.MaskedParameters(mask, std, NumOps.One);
    }

    /// <summary>
    /// Transforms the data by applying global contrast scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The scaled data with values typically between 0 and 1.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _stdDev is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        // (x - mean) / (2 * stdDev) + 0.5 on the processed columns; the other columns pass through.
        var doubledStd = Engine.TensorMultiplyScalar(Tensor<T>.FromVector(_stdDev), NumOps.FromDouble(2.0)).ToVector();
        var standardized = ScalerTensorOps.ShiftThenScale(data, _mean, doubledStd);
        if (data.Rows == 0 || data.Columns == 0)
        {
            return standardized;
        }

        var shifted = Engine.TensorAddScalar(Tensor<T>.FromMatrix(standardized), NumOps.FromDouble(0.5));
        var mask = ScalerTensorOps.ColumnMask<T>(data.Columns, GetColumnsToProcess(data.Columns));
        return ScalerTensorOps.MergeColumns(mask, shifted, data);
    }

    /// <summary>
    /// Reverses the global contrast scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _stdDev is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _mean.Length);
        if (data.Rows == 0 || data.Columns == 0)
        {
            return new Matrix<T>(data.Rows, data.Columns);
        }

        // Reverse: (x - 0.5) * 2 * stdDev + mean
        var doubledStd = Engine.TensorMultiplyScalar(Tensor<T>.FromVector(_stdDev), NumOps.FromDouble(2.0)).ToVector();
        var centered = Engine.TensorSubtractScalar(Tensor<T>.FromMatrix(data), NumOps.FromDouble(0.5)).ToMatrix();
        var restored = ScalerTensorOps.ScaleThenShift(centered, doubledStd, _mean);
        var mask = ScalerTensorOps.ColumnMask<T>(data.Columns, GetColumnsToProcess(data.Columns));
        return ScalerTensorOps.MergeColumns(mask, Tensor<T>.FromMatrix(restored), data);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (GlobalContrastScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
