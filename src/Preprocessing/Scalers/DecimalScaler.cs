using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales features by dividing by the smallest power of 10 greater than the max absolute value.
/// </summary>
/// <remarks>
/// <para>
/// Decimal scaling transforms values to fall between -1 and 1 by dividing by an appropriate power of 10.
/// This preserves the relative decimal positions and signs of values.
/// </para>
/// <para><b>For Beginners:</b> This scaler adjusts numbers to show them in appropriate decimal places:
/// - If your largest value is 750, it divides everything by 1,000
/// - So 750 becomes 0.75, 42 becomes 0.042, etc.
/// - All values end up between -1 and 1
///
/// This is useful when you want to keep relative sizes clear and decimal places meaningful.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Scaler)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DecimalScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _scale;

    /// <summary>
    /// Gets the power-of-10 scale factor for each feature computed during fitting.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? Scale => _scale;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="DecimalScaler{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public DecimalScaler(int[]? columnIndices = null)
        : base(columnIndices)
    {
    }

    /// <summary>
    /// Computes the power-of-10 scale factor for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        if (data.Rows == 0 || numColumns == 0)
        {
            _scale = ScalerTensorOps.FilledVector(numColumns, NumOps.One);
            return;
        }

        // Decimal scaling (Han & Kamber): scale = 10^j for the smallest j >= 0 with max|x| / 10^j < 1, i.e. the
        // smallest power of ten strictly above max|x|. NaN does not take part in the maximum.
        var maxAbs = Engine.TensorMax(
            ScalerTensorOps.NanColumnMax(Engine.TensorAbs(Tensor<T>.FromMatrix(data))), NumOps.Zero);
        T ten = NumOps.FromDouble(10.0);
        var tens = PreprocessingTensorOps.Filled(maxAbs.Shape.ToArray(), ten);
        var log10 = Engine.TensorDivideScalar(Engine.TensorLog(maxAbs), NumOps.Log(ten));
        var exponent = Engine.TensorMax(Engine.TensorAddScalar(Engine.TensorFloor(log10), NumOps.One), NumOps.Zero);
        var scale = Engine.TensorPower(tens, exponent);

        // log10 can land a hair off an exact power of ten; step the power once either way to fix it.
        var tooSmall = Engine.TensorGreaterEqual(maxAbs, scale);
        scale = Engine.TensorWhere(tooSmall, Engine.TensorMultiplyScalar(scale, ten), scale);
        var lower = Engine.TensorDivideScalar(scale, ten);
        var tooLarge = Engine.TensorMultiply(
            Engine.TensorLessThan(maxAbs, lower), Engine.TensorGreaterEqual(lower, PreprocessingTensorOps.Filled(lower.Shape.ToArray(), NumOps.One)));
        scale = Engine.TensorWhere(tooLarge, lower, scale);

        _scale = ScalerTensorOps.MaskedParameters(mask, scale, NumOps.One);
    }

    /// <summary>
    /// Transforms the data by applying decimal scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The scaled data with values between -1 and 1.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_scale is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        // Unprocessed columns have scale 1, so x / 1 leaves them exactly unchanged.
        return ScalerTensorOps.ShiftThenScale(data, ScalerTensorOps.FilledVector(_scale.Length, NumOps.Zero), _scale);
    }

    /// <summary>
    /// Reverses the decimal scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_scale is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        return ScalerTensorOps.ScaleThenShift(data, _scale, ScalerTensorOps.FilledVector(_scale.Length, NumOps.Zero));
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (DecimalScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
