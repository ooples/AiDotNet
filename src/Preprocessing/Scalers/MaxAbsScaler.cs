using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales each feature by its maximum absolute value.
/// </summary>
/// <remarks>
/// <para>
/// Max absolute scaling transforms each feature by dividing by its maximum absolute value,
/// resulting in features with values in the range [-1, 1]. This scaler does not shift or center
/// the data, so it preserves sparsity.
/// </para>
/// <para><b>For Beginners:</b> This scaler divides each feature by its largest absolute value:
/// - The largest value (positive or negative) becomes 1 or -1
/// - All other values are proportionally scaled
/// - Zero values remain zero (preserves sparsity)
///
/// This is useful when:
/// - You have sparse data and want to preserve zeros
/// - You don't want to center your data
/// - You want values bounded between -1 and 1
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Scaler)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MaxAbsScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _maxAbs;

    /// <summary>
    /// Gets the maximum absolute value of each feature computed during fitting.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? MaxAbsolute => _maxAbs;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="MaxAbsScaler{T}"/>.
    /// </summary>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public MaxAbsScaler(int[]? columnIndices = null)
        : base(columnIndices)
    {
    }

    /// <summary>
    /// Computes the maximum absolute value of each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        if (data.Rows == 0 || numColumns == 0)
        {
            _maxAbs = ScalerTensorOps.FilledVector(numColumns, NumOps.One);
            return;
        }

        // scikit-learn: max_abs_ = nanmax(|X|), and scale_ treats a zero (or sub-10-eps) maximum as 1.
        var maxAbs = ScalerTensorOps.NanColumnMax(Engine.TensorAbs(Tensor<T>.FromMatrix(data)));
        _maxAbs = ScalerTensorOps.MaskedParameters(mask, ScalerTensorOps.HandleZerosInScale(maxAbs), NumOps.One);
    }

    /// <summary>
    /// Transforms the data by applying max absolute scaling.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The scaled data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_maxAbs is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        // Unprocessed columns have scale 1, so x / 1 leaves them exactly unchanged.
        return ScalerTensorOps.ShiftThenScale(data, ScalerTensorOps.FilledVector(_maxAbs.Length, NumOps.Zero), _maxAbs);
    }

    /// <summary>
    /// Reverses the max absolute scaling transformation.
    /// </summary>
    /// <param name="data">The scaled data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_maxAbs is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        return ScalerTensorOps.ScaleThenShift(data, _maxAbs, ScalerTensorOps.FilledVector(_maxAbs.Length, NumOps.Zero));
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (MaxAbsScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
