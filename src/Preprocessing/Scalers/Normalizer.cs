using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Specifies the norm to use for sample normalization.
/// </summary>
public enum NormType
{
    /// <summary>
    /// L1 norm (Manhattan distance) - sum of absolute values.
    /// </summary>
    L1,

    /// <summary>
    /// L2 norm (Euclidean distance) - square root of sum of squares.
    /// </summary>
    L2,

    /// <summary>
    /// Max norm (Chebyshev distance) - maximum absolute value.
    /// </summary>
    Max
}

/// <summary>
/// Normalizes samples (rows) individually to unit norm (L1, L2, or Max).
/// </summary>
/// <remarks>
/// <para>
/// Unlike scalers that operate on columns (features), this normalizer operates on rows (samples).
/// Each sample is scaled to have a unit norm (length of 1) in the specified norm type.
/// This is useful when the magnitude of samples varies but their direction matters.
/// </para>
/// <para><b>For Beginners:</b> This normalizer scales each row so its "length" equals 1:
/// - L1 norm: The sum of absolute values equals 1
/// - L2 norm: The Euclidean length (sqrt of sum of squares) equals 1
/// - Max norm: The maximum absolute value equals 1
///
/// Example with L2 norm: [3, 4] has length 5, so it becomes [0.6, 0.8] (length = 1)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Scaler)]
[PipelineStage(PipelineStage.Preprocessing)]
public class Normalizer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    private readonly NormType _normType;

    /// <summary>
    /// Gets the norm type used for normalization.
    /// </summary>
    [JsonIgnore]
    public NormType NormType => _normType;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// Inverse transform is not supported because the original row norms are not stored.
    /// </remarks>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="Normalizer{T}"/>.
    /// </summary>
    /// <param name="normType">The type of norm to use (L1, L2, or Max). Defaults to L2.</param>
    /// <param name="columnIndices">The column indices to include in normalization, or null for all columns.</param>
    public Normalizer(NormType normType = NormType.L2, int[]? columnIndices = null)
        : base(columnIndices)
    {
        _normType = normType;
    }

    /// <summary>
    /// Fits the normalizer to the training data.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    /// <remarks>
    /// Normalizer is stateless for transformation - fitting validates the data structure.
    /// </remarks>
    protected override void FitCore(Matrix<T> data)
    {
        // Normalizer is stateless - each row is normalized independently
        // No parameters need to be learned from training data
    }

    /// <summary>
    /// Transforms the data by normalizing each row to unit norm.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The normalized data where each row has unit norm.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        int numRows = data.Rows;
        int numColumns = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        if (numRows == 0 || numColumns == 0)
        {
            return new Matrix<T>(numRows, numColumns);
        }

        // Row norms over the selected columns; like scikit-learn's normalize, a norm below 10 * eps is treated
        // as 1 so an all-zero row stays zero.
        var x = Tensor<T>.FromMatrix(data);
        // Half is normalized in float (see ScalerTensorOps.ReducesInFloat): the squares of entries past 256 and
        // row sums past 65504 overflow Half, which zeroed rows whose unit-norm values are representable.
        var normalized = ScalerTensorOps.ReducesInFloat<T>()
            ? NormalizeRows(x.Cast<float>(), mask.Cast<float>()).Cast<T>()
            : NormalizeRows(x, mask);
        return ScalerTensorOps.MergeColumns(mask, normalized, data);
    }

    /// <summary>
    /// Every row of <paramref name="x"/> ([n, p]) divided by its norm over the columns selected by the 0/1
    /// <paramref name="columnMask"/> ([p]).
    /// </summary>
    private Tensor<TW> NormalizeRows<TW>(Tensor<TW> x, Tensor<TW> columnMask)
    {
        var shape = x.Shape.ToArray();
        // Unselected entries are replaced by 0, not multiplied by it: NaN * 0 and Inf * 0 are NaN, which would
        // poison the norm of the selected columns.
        var selected = Engine.TensorWhere(ScalerTensorOps.BroadcastRow(columnMask, shape[0]), x, new Tensor<TW>(shape));
        // scikit-learn's _handle_zeros_in_scale with the eps of the data's dtype T, not of the wider TW
        var norm = ScalerTensorOps.HandleZerosInScale(
            CalculateRowNorms(selected), null, ScalerTensorOps.MachineEpsilon<T>());
        return Engine.TensorDivide(x, Engine.TensorBroadcastTo(norm, shape));
    }

    /// <summary>The <see cref="NormType"/> norm of every row of <paramref name="x"/>, as [n, 1].</summary>
    private Tensor<TW> CalculateRowNorms<TW>(Tensor<TW> x)
    {
        var axis = new[] { 1 };
        switch (_normType)
        {
            case NormType.L1:
                return Engine.ReduceSum(Engine.TensorAbs(x), axis, true);
            case NormType.L2:
                return Engine.TensorSqrt(Engine.ReduceSum(Engine.TensorSquare(x), axis, true));
            case NormType.Max:
                return Engine.ReduceMax(Engine.TensorAbs(x), axis, true);
            default:
                throw new ArgumentException($"Unknown norm type: {_normType}");
        }
    }

    /// <summary>
    /// Inverse transformation is not supported for sample normalization.
    /// </summary>
    /// <param name="data">The normalized data.</param>
    /// <returns>Never returns - always throws.</returns>
    /// <exception cref="NotSupportedException">Always thrown because sample norms are not stored.</exception>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException(
            "Normalizer does not support inverse transformation. " +
            "The original row norms are not stored during transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (Normalizer doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
