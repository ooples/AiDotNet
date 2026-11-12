using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Scales features (columns) by dividing each element by the column's Lp-norm.
/// </summary>
/// <remarks>
/// <para>
/// The LpNormScaler normalizes each feature (column) by dividing every element by the
/// column's Lp-norm. This results in each column having a unit Lp-norm. The Lp-norm is a
/// generalization of different vector norms based on the parameter p:
/// - p = 1: Manhattan (L1) norm (sum of absolute values)
/// - p = 2: Euclidean (L2) norm (square root of sum of squares)
/// - p = infinity: Maximum (L-infinity) norm (maximum absolute value)
/// </para>
/// <para>
/// Note: This scaler operates on columns (features), which is different from the Normalizer class
/// that operates on rows (samples). Use this when you want to normalize feature vectors;
/// use Normalizer when you want to normalize sample vectors.
/// </para>
/// <para><b>For Beginners:</b> This scaler standardizes each feature column to have a consistent "length".
///
/// Think of each column as an arrow (vector) in space:
/// - The Lp-norm measures the "length" of this arrow
/// - This scaler divides each element by the length
/// - The result is an arrow pointing in the same direction but with length = 1
///
/// Different p values provide different ways to measure length:
/// - p = 1: Like measuring distance by walking along city blocks
/// - p = 2: Like measuring distance "as the crow flies" (straight line)
/// - Higher p values: Increasingly emphasize the largest component
///
/// For example, normalizing the column [3, 4] with p = 2 (Euclidean norm):
/// - The norm is sqrt(3^2 + 4^2) = sqrt(25) = 5
/// - The normalized column is [3/5, 4/5] = [0.6, 0.8]
/// - This new column has length 1 using the L2 norm
///
/// This is useful for:
/// - Feature normalization in machine learning models
/// - Ensuring consistent scaling across feature vectors
/// - Applications where feature direction matters more than magnitude
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Scaler)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LpNormScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    private readonly double _p;

    // Fitted parameters: the Lp-norm of each column
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _columnNorms;
    [JsonProperty]
    private int _nColumns;

    /// <summary>
    /// Gets the p parameter that defines which Lp-norm to use.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The p value determines how the "length" of each column is measured:
    /// - p = 1: Sum of absolute values (Manhattan norm)
    /// - p = 2: Square root of sum of squares (Euclidean norm, most common)
    /// - Large p values: Approaches the maximum absolute value
    ///
    /// Most applications use p = 2 (Euclidean norm).
    /// </para>
    /// </remarks>
    public double P => _p;

    /// <summary>
    /// Gets the Lp-norm of each column computed during fitting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After fitting, this property contains the "length" of each
    /// original feature column. These values were used to divide each column during transformation,
    /// and will be used to multiply back during inverse transformation.
    /// </para>
    /// </remarks>
    public Vector<T>? ColumnNorms => _columnNorms;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Yes, Lp-norm scaling can be reversed. Since we just divided
    /// by the norm, we can multiply by the norm to get back the original values.
    /// </para>
    /// </remarks>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="LpNormScaler{T}"/>.
    /// </summary>
    /// <param name="p">The p parameter for the Lp-norm. Must be >= 1. Defaults to 2 (Euclidean norm).</param>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new Lp-norm scaler.
    ///
    /// Common p values:
    /// - p = 1: L1 norm (Manhattan distance)
    /// - p = 2: L2 norm (Euclidean distance) - the default and most common choice
    /// - p = large value (e.g., 100): Approximates L-infinity norm (max norm)
    ///
    /// Use columnIndices to apply scaling only to specific columns, leaving others unchanged.
    /// </para>
    /// </remarks>
    public LpNormScaler(double p = 2.0, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (p < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(p),
                "The p parameter must be >= 1 for a valid Lp-norm.");
        }

        _p = p;
    }

    /// <summary>
    /// Computes the Lp-norm of each column.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method measures the "length" of each feature column.
    ///
    /// For each column, it calculates: (|x1|^p + |x2|^p + ... + |xn|^p)^(1/p)
    ///
    /// These norm values are stored and used to:
    /// - Divide each element during Transform (to normalize)
    /// - Multiply each element during InverseTransform (to denormalize)
    /// </para>
    /// </remarks>
    protected override void FitCore(Matrix<T> data)
    {
        _nColumns = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(_nColumns, GetColumnsToProcess(_nColumns));
        if (data.Rows == 0 || _nColumns == 0)
        {
            _columnNorms = ScalerTensorOps.FilledVector(_nColumns, NumOps.One);
            return;
        }

        // Lp-norm of each column: (sum |x|^p)^(1/p); a norm that is zero up to rounding (below 10 * eps, as
        // scikit-learn's normalize(axis=0) treats it) becomes 1, i.e. no scaling.
        var x = Tensor<T>.FromMatrix(data);
        var powered = Engine.TensorPower(Engine.TensorAbs(x), NumOps.FromDouble(_p));
        var norms = Engine.TensorPower(Engine.ReduceSum(powered, new[] { 0 }, false), NumOps.FromDouble(1.0 / _p));
        _columnNorms = ScalerTensorOps.MaskedParameters(mask, ScalerTensorOps.HandleZerosInScale(norms), NumOps.One);
    }

    /// <summary>
    /// Transforms the data by dividing each column by its Lp-norm.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The Lp-normalized data where each column has unit Lp-norm.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method divides each element by its column's norm.
    ///
    /// After transformation:
    /// - Each processed column has Lp-norm = 1
    /// - The relative proportions within each column are preserved
    /// - Columns not in columnIndices are unchanged
    ///
    /// This makes columns comparable by standardizing their "lengths".
    /// </para>
    /// </remarks>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_columnNorms is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        // Unprocessed columns have norm 1, so x / 1 leaves them exactly unchanged.
        return ScalerTensorOps.ShiftThenScale(
            data, ScalerTensorOps.FilledVector(_columnNorms.Length, NumOps.Zero), _columnNorms);
    }

    /// <summary>
    /// Reverses the Lp-norm scaling by multiplying each column by its original norm.
    /// </summary>
    /// <param name="data">The Lp-normalized data.</param>
    /// <returns>The original-scale data.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method undoes the normalization by multiplying
    /// each element by the column's original norm.
    ///
    /// Since we divided by the norm during Transform, we multiply by the norm here
    /// to get back the original values (within floating-point precision).
    /// </para>
    /// </remarks>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_columnNorms is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        return ScalerTensorOps.ScaleThenShift(
            data, _columnNorms, ScalerTensorOps.FilledVector(_columnNorms.Length, NumOps.Zero));
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (LpNormScaler doesn't change number of features).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lp-norm scaling doesn't add or remove features, it just
    /// scales them. So the feature names remain the same.
    /// </para>
    /// </remarks>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
