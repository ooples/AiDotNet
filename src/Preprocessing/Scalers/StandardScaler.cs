using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Scalers;

/// <summary>
/// Standardizes features by removing the mean and scaling to unit variance.
/// </summary>
/// <remarks>
/// <para>
/// Standard scaling (Z-score normalization) transforms data to have a mean of 0 and a
/// standard deviation of 1. This is important for many machine learning algorithms as
/// it puts different features on comparable scales.
/// </para>
/// <para><b>For Beginners:</b> This scaler converts your data to a standard scale:
/// - The center of your data (mean) becomes 0
/// - The spread of your data (standard deviation) becomes 1
///
/// This is like converting different currencies to a common one - it makes
/// different features comparable and helps many ML algorithms work better.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Scaler)]
[PipelineStage(PipelineStage.Preprocessing)]
public class StandardScaler<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _mean;
    [JsonProperty]
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _stdDev;
    [JsonProperty]
    private readonly bool _withMean;
    [JsonProperty]
    private readonly bool _withStd;

    /// <summary>
    /// Gets the mean of each feature computed during fitting.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? Mean => _mean;

    /// <summary>
    /// Gets the standard deviation of each feature computed during fitting.
    /// </summary>
    [JsonIgnore]
    public Vector<T>? StandardDeviation => _stdDev;

    /// <summary>
    /// Gets whether this scaler centers the data (subtracts mean).
    /// </summary>
    [JsonIgnore]
    public bool WithMean => _withMean;

    /// <summary>
    /// Gets whether this scaler scales the data (divides by std).
    /// </summary>
    [JsonIgnore]
    public bool WithStd => _withStd;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="StandardScaler{T}"/>.
    /// </summary>
    /// <param name="withMean">If true, center the data before scaling (subtract mean). Default is true.</param>
    /// <param name="withStd">If true, scale the data to unit variance. Default is true.</param>
    /// <param name="columnIndices">The column indices to scale, or null for all columns.</param>
    public StandardScaler(bool withMean = true, bool withStd = true, int[]? columnIndices = null)
        : base(columnIndices)
    {
        _withMean = withMean;
        _withStd = withStd;
    }

    /// <summary>
    /// Computes the mean and standard deviation of each feature from the training data.
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

        // scikit-learn semantics: NaN is missing (ignored by the statistics), the variance is the biased
        // (ddof = 0) estimator, and a column whose variance is within floating-point error of zero is constant.
        var x = Tensor<T>.FromMatrix(data);
        if (!_withStd)
        {
            _mean = ScalerTensorOps.MaskedParameters(mask, ScalerTensorOps.NanColumnMean(x), NumOps.Zero);
            _stdDev = ScalerTensorOps.FilledVector(numColumns, NumOps.One);
            return;
        }

        var (mean, std, constant) = ScalerTensorOps.NanColumnMoments(x);
        _mean = ScalerTensorOps.MaskedParameters(mask, mean, NumOps.Zero);
        _stdDev = ScalerTensorOps.MaskedParameters(mask, ScalerTensorOps.HandleZerosInScale(std, constant), NumOps.One);
    }

    /// <summary>
    /// Transforms the data by applying the computed mean and standard deviation.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The standardized data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_mean is null || _stdDev is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        // Unprocessed columns have mean 0 and scale 1, so (x - 0) / 1 leaves them exactly unchanged.
        var shift = _withMean ? _mean : ScalerTensorOps.FilledVector(_mean.Length, NumOps.Zero);
        var scale = _withStd ? _stdDev : ScalerTensorOps.FilledVector(_stdDev.Length, NumOps.One);
        return ScalerTensorOps.ShiftThenScale(data, shift, scale);
    }

    /// <summary>
    /// Reverses the standardization transformation.
    /// </summary>
    /// <param name="data">The standardized data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_mean is null || _stdDev is null)
        {
            throw new InvalidOperationException("Scaler has not been fitted.");
        }

        // Reverse standardization: x = z * std + mean
        var shift = _withMean ? _mean : ScalerTensorOps.FilledVector(_mean.Length, NumOps.Zero);
        var scale = _withStd ? _stdDev : ScalerTensorOps.FilledVector(_stdDev.Length, NumOps.One);
        return ScalerTensorOps.ScaleThenShift(data, scale, shift);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (StandardScaler doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
