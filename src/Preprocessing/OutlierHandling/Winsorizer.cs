using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Winsorizes data by replacing extreme values with percentile bounds.
/// </summary>
/// <remarks>
/// <para>
/// Winsorizer is a statistical technique that limits extreme values in the data
/// to reduce the effect of outliers. Unlike trimming (which removes outliers),
/// Winsorization replaces them with less extreme values.
/// </para>
/// <para>
/// This is equivalent to OutlierClipper but follows the traditional Winsorization
/// terminology where you specify the percentage of data to Winsorize at each tail.
/// </para>
/// <para><b>For Beginners:</b> Winsorization is named after biostatistician Charles Winsor.
/// Instead of removing outliers, it replaces them with the nearest "normal" values:
/// - If you Winsorize at 5%, the bottom 5% of values become equal to the 5th percentile
/// - The top 5% of values become equal to the 95th percentile
///
/// This preserves sample size while reducing outlier impact.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class Winsorizer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _lowerLimit;
    private readonly double _upperLimit;
    private readonly WinsorizerLimitType _limitType;

    // Fitted parameters
    private Vector<T>? _lowerBounds;
    private Vector<T>? _upperBounds;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the lower limit value.
    /// </summary>
    public double LowerLimit => _lowerLimit;

    /// <summary>
    /// Gets the upper limit value.
    /// </summary>
    public double UpperLimit => _upperLimit;

    /// <summary>
    /// Gets the type of limit (percentile or IQR).
    /// </summary>
    public WinsorizerLimitType LimitType => _limitType;

    /// <summary>
    /// Gets the computed lower bounds for each feature.
    /// </summary>
    public Vector<T>? LowerBounds => _lowerBounds;

    /// <summary>
    /// Gets the computed upper bounds for each feature.
    /// </summary>
    public Vector<T>? UpperBounds => _upperBounds;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="Winsorizer{T}"/>.
    /// </summary>
    /// <param name="lowerLimit">Lower limit. For percentile type: 0-50. For IQR type: multiplier (e.g., 1.5). Defaults to 5.</param>
    /// <param name="upperLimit">Upper limit. For percentile type: 50-100. For IQR type: multiplier (e.g., 1.5). Defaults to 95.</param>
    /// <param name="limitType">Type of limits to use. Defaults to Percentile.</param>
    /// <param name="columnIndices">The column indices to Winsorize, or null for all columns.</param>
    public Winsorizer(
        double lowerLimit = 5.0,
        double upperLimit = 95.0,
        WinsorizerLimitType limitType = WinsorizerLimitType.Percentile,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _limitType = limitType;

        if (limitType == WinsorizerLimitType.Percentile)
        {
            if (lowerLimit < 0 || lowerLimit > 50)
            {
                throw new ArgumentException("Lower percentile must be between 0 and 50.", nameof(lowerLimit));
            }

            if (upperLimit < 50 || upperLimit > 100)
            {
                throw new ArgumentException("Upper percentile must be between 50 and 100.", nameof(upperLimit));
            }

            if (lowerLimit >= upperLimit)
            {
                throw new ArgumentException("Lower limit must be less than upper limit.");
            }
        }
        else // IQR type
        {
            if (lowerLimit < 0)
            {
                throw new ArgumentException("IQR multiplier must be non-negative.", nameof(lowerLimit));
            }

            if (upperLimit < 0)
            {
                throw new ArgumentException("IQR multiplier must be non-negative.", nameof(upperLimit));
            }
        }

        _lowerLimit = lowerLimit;
        _upperLimit = upperLimit;
    }

    /// <summary>
    /// Computes the Winsorization bounds for each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(_nInputFeatures, GetColumnsToProcess(_nInputFeatures));

        // Percentile limits: linearly interpolated percentiles (numpy / pandas default); IQR limits:
        // Q1 - lower * IQR and Q3 + upper * IQR (feature-engine's Winsorizer). NaN is ignored; an empty column
        // gets 0. Pass-through columns get the widest bounds.
        Tensor<T> lower;
        Tensor<T> upper;
        if (data.Rows == 0 || _nInputFeatures == 0)
        {
            lower = new Tensor<T>(new[] { _nInputFeatures });
            upper = new Tensor<T>(new[] { _nInputFeatures });
        }
        else if (_limitType == WinsorizerLimitType.Percentile)
        {
            var x = Tensor<T>.FromMatrix(data);
            lower = Engine.TensorNanQuantile(x, _lowerLimit / 100.0, 0, false, QuantileInterpolation.Linear);
            upper = Engine.TensorNanQuantile(x, _upperLimit / 100.0, 0, false, QuantileInterpolation.Linear);
        }
        else // IQR type
        {
            var x = Tensor<T>.FromMatrix(data);
            var q1 = Engine.TensorNanQuantile(x, 0.25, 0, false, QuantileInterpolation.Linear);
            var q3 = Engine.TensorNanQuantile(x, 0.75, 0, false, QuantileInterpolation.Linear);
            var iqr = Engine.TensorSubtract(q3, q1);
            lower = Engine.TensorSubtract(q1, Engine.TensorMultiplyScalar(iqr, NumOps.FromDouble(_lowerLimit)));
            upper = Engine.TensorAdd(q3, Engine.TensorMultiplyScalar(iqr, NumOps.FromDouble(_upperLimit)));
        }

        _lowerBounds = ScalerTensorOps.MaskedParameters(mask, lower, NumOps.MinValue);
        _upperBounds = ScalerTensorOps.MaskedParameters(mask, upper, NumOps.MaxValue);
    }

    /// <summary>
    /// Winsorizes the data by replacing extreme values with bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The Winsorized data.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("Winsorizer has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _lowerBounds.Length);
        if (data.Rows == 0 || data.Columns == 0)
        {
            return new Matrix<T>(data.Rows, data.Columns);
        }

        // Clip to bounds on the processed columns; the other columns pass through.
        var clipped = ScalerTensorOps.ClipColumns(Tensor<T>.FromMatrix(data), _lowerBounds, _upperBounds);
        var mask = ScalerTensorOps.ColumnMask<T>(data.Columns, GetColumnsToProcess(data.Columns));
        return ScalerTensorOps.MergeColumns(mask, clipped, data);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("Winsorizer does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}

/// <summary>
/// Specifies how Winsorization limits are calculated.
/// </summary>
public enum WinsorizerLimitType
{
    /// <summary>
    /// Use percentile values directly (e.g., 5th and 95th percentile).
    /// </summary>
    Percentile,

    /// <summary>
    /// Use IQR-based limits (Q1 - k*IQR, Q3 + k*IQR where k is the limit value).
    /// </summary>
    IQR
}
