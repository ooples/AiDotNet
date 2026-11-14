using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Clips outliers to specified percentile bounds.
/// </summary>
/// <remarks>
/// <para>
/// OutlierClipper clips values below a lower percentile and above an upper percentile
/// to those percentile values. This reduces the impact of extreme outliers while
/// preserving the overall data distribution.
/// </para>
/// <para><b>For Beginners:</b> Outliers are extreme values that are far from most of your data.
/// They can distort your model's learning. Clipping replaces extreme values with more
/// reasonable bounds:
/// - Values below the 1st percentile → replaced with 1st percentile value
/// - Values above the 99th percentile → replaced with 99th percentile value
///
/// Example: Income data where most people earn $30K-$200K but a few billionaires
/// would be clipped to prevent the model from being skewed by extreme wealth.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class OutlierClipper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _lowerPercentile;
    private readonly double _upperPercentile;

    // Fitted parameters
    private Vector<T>? _lowerBounds;
    private Vector<T>? _upperBounds;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the lower percentile (values below this are clipped).
    /// </summary>
    public double LowerPercentile => _lowerPercentile;

    /// <summary>
    /// Gets the upper percentile (values above this are clipped).
    /// </summary>
    public double UpperPercentile => _upperPercentile;

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
    /// Creates a new instance of <see cref="OutlierClipper{T}"/>.
    /// </summary>
    /// <param name="lowerPercentile">Lower percentile bound (0-100). Values below are clipped. Defaults to 1.</param>
    /// <param name="upperPercentile">Upper percentile bound (0-100). Values above are clipped. Defaults to 99.</param>
    /// <param name="columnIndices">The column indices to clip, or null for all columns.</param>
    public OutlierClipper(
        double lowerPercentile = 1.0,
        double upperPercentile = 99.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (lowerPercentile < 0 || lowerPercentile > 100)
        {
            throw new ArgumentException("Lower percentile must be between 0 and 100.", nameof(lowerPercentile));
        }

        if (upperPercentile < 0 || upperPercentile > 100)
        {
            throw new ArgumentException("Upper percentile must be between 0 and 100.", nameof(upperPercentile));
        }

        if (lowerPercentile >= upperPercentile)
        {
            throw new ArgumentException("Lower percentile must be less than upper percentile.");
        }

        _lowerPercentile = lowerPercentile;
        _upperPercentile = upperPercentile;
    }

    /// <summary>
    /// Computes the percentile bounds for each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(_nInputFeatures, GetColumnsToProcess(_nInputFeatures));

        // Linearly interpolated percentiles (numpy's default), ignoring NaN; an empty column gets 0.
        // Pass-through columns get the widest bounds.
        Tensor<T> lower;
        Tensor<T> upper;
        if (data.Rows == 0 || _nInputFeatures == 0)
        {
            lower = new Tensor<T>(new[] { _nInputFeatures });
            upper = new Tensor<T>(new[] { _nInputFeatures });
        }
        else
        {
            var x = Tensor<T>.FromMatrix(data);
            lower = Engine.TensorNanQuantile(x, _lowerPercentile / 100.0, 0, false, QuantileInterpolation.Linear);
            upper = Engine.TensorNanQuantile(x, _upperPercentile / 100.0, 0, false, QuantileInterpolation.Linear);
        }

        _lowerBounds = ScalerTensorOps.MaskedParameters(mask, lower, NumOps.MinValue);
        _upperBounds = ScalerTensorOps.MaskedParameters(mask, upper, NumOps.MaxValue);
    }

    /// <summary>
    /// Clips values to the computed percentile bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with outliers clipped.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("Clipper has not been fitted.");
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
        throw new NotSupportedException("OutlierClipper does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
