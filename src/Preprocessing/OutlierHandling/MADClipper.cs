using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Clips outliers based on Median Absolute Deviation (MAD) bounds.
/// </summary>
/// <remarks>
/// <para>
/// MADClipper identifies outliers using the Median Absolute Deviation, which is more robust
/// to outliers than standard deviation-based methods. Values with modified Z-scores exceeding
/// the threshold are clipped to the boundary values.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// For each feature:
/// 1. Calculate the median
/// 2. Calculate MAD = median(|x - median|)
/// 3. Calculate modified Z-score = 0.6745 * (x - median) / MAD
/// 4. Clip values where |modified Z-score| > threshold
/// </para>
/// <para>
/// <b>Why MAD is Better Than Z-Score for Outliers:</b>
/// The traditional Z-score uses mean and standard deviation, which are themselves
/// affected by outliers. MAD uses medians, which are resistant to extreme values.
/// This makes MAD better at detecting outliers when your data already contains
/// significant outliers.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of the median as the "middle value" when you sort your data.
/// MAD measures how spread out your data is from this middle value. Unlike the average,
/// the median isn't pulled toward extreme values, so MAD gives a more reliable measure
/// of spread when outliers are present.
/// </para>
/// <para>
/// <b>Common Thresholds:</b>
/// - 2.5: Aggressive (more values treated as outliers)
/// - 3.5: Standard (recommended default)
/// - 5.0: Conservative (only extreme outliers)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MADClipper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    /// <summary>
    /// Constant for converting MAD to a consistent estimate of the standard deviation for normal data:
    /// 1 / Phi^-1(3/4) = 1.4826022185..., the factor scipy.stats.median_abs_deviation(scale='normal') uses.
    /// </summary>
    private const double MADScaleFactor = 1.0 / ModifiedZScoreConstant;

    /// <summary>
    /// The constant of Iglewicz and Hoaglin's modified Z-score: the 0.75 quantile of the standard normal,
    /// Phi^-1(3/4) (published rounded to 0.6745), so k * (x - median) / MAD is comparable to a Z-score.
    /// </summary>
    private const double ModifiedZScoreConstant = 0.6744897501960817;

    private readonly double _threshold;

    // Fitted parameters
    private Vector<T>? _medians;
    private Vector<T>? _mads;
    private Vector<T>? _lowerBounds;
    private Vector<T>? _upperBounds;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the modified Z-score threshold for clipping.
    /// </summary>
    public double Threshold => _threshold;

    /// <summary>
    /// Gets the computed medians for each feature.
    /// </summary>
    public Vector<T>? Medians => _medians;

    /// <summary>
    /// Gets the computed MAD values for each feature.
    /// </summary>
    public Vector<T>? MADs => _mads;

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
    /// Creates a new instance of <see cref="MADClipper{T}"/>.
    /// </summary>
    /// <param name="threshold">The modified Z-score threshold. Values beyond this are clipped. Defaults to 3.5.</param>
    /// <param name="columnIndices">The column indices to clip, or null for all columns.</param>
    public MADClipper(double threshold = 3.5, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold <= 0)
        {
            throw new ArgumentException("Threshold must be positive.", nameof(threshold));
        }

        _threshold = threshold;
    }

    /// <summary>
    /// Computes the median and MAD for each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(_nInputFeatures, GetColumnsToProcess(_nInputFeatures));

        // Median and MAD = median(|x - median|), ignoring NaN; an empty column gets 0.
        Tensor<T> median;
        Tensor<T> mad;
        if (data.Rows == 0 || _nInputFeatures == 0)
        {
            median = new Tensor<T>(new[] { _nInputFeatures });
            mad = new Tensor<T>(new[] { _nInputFeatures });
        }
        else
        {
            var x = Tensor<T>.FromMatrix(data);
            median = Engine.TensorNanQuantile(x, 0.5, 0, false, QuantileInterpolation.Linear);
            var deviation = Engine.TensorAbs(Engine.TensorSubtract(x, ScalerTensorOps.BroadcastRow(median, data.Rows)));
            mad = Engine.TensorNanQuantile(deviation, 0.5, 0, false, QuantileInterpolation.Linear);
        }

        _medians = ScalerTensorOps.MaskedParameters(mask, median, NumOps.Zero);
        _mads = ScalerTensorOps.MaskedParameters(mask, mad, NumOps.One);

        // Modified Z-score = Phi^-1(3/4) * (x - median) / MAD, so |M| <= threshold means
        // x in median -/+ threshold * MAD / Phi^-1(3/4). A zero MAD (all values equal) and the pass-through
        // columns get the widest bounds (no clipping).
        var spread = Engine.TensorMultiplyScalar(mad, NumOps.FromDouble(_threshold * MADScaleFactor));
        var clips = Engine.TensorMultiply(mask, Engine.TensorGreaterThan(mad, NumOps.Zero));
        _lowerBounds = ScalerTensorOps.MaskedParameters(clips, Engine.TensorSubtract(median, spread), NumOps.MinValue);
        _upperBounds = ScalerTensorOps.MaskedParameters(clips, Engine.TensorAdd(median, spread), NumOps.MaxValue);
    }

    /// <summary>
    /// Clips values to the computed MAD bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with outliers clipped.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("MADClipper has not been fitted.");
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
        throw new NotSupportedException("MADClipper does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Calculates the modified Z-scores for all values in the data.
    /// </summary>
    /// <param name="data">The data to calculate modified Z-scores for.</param>
    /// <returns>A matrix of modified Z-scores.</returns>
    public Matrix<T> GetModifiedZScores(Matrix<T> data)
    {
        if (_medians is null || _mads is null)
        {
            throw new InvalidOperationException("MADClipper has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _medians.Length);
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new Matrix<T>(numRows, numColumns);
        }

        // Modified Z-score = Phi^-1(3/4) * (x - median) / MAD on processed columns with MAD > 0, otherwise 0.
        var mad = Tensor<T>.FromVector(_mads);
        var positive = Engine.TensorGreaterThan(mad, NumOps.Zero);
        var scored = Engine.TensorMultiply(
            ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns)), positive);
        var safeMad = Engine.TensorWhere(positive, mad, PreprocessingTensorOps.Filled(mad.Shape.ToArray(), NumOps.One));
        var centered = Engine.TensorSubtract(Tensor<T>.FromMatrix(data), ScalerTensorOps.BroadcastRow(_medians, numRows));
        var z = Engine.TensorMultiplyScalar(
            Engine.TensorDivide(centered, ScalerTensorOps.BroadcastRow(safeMad, numRows)),
            NumOps.FromDouble(ModifiedZScoreConstant));
        return Engine.TensorWhere(
            ScalerTensorOps.BroadcastRow(scored, numRows), z, new Tensor<T>(new[] { numRows, numColumns })).ToMatrix();
    }

    /// <summary>
    /// Gets a boolean mask indicating which values are outliers.
    /// </summary>
    /// <param name="data">The data to check for outliers.</param>
    /// <returns>A 2D boolean array where true indicates an outlier.</returns>
    public bool[,] GetOutlierMask(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("MADClipper has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _lowerBounds.Length);
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new bool[numRows, numColumns];
        }

        var outside = ScalerTensorOps.OutsideMask(Tensor<T>.FromMatrix(data), _lowerBounds, _upperBounds);
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        return ScalerTensorOps.ToBoolMatrix(Engine.TensorMultiply(outside, ScalerTensorOps.BroadcastRow(mask, numRows)));
    }
}
