using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Clips outliers based on Z-score (standard deviation) bounds.
/// </summary>
/// <remarks>
/// <para>
/// ZScoreClipper identifies outliers as values that deviate from the mean by more than
/// a specified number of standard deviations. Values beyond this threshold are clipped
/// to the boundary values.
/// </para>
/// <para>
/// <b>How It Works:</b>
/// For each feature:
/// 1. Calculate the mean and standard deviation
/// 2. Compute bounds: [mean - threshold*std, mean + threshold*std]
/// 3. Clip values outside these bounds
/// </para>
/// <para>
/// <b>For Beginners:</b> Z-score measures how many standard deviations a value is from the mean.
/// - A Z-score of 0 means the value equals the mean
/// - A Z-score of 2 means the value is 2 standard deviations above the mean
/// - A Z-score of -3 means the value is 3 standard deviations below the mean
///
/// This clipper replaces extreme values (those with high absolute Z-scores) with
/// the boundary values, reducing the impact of outliers while preserving data size.
/// </para>
/// <para>
/// <b>Common Thresholds:</b>
/// - 2.0: Aggressive (clips ~5% of normally distributed data)
/// - 3.0: Standard (clips ~0.3% of normally distributed data)
/// - 3.5: Conservative (clips ~0.05% of normally distributed data)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ZScoreClipper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _threshold;

    // Fitted parameters
    private Vector<T>? _means;
    private Vector<T>? _stds;
    private Vector<T>? _lowerBounds;
    private Vector<T>? _upperBounds;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the Z-score threshold for clipping.
    /// </summary>
    public double Threshold => _threshold;

    /// <summary>
    /// Gets the computed means for each feature.
    /// </summary>
    public Vector<T>? Means => _means;

    /// <summary>
    /// Gets the computed standard deviations for each feature.
    /// </summary>
    public Vector<T>? StandardDeviations => _stds;

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
    /// Creates a new instance of <see cref="ZScoreClipper{T}"/>.
    /// </summary>
    /// <param name="threshold">The Z-score threshold. Values beyond mean +/- threshold*std are clipped. Defaults to 3.0.</param>
    /// <param name="columnIndices">The column indices to clip, or null for all columns.</param>
    public ZScoreClipper(double threshold = 3.0, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (threshold <= 0)
        {
            throw new ArgumentException("Threshold must be positive.", nameof(threshold));
        }

        _threshold = threshold;
    }

    /// <summary>
    /// Computes the mean and standard deviation for each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(_nInputFeatures, GetColumnsToProcess(_nInputFeatures));

        // Mean and population standard deviation of each column (NaN is treated as missing).
        Tensor<T> mean;
        Tensor<T> std;
        if (data.Rows == 0 || _nInputFeatures == 0)
        {
            mean = new Tensor<T>(new[] { _nInputFeatures });
            std = new Tensor<T>(new[] { _nInputFeatures });
        }
        else
        {
            (mean, std, _) = ScalerTensorOps.NanColumnMoments(Tensor<T>.FromMatrix(data));
        }

        _means = ScalerTensorOps.MaskedParameters(mask, mean, NumOps.Zero);
        _stds = ScalerTensorOps.MaskedParameters(mask, std, NumOps.One);

        // Bounds mean -/+ threshold * std; a zero std (all values the same) and the pass-through columns get
        // the widest bounds (no clipping).
        var spread = Engine.TensorMultiplyScalar(std, NumOps.FromDouble(_threshold));
        var clips = Engine.TensorMultiply(mask, Engine.TensorGreaterThan(std, NumOps.Zero));
        _lowerBounds = ScalerTensorOps.MaskedParameters(clips, Engine.TensorSubtract(mean, spread), NumOps.MinValue);
        _upperBounds = ScalerTensorOps.MaskedParameters(clips, Engine.TensorAdd(mean, spread), NumOps.MaxValue);
    }

    /// <summary>
    /// Clips values to the computed Z-score bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with outliers clipped.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("ZScoreClipper has not been fitted.");
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
        throw new NotSupportedException("ZScoreClipper does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Calculates the Z-scores for all values in the data.
    /// </summary>
    /// <param name="data">The data to calculate Z-scores for.</param>
    /// <returns>A matrix of Z-scores.</returns>
    public Matrix<T> GetZScores(Matrix<T> data)
    {
        if (_means is null || _stds is null)
        {
            throw new InvalidOperationException("ZScoreClipper has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _means.Length);
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new Matrix<T>(numRows, numColumns);
        }

        // (x - mean) / std on processed columns with std > 0, otherwise 0.
        var std = Tensor<T>.FromVector(_stds);
        var positive = Engine.TensorGreaterThan(std, NumOps.Zero);
        var scored = Engine.TensorMultiply(
            ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns)), positive);
        var safeStd = Engine.TensorWhere(positive, std, PreprocessingTensorOps.Filled(std.Shape.ToArray(), NumOps.One));
        var z = Tensor<T>.FromMatrix(ScalerTensorOps.ShiftThenScale(data, _means, safeStd.ToVector()));
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
            throw new InvalidOperationException("ZScoreClipper has not been fitted.");
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
