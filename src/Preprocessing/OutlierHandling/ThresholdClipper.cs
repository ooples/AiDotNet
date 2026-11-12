using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Clips values based on explicit threshold bounds.
/// </summary>
/// <remarks>
/// <para>
/// ThresholdClipper clips values to user-specified lower and upper bounds. Unlike other
/// clippers that compute bounds from data statistics, this clipper uses explicit thresholds
/// provided by the user.
/// </para>
/// <para>
/// <b>Use Cases:</b>
/// - Domain-specific constraints (e.g., percentages must be 0-100)
/// - Physical limits (e.g., temperatures can't be below absolute zero)
/// - Business rules (e.g., prices must be positive)
/// - Known valid ranges from domain expertise
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the simplest outlier clipper - you tell it exactly what
/// the minimum and maximum allowed values are, and it clips anything outside those bounds.
/// This is useful when you know from domain knowledge what the valid range should be.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ThresholdClipper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _lowerThreshold;
    private readonly double _upperThreshold;

    private int _nInputFeatures;

    /// <summary>
    /// Gets the lower threshold bound.
    /// </summary>
    public double LowerThreshold => _lowerThreshold;

    /// <summary>
    /// Gets the upper threshold bound.
    /// </summary>
    public double UpperThreshold => _upperThreshold;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="ThresholdClipper{T}"/> with symmetric bounds.
    /// </summary>
    /// <param name="threshold">The symmetric threshold. Values below -threshold or above +threshold are clipped.</param>
    /// <param name="columnIndices">The column indices to clip, or null for all columns.</param>
    public ThresholdClipper(double threshold, int[]? columnIndices = null)
        : this(-Math.Abs(threshold), Math.Abs(threshold), columnIndices)
    {
    }

    /// <summary>
    /// Creates a new instance of <see cref="ThresholdClipper{T}"/> with explicit bounds.
    /// </summary>
    /// <param name="lowerThreshold">The lower bound. Values below this are clipped up.</param>
    /// <param name="upperThreshold">The upper bound. Values above this are clipped down.</param>
    /// <param name="columnIndices">The column indices to clip, or null for all columns.</param>
    public ThresholdClipper(double lowerThreshold, double upperThreshold, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (lowerThreshold > upperThreshold)
        {
            throw new ArgumentException("Lower threshold must be less than or equal to upper threshold.");
        }

        _lowerThreshold = lowerThreshold;
        _upperThreshold = upperThreshold;
    }

    /// <summary>
    /// Records the number of input features. No statistics are computed since thresholds are explicit.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
    }

    /// <summary>
    /// Clips values to the explicit threshold bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with values clipped to bounds.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new Matrix<T>(numRows, numColumns);
        }

        // Clip to bounds on the processed columns; the other columns pass through.
        var (lower, upper) = Bounds(numColumns);
        var clipped = ScalerTensorOps.ClipColumns(Tensor<T>.FromMatrix(data), lower, upper);
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        return ScalerTensorOps.MergeColumns(mask, clipped, data);
    }

    /// <summary>The thresholds as per-column bound vectors.</summary>
    private (Vector<T> Lower, Vector<T> Upper) Bounds(int numColumns) =>
        (ScalerTensorOps.FilledVector(numColumns, NumOps.FromDouble(_lowerThreshold)),
         ScalerTensorOps.FilledVector(numColumns, NumOps.FromDouble(_upperThreshold)));

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ThresholdClipper does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets a boolean mask indicating which values are outside the threshold bounds.
    /// </summary>
    /// <param name="data">The data to check.</param>
    /// <returns>A 2D boolean array where true indicates a value outside bounds.</returns>
    public bool[,] GetOutlierMask(Matrix<T> data)
    {
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new bool[numRows, numColumns];
        }

        var (lower, upper) = Bounds(numColumns);
        var outside = ScalerTensorOps.OutsideMask(Tensor<T>.FromMatrix(data), lower, upper);
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        return ScalerTensorOps.ToBoolMatrix(Engine.TensorMultiply(outside, ScalerTensorOps.BroadcastRow(mask, numRows)));
    }

    /// <summary>
    /// Counts how many values would be clipped at each bound.
    /// </summary>
    /// <param name="data">The data to analyze.</param>
    /// <returns>A tuple of (belowLowerCount, aboveUpperCount) for each column.</returns>
    public (int[] BelowLower, int[] AboveUpper) CountOutliers(Matrix<T> data)
    {
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return (new int[numColumns], new int[numColumns]);
        }

        var x = Tensor<T>.FromMatrix(data);
        var mask = ScalerTensorOps.BroadcastRow(
            ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns)), numRows);
        var below = Engine.TensorMultiply(Engine.TensorLessThan(x, NumOps.FromDouble(_lowerThreshold)), mask);
        var above = Engine.TensorMultiply(Engine.TensorGreaterThan(x, NumOps.FromDouble(_upperThreshold)), mask);
        return (ScalerTensorOps.CountPerColumn(below), ScalerTensorOps.CountPerColumn(above));
    }
}
