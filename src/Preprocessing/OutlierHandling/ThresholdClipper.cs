using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

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
        var result = new T[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                double value = NumOps.ToDouble(data[i, j]);

                if (processSet.Contains(j))
                {
                    // Clip to bounds
                    value = Math.Max(_lowerThreshold, Math.Min(_upperThreshold, value));
                }

                result[i, j] = NumOps.FromDouble(value);
            }
        }

        return new Matrix<T>(result);
    }

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
        var mask = new bool[numRows, numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                if (processSet.Contains(j))
                {
                    double value = NumOps.ToDouble(data[i, j]);
                    mask[i, j] = value < _lowerThreshold || value > _upperThreshold;
                }
            }
        }

        return mask;
    }

    /// <summary>
    /// Counts how many values would be clipped at each bound.
    /// </summary>
    /// <param name="data">The data to analyze.</param>
    /// <returns>A tuple of (belowLowerCount, aboveUpperCount) for each column.</returns>
    public (int[] BelowLower, int[] AboveUpper) CountOutliers(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var belowLower = new int[numColumns];
        var aboveUpper = new int[numColumns];
        var columnsToProcess = GetColumnsToProcess(numColumns);
        var processSet = new HashSet<int>(columnsToProcess);

        for (int i = 0; i < data.Rows; i++)
        {
            for (int j = 0; j < numColumns; j++)
            {
                if (processSet.Contains(j))
                {
                    double value = NumOps.ToDouble(data[i, j]);
                    if (value < _lowerThreshold)
                    {
                        belowLower[j]++;
                    }
                    else if (value > _upperThreshold)
                    {
                        aboveUpper[j]++;
                    }
                }
            }
        }

        return (belowLower, aboveUpper);
    }
}
