using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Preprocessing.OutlierHandling;

/// <summary>
/// Clips outliers using the Interquartile Range (IQR) method.
/// </summary>
/// <remarks>
/// <para>
/// IQRClipper identifies outliers using the IQR method:
/// - Lower bound = Q1 - k * IQR
/// - Upper bound = Q3 + k * IQR
/// where IQR = Q3 - Q1 and k is the multiplier (default 1.5).
/// </para>
/// <para>
/// This is the same method used in box plots for identifying outliers.
/// A multiplier of 1.5 identifies "mild" outliers, while 3.0 identifies "extreme" outliers.
/// </para>
/// <para><b>For Beginners:</b> The IQR is the range where the middle 50% of data falls.
/// Values outside 1.5× this range are considered outliers:
/// - Q1 (25th percentile): 25% of data is below this value
/// - Q3 (75th percentile): 75% of data is below this value
/// - IQR = Q3 - Q1: The spread of the middle 50%
/// - Outliers: Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class IQRClipper<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _multiplier;

    // Fitted parameters
    private Vector<T>? _lowerBounds;
    private Vector<T>? _upperBounds;
    private Vector<T>? _q1Values;
    private Vector<T>? _q3Values;
    private Vector<T>? _iqrValues;

    /// <summary>
    /// Gets the IQR multiplier for determining outlier boundaries.
    /// </summary>
    public double Multiplier => _multiplier;

    /// <summary>
    /// Gets the fitted lower bounds for each feature.
    /// </summary>
    public Vector<T>? LowerBounds => _lowerBounds;

    /// <summary>
    /// Gets the fitted upper bounds for each feature.
    /// </summary>
    public Vector<T>? UpperBounds => _upperBounds;

    /// <summary>
    /// Gets the Q1 (25th percentile) values for each feature.
    /// </summary>
    public Vector<T>? Q1Values => _q1Values;

    /// <summary>
    /// Gets the Q3 (75th percentile) values for each feature.
    /// </summary>
    public Vector<T>? Q3Values => _q3Values;

    /// <summary>
    /// Gets the IQR values for each feature.
    /// </summary>
    public Vector<T>? IQRValues => _iqrValues;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="IQRClipper{T}"/>.
    /// </summary>
    /// <param name="multiplier">The IQR multiplier (default 1.5 for mild outliers, 3.0 for extreme).</param>
    /// <param name="columnIndices">The column indices to clip, or null for all columns.</param>
    public IQRClipper(
        double multiplier = 1.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (multiplier <= 0)
        {
            throw new ArgumentException("Multiplier must be positive.", nameof(multiplier));
        }

        _multiplier = multiplier;
    }

    /// <summary>
    /// Fits the clipper by computing IQR bounds for each feature.
    /// </summary>
    /// <param name="data">The training data matrix.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var mask = ScalerTensorOps.ColumnMask<T>(numColumns, GetColumnsToProcess(numColumns));
        T zero = NumOps.Zero;

        // Q1 / Q3 are the linearly interpolated 25th / 75th percentiles (numpy's default), ignoring NaN; an
        // empty column gets 0.
        Tensor<T> q1;
        Tensor<T> q3;
        if (data.Rows == 0 || numColumns == 0)
        {
            q1 = new Tensor<T>(new[] { numColumns });
            q3 = new Tensor<T>(new[] { numColumns });
        }
        else
        {
            var x = Tensor<T>.FromMatrix(data);
            q1 = Engine.TensorNanQuantile(x, 0.25, 0, false, QuantileInterpolation.Linear);
            q3 = Engine.TensorNanQuantile(x, 0.75, 0, false, QuantileInterpolation.Linear);
        }

        var iqr = Engine.TensorSubtract(q3, q1);
        var spread = Engine.TensorMultiplyScalar(iqr, NumOps.FromDouble(_multiplier));
        _q1Values = ScalerTensorOps.MaskedParameters(mask, q1, zero);
        _q3Values = ScalerTensorOps.MaskedParameters(mask, q3, zero);
        _iqrValues = ScalerTensorOps.MaskedParameters(mask, iqr, zero);

        // Non-processed columns get infinite bounds (no clipping).
        _lowerBounds = ScalerTensorOps.MaskedParameters(
            mask, Engine.TensorSubtract(q1, spread), NumOps.FromDouble(double.NegativeInfinity));
        _upperBounds = ScalerTensorOps.MaskedParameters(
            mask, Engine.TensorAdd(q3, spread), NumOps.FromDouble(double.PositiveInfinity));
    }

    /// <summary>
    /// Transforms the data by clipping values outside IQR bounds.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data with outliers clipped.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("IQRClipper has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _lowerBounds.Length);
        if (data.Rows == 0 || data.Columns == 0)
        {
            return new Matrix<T>(data.Rows, data.Columns);
        }

        return ScalerTensorOps.ClipColumns(Tensor<T>.FromMatrix(data), _lowerBounds, _upperBounds).ToMatrix();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("IQRClipper does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }

    /// <summary>
    /// Gets a mask indicating which values are outliers in the input data.
    /// </summary>
    /// <param name="data">The data to check for outliers.</param>
    /// <returns>A boolean matrix where true indicates an outlier.</returns>
    public bool[,] GetOutlierMask(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("IQRClipper has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _lowerBounds.Length);
        if (data.Rows == 0 || data.Columns == 0)
        {
            return new bool[data.Rows, data.Columns];
        }

        return ScalerTensorOps.ToBoolMatrix(
            ScalerTensorOps.OutsideMask(Tensor<T>.FromMatrix(data), _lowerBounds, _upperBounds));
    }

    /// <summary>
    /// Counts the number of outliers per feature.
    /// </summary>
    /// <param name="data">The data to analyze.</param>
    /// <returns>Array of outlier counts per feature.</returns>
    public int[] CountOutliersPerFeature(Matrix<T> data)
    {
        if (_lowerBounds is null || _upperBounds is null)
        {
            throw new InvalidOperationException("IQRClipper has not been fitted.");
        }

        ScalerTensorOps.EnsureColumnCount(data, _lowerBounds.Length);
        if (data.Rows == 0 || data.Columns == 0)
        {
            return new int[data.Columns];
        }

        return ScalerTensorOps.CountPerColumn(
            ScalerTensorOps.OutsideMask(Tensor<T>.FromMatrix(data), _lowerBounds, _upperBounds));
    }
}
