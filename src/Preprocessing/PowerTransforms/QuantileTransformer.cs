using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.Scalers;

namespace AiDotNet.Preprocessing.PowerTransforms;

/// <summary>
/// Specifies the target output distribution for quantile transformation.
/// </summary>
public enum OutputDistributionType
{
    /// <summary>
    /// Map to a uniform distribution in [0, 1].
    /// </summary>
    Uniform,

    /// <summary>
    /// Map to a standard normal distribution (Gaussian).
    /// </summary>
    Normal
}

/// <summary>
/// Transforms features to follow a uniform or normal distribution using quantile information.
/// </summary>
/// <remarks>
/// <para>
/// QuantileTransformer applies a non-linear transformation that maps the input distribution to
/// either a uniform or normal (Gaussian) distribution. This is effective at reducing the impact
/// of outliers and normalizing non-Gaussian distributions.
/// </para>
/// <para><b>For Beginners:</b> This transformer redistributes your data to match a desired pattern:
/// - Uniform: Spreads values evenly across [0, 1]
/// - Normal: Creates a bell curve distribution
///
/// Example with uniform output:
/// [1, 1, 2, 3, 5, 8, 13, 100, 1000] → [0.0, 0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
/// (values equal to the fitted minimum map to 0, as in scikit-learn)
/// Notice how extreme values (100, 1000) no longer dominate.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class QuantileTransformer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly OutputDistributionType _outputDistribution;
    private readonly int _nQuantiles;

    // Fitted parameters: quantiles for each column
    private List<T[]>? _quantiles;

    /// <summary>
    /// Gets the target output distribution.
    /// </summary>
    public OutputDistributionType OutputDistribution => _outputDistribution;

    /// <summary>
    /// Gets the number of quantiles used.
    /// </summary>
    public int NQuantiles => _nQuantiles;

    /// <summary>
    /// Gets the quantiles for each feature computed during fitting.
    /// </summary>
    public List<T[]>? Quantiles => _quantiles;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => true;

    /// <summary>
    /// Creates a new instance of <see cref="QuantileTransformer{T}"/>.
    /// </summary>
    /// <param name="outputDistribution">The target distribution. Defaults to Uniform.</param>
    /// <param name="nQuantiles">The number of quantiles to compute. Defaults to 1000.</param>
    /// <param name="columnIndices">The column indices to transform, or null for all columns.</param>
    public QuantileTransformer(
        OutputDistributionType outputDistribution = OutputDistributionType.Uniform,
        int nQuantiles = 1000,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nQuantiles < 10)
        {
            throw new ArgumentException("nQuantiles must be at least 10.", nameof(nQuantiles));
        }

        _outputDistribution = outputDistribution;
        _nQuantiles = nQuantiles;
    }

    /// <summary>
    /// Computes the quantiles for each feature from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        int numColumns = data.Columns;
        var columnsToProcess = GetColumnsToProcess(numColumns);

        // scikit-learn: n_quantiles_ = max(1, min(n_quantiles, n_samples)), references_ = linspace(0, 1, n_quantiles_),
        // quantiles_ = nanpercentile(X, references_ * 100) (linear) made non-decreasing (maximum.accumulate).
        // All samples are used (subsample=None).
        int nQuantiles = Math.Max(1, Math.Min(_nQuantiles, data.Rows));
        var references = References(nQuantiles);
        var x = data.Rows == 0 ? null : Tensor<T>.FromMatrix(data);
        T nan = NumOps.FromDouble(double.NaN);

        _quantiles = new List<T[]>();
        for (int col = 0; col < numColumns; col++)
        {
            if (!columnsToProcess.Contains(col))
            {
                _quantiles.Add(Array.Empty<T>());
                continue;
            }

            var quantiles = x is null
                ? PreprocessingTensorOps.Filled(new[] { nQuantiles }, nan)
                : ColumnQuantiles(PreprocessingTensorOps.ColumnVector(x, col), references);
            _quantiles.Add(quantiles.AsSpan().ToArray());
        }
    }

    /// <summary>numpy's <c>linspace(0, 1, count)</c> as a [count] tensor.</summary>
    private Tensor<T> References(int count)
    {
        if (count == 1)
        {
            return new Tensor<T>(new[] { 1 });
        }

        var references = Engine.TensorMultiplyScalar(PreprocessingTensorOps.Range<T>(count), NumOps.FromDouble(1.0 / (count - 1)));
        return Engine.TensorSetSlice(references, PreprocessingTensorOps.Filled(new[] { 1 }, NumOps.One), new[] { count - 1 });
    }

    /// <summary>
    /// numpy's <c>nanpercentile</c> (method "linear") of a column at the given references, made non-decreasing:
    /// virtual index <c>h = q (n - 1)</c> over the sorted non-NaN values, interpolated as numpy's <c>_lerp</c>.
    /// </summary>
    private Tensor<T> ColumnQuantiles(Tensor<T> column, Tensor<T> references)
    {
        var valid = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(ScalerTensorOps.NanMask(column)));
        int count = valid.Length;
        if (count == 0)
        {
            return PreprocessingTensorOps.Filled(new[] { references.Length }, NumOps.FromDouble(double.NaN));
        }

        var sorted = Engine.TensorSort(
            Engine.TensorIndexSelect(column, new Tensor<int>(valid, new[] { count }), 0), 0, false).Item1;
        var position = Engine.TensorMultiplyScalar(references, NumOps.FromDouble(count - 1));
        var below = Engine.TensorFloor(position);
        var fraction = Engine.TensorSubtract(position, below);
        var belowIndex = below.Cast<int>().AsSpan().ToArray();
        var aboveIndex = belowIndex.Select(index => Math.Min(index + 1, count - 1)).ToArray();
        var lower = Engine.TensorIndexSelect(sorted, new Tensor<int>(belowIndex, new[] { belowIndex.Length }), 0);
        var upper = Engine.TensorIndexSelect(sorted, new Tensor<int>(aboveIndex, new[] { aboveIndex.Length }), 0);

        // numpy _lerp: a + (b - a) t, computed as b - (b - a)(1 - t) when t >= 0.5
        var diff = Engine.TensorSubtract(upper, lower);
        var fromBelow = Engine.TensorAdd(lower, Engine.TensorMultiply(diff, fraction));
        var fromAbove = Engine.TensorSubtract(upper, Engine.TensorMultiply(diff, PreprocessingTensorOps.OneMinus(fraction)));
        var half = PreprocessingTensorOps.Filled(new[] { references.Length }, NumOps.FromDouble(0.5));
        var quantiles = Engine.TensorWhere(Engine.TensorGreaterEqual(fraction, half), fromAbove, fromBelow);
        return Engine.TensorCumMax(quantiles, 0);
    }

    /// <summary>
    /// Transforms the data by mapping to the target distribution.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data following the target distribution.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_quantiles is null)
        {
            throw new InvalidOperationException("Transformer has not been fitted.");
        }

        return MapColumns(data, inverse: false);
    }

    /// <summary>
    /// Applies scikit-learn's <c>_transform_col</c> to every processed, fitted column; the other columns pass
    /// through unchanged.
    /// </summary>
    private Matrix<T> MapColumns(Matrix<T> data, bool inverse)
    {
        var fitted = _quantiles ?? new List<T[]>();
        int numRows = data.Rows;
        int numColumns = data.Columns;
        if (numRows == 0 || numColumns == 0)
        {
            return new Matrix<T>(numRows, numColumns);
        }

        var x = Tensor<T>.FromMatrix(data);
        var processSet = new HashSet<int>(GetColumnsToProcess(numColumns));
        var columns = new List<Tensor<T>>(numColumns);
        for (int col = 0; col < numColumns; col++)
        {
            var column = PreprocessingTensorOps.ColumnVector(x, col);
            bool mapped = processSet.Contains(col) && col < fitted.Count && fitted[col].Length > 0;
            columns.Add(mapped ? MapColumn(column, fitted[col], inverse) : column);
        }

        return PreprocessingTensorOps.StackColumns(columns, numRows);
    }

    /// <summary>
    /// scikit-learn's <c>QuantileTransformer._transform_col</c>. Forward: the mean of the interpolations from
    /// both ends, <c>0.5 (interp(x, q, r) - interp(-x, -q[::-1], -r[::-1]))</c>, with values at (uniform) or
    /// within 1e-7 beyond (normal) the bounds set to 0 / 1, then <c>norm.ppf</c> clipped to
    /// <c>norm.ppf(1e-7 - eps)</c> for normal output. Inverse: <c>norm.cdf</c> for normal output, then
    /// <c>interp(x, r, q)</c> with the same bound rules. NaN passes through.
    /// </summary>
    private Tensor<T> MapColumn(Tensor<T> column, T[] quantileValues, bool inverse)
    {
        int count = quantileValues.Length;
        var quantiles = new Tensor<T>((T[])quantileValues.Clone(), new[] { count });
        var references = References(count);
        bool normal = _outputDistribution == OutputDistributionType.Normal;

        var nanMask = ScalerTensorOps.NanMask(column);
        var x = ScalerTensorOps.FillNan(column, NumOps.Zero);
        if (inverse && normal)
        {
            // norm.cdf(x) = (1 + erf(x / sqrt(2))) / 2
            var erf = Engine.TensorErf(Engine.TensorMultiplyScalar(x, NumOps.FromDouble(1.0 / Math.Sqrt(2.0))));
            x = Engine.TensorMultiplyScalar(Engine.TensorAddScalar(erf, NumOps.One), NumOps.FromDouble(0.5));
        }

        T lowerX = inverse ? NumOps.Zero : quantileValues[0];
        T upperX = inverse ? NumOps.One : quantileValues[count - 1];
        T lowerY = inverse ? quantileValues[0] : NumOps.Zero;
        T upperY = inverse ? quantileValues[count - 1] : NumOps.One;
        var threshold = NumOps.FromDouble(BoundsThreshold);
        var lowerMask = normal
            ? Engine.TensorLessThan(Engine.TensorSubtractScalar(x, threshold), lowerX)
            : Engine.TensorEquals(x, lowerX);
        var upperMask = normal
            ? Engine.TensorGreaterThan(Engine.TensorAddScalar(x, threshold), upperX)
            : Engine.TensorEquals(x, upperX);

        Tensor<T> y;
        if (inverse)
        {
            y = Interp(x, references, quantiles);
        }
        else
        {
            var forward = Interp(x, quantiles, references);
            var backward = Interp(Engine.TensorNegate(x), Engine.TensorNegate(Reverse(quantiles)), Engine.TensorNegate(Reverse(references)));
            y = Engine.TensorMultiplyScalar(Engine.TensorSubtract(forward, backward), NumOps.FromDouble(0.5));
        }

        int n = column.Length;
        y = Engine.TensorWhere(upperMask, PreprocessingTensorOps.Filled(new[] { n }, upperY), y);
        y = Engine.TensorWhere(lowerMask, PreprocessingTensorOps.Filled(new[] { n }, lowerY), y);

        if (!inverse && normal)
        {
            var clipBounds = Engine.TensorNdtri(new Tensor<T>(
                new[] { NumOps.FromDouble(BoundsThreshold - Float64Spacing), NumOps.FromDouble(1 - (BoundsThreshold - Float64Spacing)) },
                new[] { 2 })).AsSpan().ToArray();
            y = Engine.TensorClamp(Engine.TensorNdtri(y), clipBounds[0], clipBounds[1]);
        }

        return Engine.TensorWhere(nanMask, column, y);
    }

    /// <summary>
    /// numpy's <c>interp(x, xp, fp)</c> for an increasing <paramref name="xp"/>: <c>fp[0]</c> below the range,
    /// <c>fp[-1]</c> at or above its end, otherwise the line through the bracketing points
    /// <c>xp[j] &lt;= x &lt; xp[j + 1]</c>.
    /// </summary>
    private Tensor<T> Interp(Tensor<T> x, Tensor<T> xp, Tensor<T> fp)
    {
        int n = x.Length;
        int m = xp.Length;
        var points = xp.AsSpan().ToArray();
        var values = fp.AsSpan().ToArray();
        if (m == 1)
        {
            return PreprocessingTensorOps.Filled(new[] { n }, values[0]);
        }

        var right = Engine.TensorSearchSorted(xp, x, true).AsSpan().ToArray();
        var lowerIndex = right.Select(index => Math.Min(Math.Max(index - 1, 0), m - 2)).ToArray();
        var upperIndex = lowerIndex.Select(index => index + 1).ToArray();
        var lowerTensor = new Tensor<int>(lowerIndex, new[] { n });
        var upperTensor = new Tensor<int>(upperIndex, new[] { n });
        var x0 = Engine.TensorIndexSelect(xp, lowerTensor, 0);
        var x1 = Engine.TensorIndexSelect(xp, upperTensor, 0);
        var f0 = Engine.TensorIndexSelect(fp, lowerTensor, 0);
        var f1 = Engine.TensorIndexSelect(fp, upperTensor, 0);

        var dx = Engine.TensorSubtract(x1, x0);
        var safeDx = Engine.TensorWhere(Engine.TensorEquals(dx, NumOps.Zero), PreprocessingTensorOps.Filled(new[] { n }, NumOps.One), dx);
        var slope = Engine.TensorDivide(Engine.TensorSubtract(f1, f0), safeDx);
        var y = Engine.TensorAdd(Engine.TensorMultiply(slope, Engine.TensorSubtract(x, x0)), f0);

        y = Engine.TensorWhere(Engine.TensorLessThan(x, points[0]), PreprocessingTensorOps.Filled(new[] { n }, values[0]), y);
        var atOrAboveEnd = PreprocessingTensorOps.Not(Engine.TensorLessThan(x, points[m - 1]));
        return Engine.TensorWhere(atOrAboveEnd, PreprocessingTensorOps.Filled(new[] { n }, values[m - 1]), y);
    }

    /// <summary>scikit-learn's <c>BOUNDS_THRESHOLD</c>.</summary>
    private const double BoundsThreshold = 1e-7;

    /// <summary>numpy's <c>spacing(1)</c> (the float64 machine epsilon).</summary>
    private const double Float64Spacing = 2.220446049250313e-16;

    /// <summary>The elements of a 1-D tensor in reverse order.</summary>
    private Tensor<T> Reverse(Tensor<T> values)
    {
        var reversed = values.AsSpan().ToArray();
        Array.Reverse(reversed);
        return new Tensor<T>(reversed, new[] { reversed.Length });
    }

    /// <summary>
    /// Reverses the quantile transformation.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>The original-scale data.</returns>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        if (_quantiles is null)
        {
            throw new InvalidOperationException("Transformer has not been fitted.");
        }

        return MapColumns(data, inverse: true);
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The same feature names (QuantileTransformer doesn't change number of features).</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}
