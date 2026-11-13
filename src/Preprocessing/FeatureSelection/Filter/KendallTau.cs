using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Kendall Tau correlation for feature selection based on concordant pairs.
/// </summary>
/// <remarks>
/// <para>
/// Kendall's Tau measures the ordinal association between features and target by
/// comparing pairs of observations. It counts concordant pairs (both values increase
/// or both decrease) versus discordant pairs (one increases while the other decreases).
/// </para>
/// <para><b>For Beginners:</b> Kendall Tau looks at every pair of data points and asks:
/// when the feature goes up, does the target also go up? If most pairs agree (concordant),
/// the correlation is high. Unlike Pearson, it doesn't assume linear relationships and
/// is very robust to outliers. It's especially good for ordinal or ranked data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class KendallTau<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minCorrelation;

    private Vector<T>? _correlations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? Correlations => _correlations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public KendallTau(
        int nFeaturesToSelect = 10,
        double minCorrelation = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minCorrelation = minCorrelation;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KendallTau requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        // pair counts and sign sums run in double, as master did: they pass 2048 (n > 45) in Half, and the 1e-10
        // tie tolerance is 0 in Half, which made no pair tied
        var x = FilterStatisticTensorOps.AsTensor(data).Cast<double>();
        var y = FilterStatisticTensorOps.AsTensor(target).Cast<double>();
        const double tolerance = 1e-10;

        // the n x n pair grid runs one block of rows at a time and is reduced at once, so only a block x n slice is
        // ever held (four full [n, n] grids of double were 3.2 GB at 10000 rows)
        var signSums = new double[p];
        var tieSums = new double[p];
        double yTiedSum = 0;
        foreach (var block in FeatureSelectionTensorOps.RowBlocks(Enumerable.Range(0, n).ToArray(), n))
        {
            int start = block[0];
            var (yTied, ySign) = PairSigns(y, start, block.Length, tolerance);
            yTiedSum += Engine.TensorSum(yTied);
            for (int j = 0; j < p; j++)
            {
                var (xTied, xSign) = PairSigns(PreprocessingTensorOps.ColumnVector(x, j), start, block.Length, tolerance);
                signSums[j] += Engine.TensorSum(Engine.TensorMultiply(xSign, ySign));
                tieSums[j] += Engine.TensorSum(xTied);
            }
        }

        double pairs = (double)n * (n - 1) / 2;
        double yTies = (yTiedSum - n) * 0.5;
        var xTies = Engine.TensorMultiplyScalar(Engine.TensorSubtractScalar(new Tensor<double>(tieSums, new[] { p }), (double)n), 0.5);
        var numerator = Engine.TensorMultiplyScalar(new Tensor<double>(signSums, new[] { p }), 0.5);
        var untiedX = Engine.TensorNegate(Engine.TensorSubtractScalar(xTies, pairs));
        var denominator = Engine.TensorMultiplyScalar(Engine.TensorSqrt(untiedX), Math.Sqrt(pairs - yTies));
        var correlations = FilterStatisticTensorOps.GuardedDivide(numerator, denominator, 1e-10, 0.0).Cast<T>();
        _correlations = correlations.ToVector();
        _selectedIndices = FilterStatisticTensorOps.CandidatesOrTopK(
            Engine.TensorAbs(correlations), NumOps.FromDouble(_minCorrelation), Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    // For the rows i in [start, start + count) and every k of a 1-D tensor: 1 where |v[k] - v[i]| < tolerance, and
    // the sign of v[k] - v[i] (0 when tied); [count, n].
    private (Tensor<double> Tied, Tensor<double> Sign) PairSigns(Tensor<double> values, int start, int count, double tolerance)
    {
        int n = values.Length;
        var difference = Engine.TensorSubtract(
            FilterStatisticTensorOps.RowBroadcast(values, count),
            FilterStatisticTensorOps.ColumnBroadcast(Engine.TensorNarrow(values, 0, start, count), n));
        var tied = Engine.TensorLessThan(Engine.TensorAbs(difference), tolerance);
        return (tied, Engine.TensorMultiply(Engine.TensorSign(difference), PreprocessingTensorOps.OneMinus(tied)));
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KendallTau has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("KendallTau does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KendallTau has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
