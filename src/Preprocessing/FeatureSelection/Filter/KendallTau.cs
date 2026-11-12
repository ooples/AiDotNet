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
        var x = FilterStatisticTensorOps.AsTensor(data);
        T tolerance = NumOps.FromDouble(1e-10);
        T half = NumOps.FromDouble(0.5);
        var (yTied, ySign) = PairSigns(FilterStatisticTensorOps.AsTensor(target), tolerance);
        var signSums = new T[p];
        var tieSums = new T[p];
        for (int j = 0; j < p; j++)
        {
            var (xTied, xSign) = PairSigns(FeatureSelectionTensorOps.Column(x, j), tolerance);
            signSums[j] = Engine.TensorSum(Engine.TensorMultiply(xSign, ySign));
            tieSums[j] = Engine.TensorSum(xTied);
        }

        T pairs = NumOps.FromDouble((double)n * (n - 1) / 2);
        T diagonal = NumOps.FromDouble(n);
        T yTies = NumOps.Multiply(NumOps.Subtract(Engine.TensorSum(yTied), diagonal), half);
        var xTies = Engine.TensorMultiplyScalar(
            Engine.TensorSubtractScalar(new Tensor<T>(tieSums, new[] { p }), diagonal), half);
        var numerator = Engine.TensorMultiplyScalar(new Tensor<T>(signSums, new[] { p }), half);
        var untiedX = Engine.TensorNegate(Engine.TensorSubtractScalar(xTies, pairs));
        var denominator = Engine.TensorSqrt(Engine.TensorMultiplyScalar(untiedX, NumOps.Subtract(pairs, yTies)));
        var correlations = FilterStatisticTensorOps.GuardedDivide(numerator, denominator, 1e-10, NumOps.Zero);
        _correlations = correlations.ToVector();
        _selectedIndices = FilterStatisticTensorOps.CandidatesOrTopK(
            Engine.TensorAbs(correlations), NumOps.FromDouble(_minCorrelation), Math.Min(_nFeaturesToSelect, p));
        IsFitted = true;
    }

    // For every pair (i, k) of a 1-D tensor: 1 where |v[k] - v[i]| < tolerance, and the sign of v[k] - v[i] (0 when tied).
    private (Tensor<T> Tied, Tensor<T> Sign) PairSigns(Tensor<T> values, T tolerance)
    {
        int n = values.Length;
        var difference = Engine.TensorSubtract(
            FilterStatisticTensorOps.RowBroadcast(values, n), FilterStatisticTensorOps.ColumnBroadcast(values, n));
        var tied = Engine.TensorLessThan(Engine.TensorAbs(difference), tolerance);
        return (tied, Engine.TensorMultiply(Engine.TensorSign(difference), FilterStatisticTensorOps.OneMinus(tied)));
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
