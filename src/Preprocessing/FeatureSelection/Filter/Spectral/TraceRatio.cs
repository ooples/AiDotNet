using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// Trace Ratio criterion for multi-class feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Trace Ratio maximizes the ratio of between-class scatter to within-class
/// scatter. Unlike Fisher Score which works per feature, Trace Ratio considers
/// feature interactions through scatter matrices.
/// </para>
/// <para>
/// The subset is found with the iterative algorithm of Nie, Xiang, Jia, Zhang and Yan (2008), "Trace Ratio
/// Criterion for Feature Selection" (AAAI): with lambda the trace ratio of the current subset, the next subset is
/// the k features with the largest <c>Sb - lambda Sw</c>, repeated until the subset stops changing. Every change
/// raises lambda, and the loop ends at the k-subset with the globally largest trace ratio.
/// </para>
/// <para><b>For Beginners:</b> This extends Fisher Score to consider how features
/// work together. Instead of scoring each feature independently, it evaluates
/// how a set of features collectively separates classes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TraceRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    // Safety cap on the subset updates; each one strictly raises the trace ratio, so the loop ends long before.
    private const int MaxIterations = 100;

    private readonly int _nFeaturesToSelect;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;
    private T _traceRatioValue = MathHelper.GetNumericOperations<T>().Zero;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    public T TraceRatioValue => _traceRatioValue;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TraceRatio(int nFeaturesToSelect = 10, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TraceRatio requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int k = Math.Min(_nFeaturesToSelect, p);

        // Diagonal between-class (Sb) and within-class (Sw) scatter of every feature, classes = distinct targets.
        // Half and BFloat16 are evaluated in float: the trace ratio mixes the features' scatters, so they cannot be
        // rescaled one by one, and the raw scatters overflow Half once the deviations pass a few hundred (Sw = inf
        // scored a separating feature 0)
        var x = FilterStatisticTensorOps.AsTensor(data);
        var classOneHot = StatisticalTestHelper.ClassOneHot(target, out _);
        if (PreprocessingTensorOps.IsNarrowFloat<T>())
        {
            var (scores, selected, lambda) = Solve(x.Cast<float>(), classOneHot.Cast<float>(), k);
            _featureScores = scores.Cast<T>().ToVector();
            if (k > 0)
                _traceRatioValue = NumOps.FromDouble(lambda);
            _selectedIndices = selected;
        }
        else
        {
            var (scores, selected, lambda) = Solve(x, classOneHot, k);
            _featureScores = scores.ToVector();
            if (k > 0)
                _traceRatioValue = NumOps.FromDouble(lambda);
            _selectedIndices = selected;
        }

        IsFitted = true;
    }

    // The single-feature scores Sb / Sw (in TN), the selected subset and its trace ratio (Nie et al. 2008); the
    // ratio and the search over Sb - lambda Sw run in double.
    private static (Tensor<TN> Scores, int[] Selected, double Lambda) Solve<TN>(Tensor<TN> x, Tensor<TN> classOneHot, int k)
    {
        var engine = AiDotNetEngine.Current;
        var (sb, sw) = FilterStatisticTensorOps.AnovaSums(x, classOneHot);

        // Feature scores Sb / Sw; with no within-class scatter a feature whose class means differ separates the
        // classes exactly (+infinity), a constant one scores 0
        var scores = FilterStatisticTensorOps.SeparationAwareRatio(sb, sw, 1e-10);

        // Nie et al. (2008): start from the k best single-feature ratios, then repeat lambda = sum_S Sb / sum_S Sw and
        // S = the k largest Sb - lambda Sw (ties to the lower index) until S stops changing or lambda stops rising.
        // A greedy forward search (add the feature that raises the ratio most) misses the optimum, e.g. Sb = [3, 2,
        // 1, 6], Sw = [7, 6, 5, 1], k = 3: greedy keeps {0, 2, 3} (10 / 13), the optimum is {0, 1, 3} (11 / 14).
        var selected = PreprocessingTensorOps.TopKIndicesSorted(scores, k);
        var sbValues = sb.Cast<double>();
        var swValues = sw.Cast<double>();
        double lambda = SubsetRatio(sb, sw, selected);
        for (int iteration = 0; iteration < MaxIterations && k > 0; iteration++)
        {
            // an infinite ratio (no within-class scatter, some between-class scatter) cannot be beaten; NaN stops too
            if (!(lambda < double.PositiveInfinity))
                break;

            var next = PreprocessingTensorOps.TopKIndicesSorted(
                engine.TensorSubtract(sbValues, engine.TensorMultiplyScalar(swValues, lambda)), k);
            if (next.SequenceEqual(selected))
                break;

            double nextLambda = SubsetRatio(sb, sw, next);
            if (!(nextLambda > lambda))
                break;

            selected = next;
            lambda = nextLambda;
        }

        return (scores, selected, lambda);
    }

    // The trace ratio sum(Sb) / sum(Sw) of a feature subset: +infinity when the subset has between-class scatter but
    // no within-class scatter, 0 when it has neither.
    private static double SubsetRatio<TN>(Tensor<TN> sb, Tensor<TN> sw, int[] subset)
    {
        var engine = AiDotNetEngine.Current;
        var numerator = engine.Reshape(engine.ReduceSum(FeatureScoringTensorOps.Entries(sb, subset), new[] { 0 }, false), new[] { 1 });
        var denominator = engine.Reshape(engine.ReduceSum(FeatureScoringTensorOps.Entries(sw, subset), new[] { 0 }, false), new[] { 1 });
        return FilterStatisticTensorOps.SeparationAwareRatio(numerator, denominator, 1e-10).Cast<double>().ToArray()[0];
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
            throw new InvalidOperationException("TraceRatio has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TraceRatio does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TraceRatio has not been fitted.");

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
