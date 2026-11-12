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
        var engine = Engine;

        // Diagonal between-class (Sb) and within-class (Sw) scatter of every feature, classes = distinct targets
        var classOneHot = StatisticalTestHelper.ClassOneHot(target, out _);
        var (sb, sw) = FilterStatisticTensorOps.AnovaSums(FilterStatisticTensorOps.AsTensor(data), classOneHot);

        // Feature scores Sb / Sw; with no within-class scatter a feature whose class means differ separates the
        // classes exactly (+infinity), a constant one scores 0
        var scores = FilterStatisticTensorOps.SeparationAwareRatio(sb, sw, 1e-10);
        _featureScores = scores.ToVector();

        // Greedy selection to maximize the trace ratio sum(Sb) / sum(Sw) of the selected set (first feature on ties)
        T negativeInfinity = NumOps.FromDouble(double.NegativeInfinity);
        var taken = new Tensor<T>(new[] { p });
        var sbSelected = new Tensor<T>(new[] { p });
        var swSelected = new Tensor<T>(new[] { p });
        var selected = new List<int>();
        var bestRatio = new Tensor<T>(new[] { 1 });
        while (selected.Count < _nFeaturesToSelect && selected.Count < p)
        {
            var ratio = FilterStatisticTensorOps.SeparationAwareRatio(
                engine.TensorAdd(sbSelected, sb), engine.TensorAdd(swSelected, sw), 1e-10);
            var candidates = FeatureSelectionTensorOps.KeepOrFill(ratio, PreprocessingTensorOps.OneMinus(taken), negativeInfinity);
            int bestFeature = FilterStatisticTensorOps.FirstArgMax(candidates);
            var chosen = new[] { bestFeature };
            selected.Add(bestFeature);
            bestRatio = FeatureSelectionTensorOps.Entries(candidates, chosen);
            taken = engine.TensorAdd(taken, engine.Reshape(engine.TensorOneHot<T>(new Tensor<int>(chosen, new[] { 1 }), p), new[] { p }));
            sbSelected = engine.TensorBroadcastTo(engine.TensorAdd(
                FeatureSelectionTensorOps.Entries(sbSelected, chosen), FeatureSelectionTensorOps.Entries(sb, chosen)), new[] { p });
            swSelected = engine.TensorBroadcastTo(engine.TensorAdd(
                FeatureSelectionTensorOps.Entries(swSelected, chosen), FeatureSelectionTensorOps.Entries(sw, chosen)), new[] { p });
        }

        if (selected.Count > 0)
            _traceRatioValue = engine.TensorSum(bestRatio);

        _selectedIndices = selected.OrderBy(x => x).ToArray();

        IsFitted = true;
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
