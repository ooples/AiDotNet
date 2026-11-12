using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Ensemble Feature Selection combining multiple selection methods.
/// </summary>
/// <remarks>
/// <para>
/// Ensemble Feature Selection aggregates results from multiple feature selection
/// methods. Features consistently selected across methods are more likely to be
/// truly important.
/// </para>
/// <para>
/// Aggregation methods:
/// - Voting: Select features chosen by majority of methods
/// - Ranking: Aggregate rankings using Borda count or similar
/// - Weighted: Weight methods by their reliability
/// </para>
/// <para><b>For Beginners:</b> Different selection methods have different strengths.
/// By combining them, we get more robust and reliable feature selection. Features
/// that multiple methods agree on are more trustworthy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class EnsembleFeatureSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _votingThreshold;
    private readonly AggregationMethod _method;
    private readonly List<Func<Matrix<T>, Vector<T>, int[]>> _selectors;

    private Vector<T>? _aggregateScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public enum AggregationMethod { Voting, BordaCount, WeightedVoting }

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? AggregateScores => _aggregateScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public EnsembleFeatureSelector(
        int nFeaturesToSelect = 10,
        double votingThreshold = 0.5,
        AggregationMethod method = AggregationMethod.BordaCount,
        List<Func<Matrix<T>, Vector<T>, int[]>>? selectors = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (votingThreshold <= 0 || votingThreshold > 1)
            throw new ArgumentException("Voting threshold must be between 0 and 1.", nameof(votingThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _votingThreshold = votingThreshold;
        _method = method;
        _selectors = selectors ?? new List<Func<Matrix<T>, Vector<T>, int[]>>();
    }

    /// <summary>
    /// Adds a selector function to the ensemble.
    /// </summary>
    public void AddSelector(Func<Matrix<T>, Vector<T>, int[]> selector)
    {
        _selectors.Add(selector);
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "EnsembleFeatureSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // If no selectors, use default ones
        if (_selectors.Count == 0)
            AddDefaultSelectors();

        // Run all selectors
        var allSelections = new List<int[]>();
        foreach (var selector in _selectors)
        {
            try
            {
                var selected = selector(data, target);
                if (selected.Length > 0)
                    allSelections.Add(selected);
            }
            catch
            {
                // Skip failed selectors
            }
        }

        if (allSelections.Count == 0)
        {
            // Fallback: select first n features
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            _aggregateScores = new Vector<T>(p);
            IsFitted = true;
            return;
        }

        // Aggregate based on method
        switch (_method)
        {
            case AggregationMethod.Voting:
                AggregateByVoting(allSelections, p);
                break;
            case AggregationMethod.BordaCount:
                AggregateByBordaCount(allSelections, p);
                break;
            case AggregationMethod.WeightedVoting:
                AggregateByWeightedVoting(allSelections, p);
                break;
        }

        IsFitted = true;
    }

    private void AddDefaultSelectors()
    {
        int k = _nFeaturesToSelect;

        // Correlation-based selector (absolute Pearson correlation)
        _selectors.Add((data, target) => RankedTop(Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
            Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target))), k));

        // Variance-based selector (population variance)
        _selectors.Add((data, target) =>
        {
            var x = Tensor<T>.FromMatrix(data);
            var centered = Engine.TensorSubtract(x,
                Engine.TensorBroadcastTo(Engine.ReduceMean(x, new[] { 0 }, true), new[] { data.Rows, data.Columns }));
            return RankedTop(Engine.ReduceMean(Engine.TensorSquare(centered), new[] { 0 }, false), k);
        });

        // Mutual information approximation
        _selectors.Add((data, target) => RankedTop(GaussianMutualInformation(data, target), k));
    }

    /// <summary>
    /// Gaussian mutual information proxy <c>-0.5 * ln(1 - r^2 + 1e-10)</c> per feature, 0 when the feature's or the
    /// target's centred sum of squares is not above 1e-10.
    /// </summary>
    private Tensor<T> GaussianMutualInformation(Matrix<T> data, Vector<T> target)
    {
        var engine = Engine;
        int p = data.Columns;
        T floor = PreprocessingTensorOps.Tolerance<T>(1e-10);
        var x = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);
        var r = PreprocessingTensorOps.TargetCorrelations(x, y);
        var sxx = FeatureSelectionTensorOps.ColumnSumSquaredDeviations(x);
        var zeros = new Tensor<T>(new[] { p });
        if (!PreprocessingTensorOps.IsAbove(FeatureSelectionTensorOps.SumSquaredDeviations(y), floor))
            return zeros;

        var mi = engine.TensorMultiplyScalar(engine.TensorLog(engine.TensorAddScalar(
            PreprocessingTensorOps.OneMinus(engine.TensorSquare(r)), floor)), NumOps.FromDouble(-0.5));
        return engine.TensorWhere(engine.TensorGreaterThan(sxx, floor), mi, zeros);
    }

    /// <summary>The (at most) <paramref name="k"/> best features, best first (stable on ties).</summary>
    private static int[] RankedTop(Tensor<T> scores, int k) =>
        PreprocessingTensorOps.StableOrder(scores, descending: true).Take(k).ToArray();

    /// <summary>
    /// Sums, over the in-range entries of <paramref name="selection"/>, the weight of each entry's rank
    /// (<paramref name="rankWeights"/>, or 1 when null) into a [p] tensor.
    /// </summary>
    private Tensor<T> SelectionScores(int[] selection, int p, Tensor<T>? rankWeights)
    {
        var ranks = Enumerable.Range(0, selection.Length).Where(r => selection[r] >= 0 && selection[r] < p).ToArray();
        if (ranks.Length == 0)
            return new Tensor<T>(new[] { p });

        var features = ranks.Select(r => selection[r]).ToArray();
        var oneHot = Engine.TensorOneHot<T>(new Tensor<int>(features, new[] { features.Length }), p);
        if (rankWeights is null)
            return Engine.ReduceSum(oneHot, new[] { 0 }, false);

        var weights = Engine.TensorIndexSelect(rankWeights, new Tensor<int>(ranks, new[] { ranks.Length }), 0);
        return Engine.ReduceSum(Engine.TensorMultiply(oneHot, Engine.TensorBroadcastTo(
            Engine.Reshape(weights, new[] { ranks.Length, 1 }), new[] { ranks.Length, p })), new[] { 0 }, false);
    }

    private void AggregateByVoting(List<int[]> allSelections, int p)
    {
        var votes = new Tensor<T>(new[] { p });
        foreach (var selection in allSelections)
            votes = Engine.TensorAdd(votes, SelectionScores(selection, p, null));

        _aggregateScores = Engine.TensorDivideScalar(votes, NumOps.FromDouble(allSelections.Count)).ToVector();

        int threshold = (int)Math.Ceiling(_votingThreshold * allSelections.Count);
        var passing = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            PreprocessingTensorOps.AtLeast(votes, NumOps.FromDouble(threshold))));
        var order = PreprocessingTensorOps.StableOrder(votes, descending: true);
        var selected = order
            .Where(idx => passing.Contains(idx))
            .Take(_nFeaturesToSelect)
            .ToList();

        if (selected.Count == 0)
        {
            selected = order
                .Take(_nFeaturesToSelect)
                .ToList();
        }

        _selectedIndices = selected.OrderBy(x => x).ToArray();
    }

    private void AggregateByBordaCount(List<int[]> allSelections, int p)
    {
        // A selection of length L gives its rank-r entry L - r points
        var bordaScores = new Tensor<T>(new[] { p });

        foreach (var selection in allSelections)
        {
            int length = selection.Length;
            var points = Engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { length }, PreprocessingTensorOps.CountAsScalar<T>(length)),
                PreprocessingTensorOps.Range<T>(length));
            bordaScores = Engine.TensorAdd(bordaScores, SelectionScores(selection, p, points));
        }

        _aggregateScores = bordaScores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(bordaScores, Math.Min(_nFeaturesToSelect, p));
    }

    private void AggregateByWeightedVoting(List<int[]> allSelections, int p)
    {
        // Equal weights for now (can be extended with method weights)
        T weight = NumOps.FromDouble(1.0 / allSelections.Count);

        var scores = new Tensor<T>(new[] { p });
        foreach (var selection in allSelections)
            scores = Engine.TensorAdd(scores, Engine.TensorMultiplyScalar(SelectionScores(selection, p, null), weight));

        _aggregateScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));
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
            throw new InvalidOperationException("EnsembleFeatureSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("EnsembleFeatureSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EnsembleFeatureSelector has not been fitted.");

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
