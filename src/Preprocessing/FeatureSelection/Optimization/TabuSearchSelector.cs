using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Optimization;

/// <summary>
/// Tabu Search based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses tabu search optimization with a memory structure that prevents revisiting
/// recently explored solutions, helping escape local optima.
/// </para>
/// <para><b>For Beginners:</b> Tabu search keeps a "forbidden list" of recent moves.
/// This forces the algorithm to explore new areas instead of cycling back to old
/// solutions, helping find better feature combinations.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TabuSearchSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _tabuTenure;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int TabuTenure => _tabuTenure;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TabuSearchSelector(
        int nFeaturesToSelect = 10,
        int tabuTenure = 10,
        int nIterations = 500,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _tabuTenure = tabuTenure;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TabuSearchSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var engine = Engine;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var scores = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Initialize current solution
        var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(numToSelect).ToArray();
        var current = PopulationSearchOps.IndexMask<T>(indices, p);
        var currentScore = Score(current, scores);
        var best = current;
        var bestScore = currentScore;

        // Tabu list: the iteration until which each feature may not be swapped in or out
        // (kept in int: iteration numbers in T stop being exact past 2048 for Half)
        var tabuUntil = new int[p];
        // Importance totals are accumulated in double: a running total in a narrow T such as Half stops growing at
        // 2048 (2048 + 1 rounds back to 2048), which flattened the normalized importances over long runs.
        var importances = new Tensor<double>(new[] { p });

        for (int iter = 0; iter < _nIterations; iter++)
        {
            var tabu = TabuMask(tabuUntil, iter);
            var move = BestSwap(current, scores, tabu, bestScore, out var moveScore);
            if (move is not null)
            {
                var (toRemove, toAdd) = move.Value;
                current = engine.TensorAdd(
                    engine.TensorSubtract(current, FeatureSelectionTensorOps.OneHot<T>(toRemove, p)),
                    FeatureSelectionTensorOps.OneHot<T>(toAdd, p));
                tabuUntil[toRemove] = iter + _tabuTenure;
                tabuUntil[toAdd] = iter + _tabuTenure;
                currentScore = moveScore;

                if (PopulationSearchOps.Greater(currentScore, bestScore))
                {
                    best = current;
                    bestScore = currentScore;
                }
            }

            // Track feature importance
            importances = engine.TensorAdd(
                importances, PopulationSearchOps.Scale(current.Cast<double>(), currentScore.Cast<double>()));
        }

        _selectedIndices = PreprocessingTensorOps.MaskPositions(best);

        // Normalize feature importances
        double maxImportance = engine.TensorMaxValue(importances);
        if (maxImportance > 0)
            importances = engine.TensorDivideScalar(importances, maxImportance);
        _featureImportances = importances.Cast<T>().ToVector();

        IsFitted = true;
    }

    /// <summary>The mean absolute target correlation of the features of a 0/1 mask [p] (0 when empty), as [1].</summary>
    private Tensor<T> Score(Tensor<T> mask, Tensor<T> scores) =>
        PopulationSearchOps.SubsetFitness(Engine.Reshape(mask, new[] { 1, mask.Length }), scores, 0, NumOps.Zero, NumOps.Zero);

    /// <summary>The 0/1 [p] mask of the features still tabu at <paramref name="iter"/> (tabuUntil[j] &gt; iter).</summary>
    internal static Tensor<T> TabuMask(int[] tabuUntil, int iter) =>
        AiDotNetEngine.Current.TensorGreaterThan(
            new Tensor<int>((int[])tabuUntil.Clone(), new[] { tabuUntil.Length }), iter).Cast<T>();

    /// <summary>
    /// The best admissible swap of tabu search (Glover 1989): every pair of a selected feature r and an unselected
    /// feature a is scored by the mean score of the swapped subset, (S - s_r + s_a) / k. A swap is admissible when
    /// neither feature is tabu, or by aspiration when its score beats <paramref name="bestScore"/>. The first
    /// highest-scoring admissible swap (by r, then a) is returned with its score, or null when none is admissible.
    /// </summary>
    internal static (int Remove, int Add)? BestSwap(
        Tensor<T> current, Tensor<T> scores, Tensor<T> tabu, Tensor<T> bestScore, out Tensor<T> moveScore)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        moveScore = new Tensor<T>(new[] { 1 });

        // Only selected-to-unselected pairs are swaps: they are scored in blocks of selected rows against every
        // unselected feature ([rows, u], not a dense [p, p]: 800 MB of doubles for 10,000 features). Both index
        // lists are ascending and a later block only wins when strictly better, so ties still go to the lowest r,
        // then the lowest a.
        var selected = PreprocessingTensorOps.MaskPositions(current);
        var unselected = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.OneMinus(current));
        int k = selected.Length;
        int u = unselected.Length;
        if (k == 0 || u == 0)
            return null;

        // counted in int: a 0/1 sum in T stops counting at 2048 for Half
        var count = engine.ReduceSum(current.Cast<int>(), new[] { 0 }, true).Cast<T>();
        var total = engine.ReduceSum(engine.TensorMultiply(current, scores), new[] { 0 }, true);
        var unselectedIndex = new Tensor<int>(unselected, new[] { u });
        var addedScores = engine.Reshape(engine.TensorIndexSelect(scores, unselectedIndex, 0), new[] { 1, u });
        var addedTabu = engine.Reshape(engine.TensorIndexSelect(tabu, unselectedIndex, 0), new[] { 1, u });

        int blockRows = Math.Max(1, SwapBlockElements / u);
        (int Remove, int Add)? bestMove = null;
        Tensor<T>? bestValue = null;
        for (int start = 0; start < k; start += blockRows)
        {
            int rows = Math.Min(blockRows, k - start);
            var shape = new[] { rows, u };
            var removedIndex = new Tensor<int>(selected.Skip(start).Take(rows).ToArray(), new[] { rows });
            var removed = engine.TensorBroadcastTo(
                engine.Reshape(engine.TensorIndexSelect(scores, removedIndex, 0), new[] { rows, 1 }), shape);
            var swapScore = engine.TensorDivide(
                engine.TensorAdd(engine.TensorSubtract(
                    engine.TensorBroadcastTo(engine.Reshape(total, new[] { 1, 1 }), shape), removed),
                    engine.TensorBroadcastTo(addedScores, shape)),
                engine.TensorBroadcastTo(engine.Reshape(count, new[] { 1, 1 }), shape));

            var free = PreprocessingTensorOps.OneMinus(engine.TensorClamp(
                engine.TensorAdd(
                    engine.TensorBroadcastTo(
                        engine.Reshape(engine.TensorIndexSelect(tabu, removedIndex, 0), new[] { rows, 1 }), shape),
                    engine.TensorBroadcastTo(addedTabu, shape)),
                ops.Zero, ops.One));
            var aspiration = engine.TensorGreaterThan(
                swapScore, engine.TensorBroadcastTo(engine.Reshape(bestScore, new[] { 1, 1 }), shape));
            var admissible = engine.TensorClamp(engine.TensorAdd(free, aspiration), ops.Zero, ops.One);
            if (!PopulationSearchOps.Any(admissible))
                continue;

            var candidates = engine.Reshape(
                engine.TensorWhere(admissible, swapScore, PreprocessingTensorOps.Filled(shape, ops.MinValue)), new[] { rows * u });
            int flat = PopulationSearchOps.ArgMax(candidates);
            var value = PopulationSearchOps.At(candidates, flat);
            if (bestValue is null || PopulationSearchOps.Greater(value, bestValue))
            {
                bestValue = value;
                bestMove = (selected[start + flat / u], unselected[flat % u]);
            }
        }

        if (bestValue is not null)
            moveScore = bestValue;
        return bestMove;
    }

    // Upper bound on the elements of one block of selected-by-unselected swap scores ([rows, u]).
    private const int SwapBlockElements = 1 << 22;

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
            throw new InvalidOperationException("TabuSearchSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TabuSearchSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TabuSearchSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
