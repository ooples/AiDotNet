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
        var tabuUntil = new Tensor<T>(new[] { p });
        var importances = new Tensor<T>(new[] { p });

        for (int iter = 0; iter < _nIterations; iter++)
        {
            var tabu = engine.TensorGreaterThan(tabuUntil, FeatureSelectionTensorOps.Count<T>(iter));
            var move = BestSwap(current, scores, tabu, bestScore, out var moveScore);
            if (move is not null)
            {
                var (toRemove, toAdd) = move.Value;
                current = engine.TensorAdd(
                    engine.TensorSubtract(current, FeatureSelectionTensorOps.OneHot<T>(toRemove, p)),
                    FeatureSelectionTensorOps.OneHot<T>(toAdd, p));
                var until = PreprocessingTensorOps.Filled(new[] { 2 }, FeatureSelectionTensorOps.Count<T>(iter + _tabuTenure));
                tabuUntil = PopulationSearchOps.ScatterAt(tabuUntil, new[] { toRemove, toAdd }, until);
                currentScore = moveScore;

                if (PopulationSearchOps.Greater(currentScore, bestScore))
                {
                    best = current;
                    bestScore = currentScore;
                }
            }

            // Track feature importance
            importances = engine.TensorAdd(importances, PopulationSearchOps.Scale(current, currentScore));
        }

        _selectedIndices = PreprocessingTensorOps.MaskPositions(best);

        // Normalize feature importances
        var zero = new Tensor<T>(new[] { 1 });
        var maxImportance = ParetoOps.Max(importances);
        if (PopulationSearchOps.Greater(maxImportance, zero))
            importances = engine.TensorDivide(importances, engine.TensorBroadcastTo(maxImportance, new[] { p }));
        _featureImportances = importances.ToVector();

        IsFitted = true;
    }

    /// <summary>The mean absolute target correlation of the features of a 0/1 mask [p] (0 when empty), as [1].</summary>
    private Tensor<T> Score(Tensor<T> mask, Tensor<T> scores) =>
        PopulationSearchOps.SubsetFitness(Engine.Reshape(mask, new[] { 1, mask.Length }), scores, 0, NumOps.Zero, NumOps.Zero);

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
        int p = current.Length;
        var square = new[] { p, p };
        moveScore = new Tensor<T>(new[] { 1 });

        var count = engine.ReduceSum(current, new[] { 0 }, true);
        var total = engine.ReduceSum(engine.TensorMultiply(current, scores), new[] { 0 }, true);
        var removed = engine.TensorBroadcastTo(engine.Reshape(scores, new[] { p, 1 }), square);
        var added = engine.TensorBroadcastTo(engine.Reshape(scores, new[] { 1, p }), square);
        var swapScore = engine.TensorDivide(
            engine.TensorAdd(engine.TensorSubtract(engine.TensorBroadcastTo(engine.Reshape(total, new[] { 1, 1 }), square), removed), added),
            engine.TensorBroadcastTo(engine.Reshape(count, new[] { 1, 1 }), square));

        var outOf = engine.TensorBroadcastTo(engine.Reshape(current, new[] { p, 1 }), square);
        var into = engine.TensorBroadcastTo(engine.Reshape(PreprocessingTensorOps.OneMinus(current), new[] { 1, p }), square);
        var free = PreprocessingTensorOps.OneMinus(engine.TensorClamp(
            engine.TensorAdd(
                engine.TensorBroadcastTo(engine.Reshape(tabu, new[] { p, 1 }), square),
                engine.TensorBroadcastTo(engine.Reshape(tabu, new[] { 1, p }), square)),
            ops.Zero, ops.One));
        var aspiration = engine.TensorGreaterThan(swapScore, engine.TensorBroadcastTo(engine.Reshape(bestScore, new[] { 1, 1 }), square));
        var admissible = engine.TensorMultiply(
            engine.TensorMultiply(outOf, into), engine.TensorClamp(engine.TensorAdd(free, aspiration), ops.Zero, ops.One));
        if (!PopulationSearchOps.Any(admissible))
            return null;

        var candidates = engine.Reshape(
            engine.TensorWhere(admissible, swapScore, PreprocessingTensorOps.Filled(square, ops.MinValue)), new[] { p * p });
        int flat = PopulationSearchOps.ArgMax(candidates);
        moveScore = PopulationSearchOps.At(candidates, flat);
        return (flat / p, flat % p);
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
