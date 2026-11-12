using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiObjective;

/// <summary>
/// SPEA-II Multi-objective Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Strength Pareto Evolutionary Algorithm II (SPEA-II) which maintains
/// an archive of non-dominated solutions and uses fine-grained fitness assignment.
/// </para>
/// <para><b>For Beginners:</b> SPEA-II keeps track of the best solutions found
/// so far in an "archive." It measures how good a solution is by counting how
/// many other solutions it dominates (is better than in all objectives).
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SPEAIISelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _archiveSize;
    private readonly int _nGenerations;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SPEAIISelector(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int archiveSize = 25,
        int nGenerations = 100,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _archiveSize = archiveSize;
        _nGenerations = nGenerations;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SPEAIISelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var engine = Engine;

        var rand = RandomHelper.CreateSecureRandom();
        var scores = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);

        // Initialize population; the archive starts empty
        var population = ParetoOps.RandomSubsets<T>(rand, _populationSize, p, _nFeaturesToSelect);
        Tensor<T>? archive = null;

        // SPEA2 main loop (Zitzler, Laumanns and Thiele 2001)
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            var combined = archive is null ? population : engine.TensorConcatenate(new[] { population, archive }, 0);
            var (accuracy, featureCount) = EvaluateObjectives(combined, scores);

            // Fitness assignment and environmental selection into the next archive
            var fitness = ComputeFitness(accuracy, featureCount, out var distances);
            var members = EnvironmentalSelection(fitness, distances, _archiveSize);
            var memberIndex = new Tensor<int>(members, new[] { members.Length });
            archive = engine.TensorIndexSelect(combined, memberIndex, 0);
            var archiveFitness = engine.TensorIndexSelect(fitness, memberIndex, 0);

            // Mating selection on the new archive (binary tournament on its fitness) and variation
            var children = new Tensor<T>[_populationSize];
            for (int c = 0; c < _populationSize; c++)
            {
                int p1 = TournamentSelect(archiveFitness, rand);
                int p2 = TournamentSelect(archiveFitness, rand);
                var child = ParetoOps.EnsureNonEmpty(
                    ParetoOps.UniformCrossover(
                        PopulationSearchOps.Row(archive, p1), PopulationSearchOps.Row(archive, p2),
                        PopulationSearchOps.Uniform<T>(rand, p)),
                    rand);
                child = ParetoOps.EnsureNonEmpty(ParetoOps.BitFlip(child, PopulationSearchOps.Uniform<T>(rand, p), 0.05), rand);
                children[c] = engine.Reshape(child, new[] { 1, p });
            }

            population = engine.TensorConcatenate(children, 0);
        }

        if (archive is null)
            archive = population;

        // Track feature importance from archive
        var (archiveAccuracy, archiveCount) = EvaluateObjectives(archive, scores);
        var importances = engine.Reshape(
            engine.TensorMatMul(
                engine.TensorTranspose(archive), engine.Reshape(archiveAccuracy, new[] { archive.Shape[0], 1 })),
            new[] { p });
        _featureImportances = importances.ToVector();

        // Select best solution from archive: accuracy - 0.1 * features / p
        var tradeOff = engine.TensorSubtract(archiveAccuracy, engine.TensorMultiplyScalar(archiveCount, NumOps.FromDouble(0.1 / p)));
        var best = PopulationSearchOps.Row(archive, PopulationSearchOps.ArgMax(tradeOff));

        _selectedIndices = FeatureSelectionTensorOps.MaskPositions(best)
            .Take(_nFeaturesToSelect)
            .ToArray();

        IsFitted = true;
    }

    /// <summary>
    /// The two objectives of every row of a 0/1 mask [m, p]: the mean absolute target correlation of the selected
    /// features (to maximize) and the number of selected features (to minimize); an empty row scores (0, p).
    /// </summary>
    private (Tensor<T> Accuracy, Tensor<T> FeatureCount) EvaluateObjectives(Tensor<T> masks, Tensor<T> scores)
    {
        var engine = Engine;
        int m = masks.Shape[0];
        var count = engine.ReduceSum(masks, new[] { 1 }, false);
        var emptyCount = FeatureSelectionTensorOps.Filled(new[] { m }, NumOps.FromDouble(masks.Shape[1]));
        return (PopulationSearchOps.SubsetFitness(masks, scores, 0, NumOps.Zero, NumOps.Zero),
                engine.TensorWhere(engine.TensorGreaterThan(count, NumOps.Zero), count, emptyCount));
    }

    /// <summary>
    /// SPEA2 fitness F(i) = R(i) + D(i): the strength S(j) is the number of solutions j dominates, the raw
    /// fitness R(i) sums S(j) over the solutions j that dominate i, and the density D(i) = 1 / (sigma_i^k + 2)
    /// with sigma_i^k the objective-space distance to the k-th nearest neighbour, k = floor(sqrt(m)).
    /// <paramref name="distances"/> is the pairwise distance matrix [m, m] of the objective vectors.
    /// </summary>
    internal static Tensor<T> ComputeFitness(Tensor<T> accuracy, Tensor<T> featureCount, out Tensor<T> distances)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = accuracy.Length;

        var dominance = ParetoOps.Dominance(accuracy, featureCount);
        var strength = engine.ReduceSum(dominance, new[] { 1 }, true); // [m, 1]
        var raw = engine.Reshape(engine.TensorMatMul(engine.TensorTranspose(dominance), strength), new[] { m });

        var objectives = engine.TensorConcatenate(
            new[] { engine.Reshape(accuracy, new[] { m, 1 }), engine.Reshape(featureCount, new[] { m, 1 }) }, 1);
        distances = FeatureSelectionTensorOps.PairwiseDistances(objectives);

        // Row-sorted distances start with the zero distance to the solution itself, so column k is the k-th nearest
        int k = Math.Min((int)Math.Sqrt(m), m - 1);
        var sorted = engine.TensorSort(distances, 1, false).Item1;
        var kth = engine.Reshape(engine.TensorSlice(sorted, new[] { 0, k }, new[] { m, 1 }), new[] { m });
        var density = engine.TensorDivide(
            FeatureSelectionTensorOps.Filled(new[] { m }, ops.One), engine.TensorAddScalar(kth, ops.FromDouble(2)));
        return engine.TensorAdd(raw, density);
    }

    /// <summary>
    /// SPEA2 environmental selection: every nondominated solution (F &lt; 1) enters the archive. When they are
    /// fewer than <paramref name="archiveSize"/>, the best dominated ones by fitness fill it; when they are more,
    /// the archive truncation operator removes, one at a time, the solution whose sorted distances to the other
    /// remaining ones are lexicographically smallest.
    /// </summary>
    internal static int[] EnvironmentalSelection(Tensor<T> fitness, Tensor<T> distances, int archiveSize)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var nondominated = FeatureSelectionTensorOps.MaskPositions(engine.TensorLessThan(fitness, ops.One));
        if (nondominated.Length <= archiveSize)
        {
            // Nondominated solutions have F < 1 <= F of every dominated one, so they lead the fitness order
            return FeatureSelectionTensorOps.StableOrder(fitness, descending: false)
                .Take(Math.Min(archiveSize, fitness.Length))
                .ToArray();
        }

        var members = new List<int>(nondominated);
        while (members.Count > archiveSize)
        {
            var index = new Tensor<int>(members.ToArray(), new[] { members.Count });
            var among = engine.TensorIndexSelect(engine.TensorIndexSelect(distances, index, 0), index, 1);
            members.RemoveAt(LexicographicMinimumRow(engine.TensorSort(among, 1, false).Item1));
        }

        return members.ToArray();
    }

    /// <summary>The first row of a row-sorted distance matrix [L, L] that is lexicographically smallest.</summary>
    private static int LexicographicMinimumRow(Tensor<T> sortedRows)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int count = sortedRows.Shape[0];
        var candidates = FeatureSelectionTensorOps.Filled(new[] { count }, ops.One);
        var excluded = FeatureSelectionTensorOps.Filled(new[] { count }, ops.MaxValue);
        for (int c = 1; c < count && FeatureSelectionTensorOps.CountNonZero(candidates) > 1; c++)
        {
            var column = engine.Reshape(engine.TensorSlice(sortedRows, new[] { 0, c }, new[] { count, 1 }), new[] { count });
            var masked = engine.TensorWhere(engine.TensorGreaterThan(candidates, ops.Zero), column, excluded);
            var smallest = engine.TensorBroadcastTo(ParetoOps.Min(masked), new[] { count });
            candidates = engine.TensorMultiply(candidates, engine.TensorEquals(masked, smallest));
        }

        return FeatureSelectionTensorOps.MaskPositions(candidates)[0];
    }

    /// <summary>Binary tournament with replacement: the solution with the smaller fitness wins (the second on ties).</summary>
    private static int TournamentSelect(Tensor<T> fitness, Random rand)
    {
        int a = rand.Next(fitness.Length);
        int b = rand.Next(fitness.Length);
        return PopulationSearchOps.Greater(PopulationSearchOps.At(fitness, b), PopulationSearchOps.At(fitness, a)) ? a : b;
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
            throw new InvalidOperationException("SPEAIISelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SPEAIISelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SPEAIISelector has not been fitted.");

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
