using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiObjective;

/// <summary>
/// NSGA-II Multi-objective Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to optimize
/// multiple objectives simultaneously: maximizing prediction accuracy while
/// minimizing the number of selected features.
/// </para>
/// <para><b>For Beginners:</b> Sometimes we want both good accuracy AND few
/// features. NSGA-II finds multiple solutions that trade off between these goals.
/// Some solutions have high accuracy but many features; others have fewer
/// features but slightly lower accuracy. You can then choose the best trade-off.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class NSGAIISelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _maxFeatures;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _crossoverRate;
    private readonly double _mutationRate;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MaxFeatures => _maxFeatures;
    public int PopulationSize => _populationSize;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public NSGAIISelector(
        int maxFeatures = 20,
        int populationSize = 50,
        int nGenerations = 100,
        double crossoverRate = 0.9,
        double mutationRate = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxFeatures < 1)
            throw new ArgumentException("Max features must be at least 1.", nameof(maxFeatures));

        _maxFeatures = maxFeatures;
        _populationSize = populationSize;
        _nGenerations = nGenerations;
        _crossoverRate = crossoverRate;
        _mutationRate = mutationRate;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "NSGAIISelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize population
        var population = ParetoOps.RandomSubsets<T>(rand, _populationSize, p, _maxFeatures);

        // NSGA-II main loop (Deb et al. 2002)
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            // Evaluate objectives, non-dominated sorting and crowding distance of the parents
            var (accuracy, featureCount) = EvaluateObjectives(population, scores);
            var fronts = ParetoOps.Fronts(ParetoOps.Dominance(accuracy, featureCount), out var rank);
            var crowding = ParetoOps.CrowdingDistance(fronts, accuracy, featureCount);

            // Create offspring by crowded binary tournament, one-point crossover and bit-flip mutation
            var offspring = new Tensor<T>[_populationSize];
            for (int i = 0; i < _populationSize; i += 2)
            {
                int parent1 = TournamentSelect(rank, crowding, rand);
                int parent2 = TournamentSelect(rank, crowding, rand);
                var (child1, child2) = Crossover(
                    PopulationSearchOps.Row(population, parent1), PopulationSearchOps.Row(population, parent2), p, rand);

                offspring[i] = engine.Reshape(Mutate(child1, p, rand), new[] { 1, p });
                var mutated2 = Mutate(child2, p, rand);
                if (i + 1 < _populationSize)
                    offspring[i + 1] = engine.Reshape(mutated2, new[] { 1, p });
            }

            // Combine parent and offspring, then keep the best fronts and the least crowded of the last one
            var combined = engine.TensorConcatenate(new[] { population, engine.TensorConcatenate(offspring, 0) }, 0);
            var (combinedAccuracy, combinedCount) = EvaluateObjectives(combined, scores);
            var combinedFronts = ParetoOps.Fronts(ParetoOps.Dominance(combinedAccuracy, combinedCount), out _);
            var combinedCrowding = ParetoOps.CrowdingDistance(combinedFronts, combinedAccuracy, combinedCount);
            var survivors = SelectSurvivors(combinedFronts, combinedCrowding, _populationSize);
            population = engine.TensorIndexSelect(combined, new Tensor<int>(survivors, new[] { survivors.Length }), 0);
        }

        // Select best solution (knee point)
        var (finalAccuracy, finalCount) = EvaluateObjectives(population, scores);
        int bestIdx = FindKneePoint(finalAccuracy, finalCount);
        var bestSolution = PopulationSearchOps.Row(population, bestIdx);

        // Feature importance: the accuracy of every final solution that selects the feature
        var importances = engine.Reshape(
            engine.TensorMatMul(
                engine.TensorTranspose(population), engine.Reshape(finalAccuracy, new[] { population.Shape[0], 1 })),
            new[] { p });
        _featureImportances = importances.ToVector();

        _selectedIndices = FeatureSelectionTensorOps.MaskPositions(bestSolution);

        // Limit to max features
        if (_selectedIndices.Length > _maxFeatures)
            _selectedIndices = FeatureSelectionTensorOps.TopKWhere(importances, bestSolution, _maxFeatures);

        IsFitted = true;
    }

    /// <summary>
    /// The two objectives of every row of a 0/1 mask [m, p]: the mean absolute target correlation of the selected
    /// features (to maximize; 0 for an empty row) and the number of selected features (to minimize).
    /// </summary>
    private (Tensor<T> Accuracy, Tensor<T> FeatureCount) EvaluateObjectives(Tensor<T> masks, Tensor<T> scores) =>
        (PopulationSearchOps.SubsetFitness(masks, scores, 0, NumOps.Zero, NumOps.Zero),
         Engine.ReduceSum(masks, new[] { 1 }, false));

    /// <summary>
    /// Crowded binary tournament: of two uniformly drawn solutions the one in the better front wins, and within a
    /// front the one with the larger crowding distance.
    /// </summary>
    private static int TournamentSelect(Tensor<T> rank, Tensor<T> crowding, Random rand)
    {
        int a = rand.Next(rank.Length);
        int b = rand.Next(rank.Length);

        var rankA = PopulationSearchOps.At(rank, a);
        var rankB = PopulationSearchOps.At(rank, b);
        if (PopulationSearchOps.Greater(rankB, rankA)) return a;
        if (PopulationSearchOps.Greater(rankA, rankB)) return b;
        return PopulationSearchOps.Greater(PopulationSearchOps.At(crowding, a), PopulationSearchOps.At(crowding, b)) ? a : b;
    }

    private (Tensor<T>, Tensor<T>) Crossover(Tensor<T> parent1, Tensor<T> parent2, int p, Random rand)
    {
        if (rand.NextDouble() < _crossoverRate)
        {
            int point = rand.Next(1, p);
            return (ParetoOps.OnePointCrossover(parent1, parent2, point), ParetoOps.OnePointCrossover(parent2, parent1, point));
        }

        return (parent1, parent2);
    }

    private Tensor<T> Mutate(Tensor<T> individual, int p, Random rand) =>
        ParetoOps.EnsureNonEmpty(ParetoOps.BitFlip(individual, PopulationSearchOps.Uniform<T>(rand, p), _mutationRate), rand);

    /// <summary>
    /// NSGA-II environmental selection: whole fronts are taken in order while they fit in <paramref name="size"/>,
    /// and the remaining places go to the solutions of the next front with the largest crowding distance.
    /// </summary>
    internal static int[] SelectSurvivors(IReadOnlyList<int[]> fronts, Tensor<T> crowding, int size)
    {
        var selected = new List<int>(size);
        int frontIdx = 0;
        while (frontIdx < fronts.Count && selected.Count + fronts[frontIdx].Length <= size)
        {
            selected.AddRange(fronts[frontIdx]);
            frontIdx++;
        }

        if (selected.Count < size && frontIdx < fronts.Count)
        {
            var front = fronts[frontIdx];
            var frontCrowding = AiDotNetEngine.Current.TensorIndexSelect(
                crowding, new Tensor<int>((int[])front.Clone(), new[] { front.Length }), 0);
            var order = FeatureSelectionTensorOps.StableOrder(frontCrowding, descending: true);
            for (int r = 0; r < size - selected.Count && r < order.Length; r++)
                selected.Add(front[order[r]]);
        }

        return selected.ToArray();
    }

    /// <summary>
    /// The knee point: the solution closest to the ideal (1, 1) of (accuracy / best accuracy,
    /// fewest features / features), first on ties.
    /// </summary>
    private int FindKneePoint(Tensor<T> accuracy, Tensor<T> featureCount)
    {
        var engine = Engine;
        int m = accuracy.Length;
        var zeros = new Tensor<T>(new[] { m });
        var maxAcc = engine.TensorBroadcastTo(ParetoOps.Max(accuracy), new[] { m });
        var minFeat = engine.TensorBroadcastTo(ParetoOps.Min(featureCount), new[] { m });
        var ones = FeatureSelectionTensorOps.Filled(new[] { m }, NumOps.One);

        var accPositive = engine.TensorGreaterThan(maxAcc, NumOps.Zero);
        var normAcc = engine.TensorWhere(accPositive, engine.TensorDivide(accuracy, engine.TensorWhere(accPositive, maxAcc, ones)), zeros);
        var featPositive = engine.TensorGreaterThan(minFeat, NumOps.Zero);
        var safeCount = engine.TensorWhere(engine.TensorGreaterThan(featureCount, NumOps.Zero), featureCount, ones);
        var normFeat = engine.TensorWhere(featPositive, engine.TensorDivide(minFeat, safeCount), zeros);

        var dist = engine.TensorSqrt(engine.TensorAdd(
            engine.TensorSquare(FeatureSelectionTensorOps.OneMinus(normAcc)),
            engine.TensorSquare(FeatureSelectionTensorOps.OneMinus(normFeat))));
        return PopulationSearchOps.ArgMin(dist);
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
            throw new InvalidOperationException("NSGAIISelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("NSGAIISelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NSGAIISelector has not been fitted.");

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
