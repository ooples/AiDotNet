using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiObjective;

/// <summary>
/// MOEA/D Multi-objective Feature Selection using decomposition.
/// </summary>
/// <remarks>
/// <para>
/// Uses the MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)
/// to optimize accuracy and feature count simultaneously by decomposing the
/// multi-objective problem into scalar subproblems.
/// </para>
/// <para><b>For Beginners:</b> MOEA/D breaks down the hard problem of optimizing
/// multiple goals at once into many simpler single-goal problems. Each simple
/// problem focuses on a different trade-off between accuracy and feature count.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MOEADSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly int _neighborhoodSize;

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

    public MOEADSelector(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int nGenerations = 100,
        int neighborhoodSize = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 2)
            throw new ArgumentException("Population size must be at least 2 to spread the weight vectors.", nameof(populationSize));
        if (neighborhoodSize < 1)
            throw new ArgumentException("Neighborhood size must be at least 1.", nameof(neighborhoodSize));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nGenerations = nGenerations;
        _neighborhoodSize = neighborhoodSize;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MOEADSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Evenly spread weight vectors (i / (N - 1), 1 - i / (N - 1))
        var weights = WeightVectors(_populationSize);

        // Compute neighborhoods: the T weight vectors closest to each one (itself included)
        var neighborhoods = ComputeNeighborhoods(weights, Math.Min(_neighborhoodSize, _populationSize));

        // Initialize population
        var population = ParetoOps.RandomSubsets<T>(rand, _populationSize, p, _nFeaturesToSelect);
        var (accuracy, featureCount) = EvaluateObjectives(population, scores);

        // Reference point
        var zAcc = ParetoOps.Max(accuracy);
        var zFeat = ParetoOps.Min(featureCount);

        // MOEA/D main loop (Zhang and Li 2007)
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                // Select parents from neighborhood
                var neighbors = neighborhoods[i];
                int p1 = neighbors[rand.Next(neighbors.Length)];
                int p2 = neighbors[rand.Next(neighbors.Length)];

                // Crossover and mutation
                var child = ParetoOps.EnsureNonEmpty(
                    ParetoOps.OnePointCrossover(
                        PopulationSearchOps.Row(population, p1), PopulationSearchOps.Row(population, p2), rand.Next(1, p)),
                    rand);
                child = ParetoOps.EnsureNonEmpty(ParetoOps.BitFlip(child, PopulationSearchOps.Uniform<T>(rand, p), 0.1), rand);
                var (childAcc, childFeat) = EvaluateObjectives(engine.Reshape(child, new[] { 1, p }), scores);

                // Update reference point
                zAcc = engine.TensorMax(zAcc, childAcc);
                zFeat = engine.TensorMin(zFeat, childFeat);

                // Update neighbors whose Tchebycheff value the child improves
                var index = new Tensor<int>((int[])neighbors.Clone(), new[] { neighbors.Length });
                var neighborWeights = engine.TensorIndexSelect(weights, index, 0);
                var childScalar = Tchebycheff(
                    engine.TensorBroadcastTo(childAcc, new[] { neighbors.Length }),
                    engine.TensorBroadcastTo(childFeat, new[] { neighbors.Length }),
                    neighborWeights, zAcc, zFeat);
                var currentScalar = Tchebycheff(
                    engine.TensorIndexSelect(accuracy, index, 0), engine.TensorIndexSelect(featureCount, index, 0),
                    neighborWeights, zAcc, zFeat);
                var improved = PreprocessingTensorOps.MaskPositions(engine.TensorLessThan(childScalar, currentScalar));
                if (improved.Length == 0)
                    continue;

                var replaced = new int[improved.Length];
                for (int r = 0; r < improved.Length; r++)
                    replaced[r] = neighbors[improved[r]];

                population = PopulationSearchOps.ScatterRows(
                    population, replaced,
                    engine.TensorBroadcastTo(engine.Reshape(child, new[] { 1, p }), new[] { replaced.Length, p }));
                accuracy = PopulationSearchOps.ScatterAt(
                    accuracy, replaced, engine.TensorBroadcastTo(childAcc, new[] { replaced.Length }));
                featureCount = PopulationSearchOps.ScatterAt(
                    featureCount, replaced, engine.TensorBroadcastTo(childFeat, new[] { replaced.Length }));
            }
        }

        // Track feature importance
        var importances = engine.Reshape(
            engine.TensorMatMul(engine.TensorTranspose(population), engine.Reshape(accuracy, new[] { _populationSize, 1 })),
            new[] { p });
        _featureImportances = importances.ToVector();

        // Select best solution: the smallest Tchebycheff value under the middle weight vector
        var middle = engine.TensorBroadcastTo(
            PopulationSearchOps.Rows(weights, new[] { _populationSize / 2 }), new[] { _populationSize, 2 });
        int bestIdx = PopulationSearchOps.ArgMin(Tchebycheff(accuracy, featureCount, middle, zAcc, zFeat));
        _selectedIndices = CompleteSelection(
            PopulationSearchOps.Row(population, bestIdx), scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <summary>
    /// The features of the 0/1 solution <paramref name="best"/> [p], cut to the first <paramref name="count"/> or,
    /// when it holds fewer, completed with the unselected features of highest <paramref name="scores"/> (ties to
    /// the lower index), in ascending index order. The selector keeps a fixed number of columns, and neither the
    /// initial subsets (1 to k features) nor crossover and bit-flip mutation keep a solution at k features.
    /// </summary>
    internal static int[] CompleteSelection(Tensor<T> best, Tensor<T> scores, int count)
    {
        var chosen = PreprocessingTensorOps.MaskPositions(best);
        if (chosen.Length >= count)
            return chosen.Take(count).ToArray();

        var taken = new HashSet<int>(chosen);
        var completed = chosen
            .Concat(PreprocessingTensorOps.StableOrder(scores, descending: true).Where(j => !taken.Contains(j)))
            .Take(count)
            .ToArray();
        Array.Sort(completed);
        return completed;
    }

    /// <summary>The weight vectors [N, 2] with rows (i / (N - 1), 1 - i / (N - 1)).</summary>
    internal static Tensor<T> WeightVectors(int count)
    {
        var engine = AiDotNetEngine.Current;
        // i / (N - 1) formed in double and rounded to T once: in Half the positions past 2048 are not exact (2999
        // rounds to 3000), which pushed the last weight past 1 and its complement below 0
        var first = engine.TensorDivideScalar(engine.TensorArange<double>(0, count, 1), count - 1.0).Cast<T>();
        return engine.TensorConcatenate(
            new[]
            {
                engine.Reshape(first, new[] { count, 1 }),
                engine.Reshape(PreprocessingTensorOps.OneMinus(first), new[] { count, 1 }),
            },
            1);
    }

    /// <summary>For each weight vector, the indices of the <paramref name="size"/> closest ones (stable order).</summary>
    private static int[][] ComputeNeighborhoods(Tensor<T> weights, int size)
    {
        int count = weights.Shape[0];
        var distances = FeatureSelectionTensorOps.PairwiseDistances(weights);
        var neighborhoods = new int[count][];
        for (int i = 0; i < count; i++)
        {
            neighborhoods[i] = PreprocessingTensorOps.StableOrder(PopulationSearchOps.Row(distances, i), descending: false)
                .Take(size)
                .ToArray();
        }

        return neighborhoods;
    }

    /// <summary>
    /// The two objectives of every row of a 0/1 mask [m, p]: the mean absolute target correlation of the selected
    /// features (to maximize) and the number of selected features (to minimize); an empty row scores (0, p).
    /// </summary>
    internal static (Tensor<T> Accuracy, Tensor<T> FeatureCount) EvaluateObjectives(Tensor<T> masks, Tensor<T> scores)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = masks.Shape[0];
        // counted in int: a 0/1 sum in T stops counting at 2048 for Half
        var count = engine.ReduceSum(masks.Cast<int>(), new[] { 1 }, false).Cast<T>();
        var emptyCount = PreprocessingTensorOps.Filled(new[] { m }, ops.FromDouble(masks.Shape[1]));
        return (PopulationSearchOps.SubsetFitness(masks, scores, 0, ops.Zero, ops.Zero),
                engine.TensorWhere(engine.TensorGreaterThan(count, ops.Zero), count, emptyCount));
    }

    /// <summary>
    /// The Tchebycheff scalarization max(w1 |z_acc - acc|, w2 |features - z_feat|) of each solution with the
    /// weight vector of the same row of <paramref name="weights"/> [m, 2].
    /// </summary>
    internal static Tensor<T> Tchebycheff(
        Tensor<T> accuracy, Tensor<T> featureCount, Tensor<T> weights, Tensor<T> zAcc, Tensor<T> zFeat)
    {
        var engine = AiDotNetEngine.Current;
        int m = accuracy.Length;
        var shape = new[] { m };
        var w1 = engine.Reshape(engine.TensorSlice(weights, new[] { 0, 0 }, new[] { m, 1 }), shape);
        var w2 = engine.Reshape(engine.TensorSlice(weights, new[] { 0, 1 }, new[] { m, 1 }), shape);
        var d1 = engine.TensorMultiply(w1, engine.TensorAbs(engine.TensorSubtract(engine.TensorBroadcastTo(zAcc, shape), accuracy)));
        var d2 = engine.TensorMultiply(w2, engine.TensorAbs(engine.TensorSubtract(featureCount, engine.TensorBroadcastTo(zFeat, shape))));
        return engine.TensorMax(d1, d2);
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
            throw new InvalidOperationException("MOEADSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MOEADSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MOEADSelector has not been fitted.");

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
