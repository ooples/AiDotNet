using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Artificial Bee Colony for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Artificial Bee Colony is inspired by the foraging behavior of honey bees.
/// Employed bees exploit known food sources, onlookers choose sources based on
/// quality, and scouts search for new sources when old ones are exhausted.
/// </para>
/// <para><b>For Beginners:</b> Bees work together to find the best flowers.
/// Some bees (employed) keep working on known flowers, others (onlookers) watch
/// and join the bees with the best finds, and scout bees explore randomly for
/// new flowers. This balance of exploitation and exploration helps find optimal
/// feature subsets.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ArtificialBeeColony<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _colonySize;
    private readonly int _maxIterations;
    private readonly int _limit;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int ColonySize => _colonySize;
    public int MaxIterations => _maxIterations;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ArtificialBeeColony(
        int nFeaturesToSelect = 10,
        int colonySize = 30,
        int maxIterations = 100,
        int limit = 20,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (colonySize < 4)
            throw new ArgumentException(
                "Colony size must be at least 4 so there are at least two food sources to pair.", nameof(colonySize));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _colonySize = colonySize;
        _maxIterations = maxIterations;
        _limit = limit;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ArtificialBeeColony requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int nFood = _colonySize / 2;
        int targetCount = Math.Min(_nFeaturesToSelect, p);

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var correlations = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);

        // Initialize food sources
        var foodSources = PopulationSearchOps.Uniform<T>(random, nFood, p);
        var fitness = PopulationSearchOps.ThresholdFitness(foodSources, correlations, targetCount, 0.1);
        var trials = new int[nFood];

        // Memorize the best source
        int bestIdx = PopulationSearchOps.ArgMax(fitness);
        var bestFitness = PopulationSearchOps.At(fitness, bestIdx);
        var bestSolution = PopulationSearchOps.Row(foodSources, bestIdx);

        // Main loop (Karaboga 2005; one cycle = employed, onlooker and scout phases)
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // Employed bees phase: each bee tries a neighbour of its own source
            for (int i = 0; i < nFood; i++)
                ExploitSource(i, ref foodSources, ref fitness, trials, correlations, targetCount, random);

            // Onlooker bees phase: each onlooker picks a source by roulette on the ABC fitness and exploits it
            var probabilities = SelectionProbabilities(fitness);
            var cumulative = Engine.TensorCumSum(probabilities, 0);
            for (int t = 0; t < nFood; t++)
            {
                int selected = RouletteIndex(cumulative, PopulationSearchOps.Uniform<T>(random, 1));
                ExploitSource(selected, ref foodSources, ref fitness, trials, correlations, targetCount, random);
            }

            // Memorize the best source
            int cycleBest = PopulationSearchOps.ArgMax(fitness);
            var cycleBestFitness = PopulationSearchOps.At(fitness, cycleBest);
            if (PopulationSearchOps.Greater(cycleBestFitness, bestFitness))
            {
                bestFitness = cycleBestFitness;
                bestSolution = PopulationSearchOps.Row(foodSources, cycleBest);
            }

            // Scout bee phase: the most exhausted source is abandoned once its trial count reaches the limit
            int exhausted = 0;
            for (int i = 1; i < nFood; i++)
            {
                if (trials[i] > trials[exhausted])
                    exhausted = i;
            }

            if (trials[exhausted] >= _limit)
            {
                ScoutPhase(
                    ref foodSources, ref fitness, trials, exhausted, PopulationSearchOps.Uniform<T>(random, 1, p),
                    scout => PopulationSearchOps.ThresholdFitness(scout, correlations, targetCount, 0.1),
                    ref bestSolution, ref bestFitness);
            }
        }

        _featureScores = bestSolution.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(bestSolution, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// The scout phase: the exhausted source is replaced by the <paramref name="scout"/> [1, p], its trial count is
    /// reset, and the scout becomes the retained best when it beats <paramref name="bestFitness"/> (a scout found in
    /// the last cycle is otherwise never compared with the best).
    /// </summary>
    internal static void ScoutPhase(
        ref Tensor<T> foodSources, ref Tensor<T> fitness, int[] trials, int exhausted, Tensor<T> scout,
        Func<Tensor<T>, Tensor<T>> evaluate, ref Tensor<T> bestSolution, ref Tensor<T> bestFitness)
    {
        var scoutFitness = evaluate(scout);
        foodSources = PopulationSearchOps.ScatterRows(foodSources, new[] { exhausted }, scout);
        fitness = PopulationSearchOps.ReplaceAt(fitness, exhausted, scoutFitness);
        trials[exhausted] = 0;
        if (PopulationSearchOps.Greater(scoutFitness, bestFitness))
        {
            bestFitness = scoutFitness;
            bestSolution = PopulationSearchOps.Row(foodSources, exhausted);
        }
    }

    /// <summary>
    /// One neighbourhood search on source <paramref name="i"/>: a random dimension j and a random partner k != i,
    /// then v = x_i with v_j = x_ij + phi (x_ij - x_kj), phi ~ U[-1, 1], clamped to [0, 1]. Greedy selection keeps
    /// the better of v and x_i and counts a failed trial otherwise.
    /// </summary>
    private void ExploitSource(
        int i, ref Tensor<T> foodSources, ref Tensor<T> fitness, int[] trials, Tensor<T> correlations, int targetCount,
        Random random)
    {
        int nFood = foodSources.Shape[0];
        int p = foodSources.Shape[1];
        int dimension = random.Next(p);
        int partner = random.Next(nFood);
        while (partner == i)
            partner = random.Next(nFood);
        var phi = PopulationSearchOps.Affine(PopulationSearchOps.Uniform<T>(random, 1), 2, -1);

        var candidate = NeighbourCandidate(foodSources, i, partner, dimension, phi);
        var candidateFitness = PopulationSearchOps.ThresholdFitness(
            Engine.Reshape(candidate, new[] { 1, p }), correlations, targetCount, 0.1);
        if (PopulationSearchOps.Greater(candidateFitness, PopulationSearchOps.At(fitness, i)))
        {
            foodSources = PopulationSearchOps.ReplaceRow(foodSources, i, candidate);
            fitness = PopulationSearchOps.ReplaceAt(fitness, i, candidateFitness);
            trials[i] = 0;
        }
        else
        {
            trials[i]++;
        }
    }

    /// <summary>
    /// Karaboga's candidate v = x_i, except v_j = clamp(x_ij + phi (x_ij - x_kj), 0, 1); <paramref name="phi"/> is
    /// a [1] tensor.
    /// </summary>
    internal static Tensor<T> NeighbourCandidate(Tensor<T> foodSources, int i, int partner, int dimension, Tensor<T> phi)
    {
        var engine = AiDotNetEngine.Current;
        int p = foodSources.Shape[1];
        var source = PopulationSearchOps.Row(foodSources, i);
        var moved = PopulationSearchOps.Clamp01(engine.TensorAdd(
            source,
            PopulationSearchOps.Scale(engine.TensorSubtract(source, PopulationSearchOps.Row(foodSources, partner)), phi)));
        return engine.TensorWhere(FeatureSelectionTensorOps.OneHot<T>(dimension, p), moved, source);
    }

    /// <summary>
    /// Roulette probabilities p_i = fit_i / sum(fit). The selector maximizes its subset score F, so the cost Karaboga
    /// minimizes is f = -F, and fit = 1 / (1 + f) for f &gt;= 0 and 1 + |f| for f &lt; 0.
    /// </summary>
    internal static Tensor<T> SelectionProbabilities(Tensor<T> score)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = score.Shape[0];
        var cost = engine.TensorNegate(score);
        var negativeCost = engine.TensorLessThan(cost, ops.Zero);
        var ones = PreprocessingTensorOps.Filled(new[] { m }, ops.One);
        var fit = engine.TensorWhere(
            negativeCost,
            engine.TensorAdd(ones, engine.TensorAbs(cost)),
            engine.TensorDivide(ones, engine.TensorAddScalar(cost, ops.One)));
        return engine.TensorDivideScalar(fit, engine.TensorSum(fit));
    }

    /// <summary>
    /// The first index i whose cumulative probability reaches the uniform draw <paramref name="u"/> [1] (the last
    /// index when rounding leaves the total below the draw).
    /// </summary>
    internal static int RouletteIndex(Tensor<T> cumulative, Tensor<T> u)
    {
        var engine = AiDotNetEngine.Current;
        int m = cumulative.Shape[0];
        int below = engine.TensorCountNonzero(engine.TensorLessThan(cumulative, engine.TensorBroadcastTo(u, new[] { m })));
        return Math.Min(below, m - 1);
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
            throw new InvalidOperationException("ArtificialBeeColony has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ArtificialBeeColony does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ArtificialBeeColony has not been fitted.");

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
