using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Salp Swarm Algorithm (SSA) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Salp Swarm Algorithm is inspired by the swarming behavior of salps in oceans.
/// Salps form a chain where the leader follows the food source and followers follow
/// the salp ahead of them.
/// </para>
/// <para><b>For Beginners:</b> Salps are sea creatures that form chains. The first
/// salp (leader) moves toward the best food, while others follow in line. This
/// creates a balance between exploring new areas and exploiting known good spots,
/// which helps find the best feature combinations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SalpSwarm<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int NIterations => _nIterations;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SalpSwarm(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int nIterations = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 2)
            throw new ArgumentException("Population size must be at least 2.", nameof(populationSize));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SalpSwarm requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var scores = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);
        _featureScores = scores.ToVector();

        // Initialize salp positions
        var positions = PopulationSearchOps.Uniform<T>(random, _populationSize, p);
        var fitness = PopulationSearchOps.ThresholdFitness(positions, scores, _nFeaturesToSelect, 0.1);

        // The food source is the best solution found so far
        int bestIdx = PopulationSearchOps.ArgMax(fitness);
        var foodFitness = PopulationSearchOps.At(fitness, bestIdx);
        var foodPosition = PopulationSearchOps.Row(positions, bestIdx);
        int nLeaders = _populationSize / 2;

        // Main loop (Mirjalili et al. 2017): the chain moves against the food found up to the previous iteration
        for (int iter = 0; iter < _nIterations; iter++)
        {
            double c1 = 2 * Math.Exp(-Math.Pow(4.0 * iter / _nIterations, 2));

            // Per leader and dimension: c2 then c3, in the order of the scalar draws
            var draws = nLeaders > 0 ? PopulationSearchOps.Uniform<T>(random, nLeaders, p, 2) : null;
            positions = MoveChain(positions, foodPosition, draws, c1);
            fitness = PopulationSearchOps.ThresholdFitness(positions, scores, _nFeaturesToSelect, 0.1);

            int iterBest = PopulationSearchOps.ArgMax(fitness);
            var iterBestFitness = PopulationSearchOps.At(fitness, iterBest);
            if (PopulationSearchOps.Greater(iterBestFitness, foodFitness))
            {
                foodFitness = iterBestFitness;
                foodPosition = PopulationSearchOps.Row(positions, iterBest);
            }
        }

        // Extract selected features from food position
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(
            Engine.TensorMultiply(foodPosition, scores), _nFeaturesToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// One move of the salp chain. The first <c>draws.Shape[0]</c> salps lead: x_j = F_j + c1 c2 when c3 &lt; 0.5
    /// and F_j - c1 c2 otherwise (bounds [0, 1]); every follower i moves to (x_i + x_{i-1}) / 2 using the salp ahead
    /// as already moved. Positions are clamped to [0, 1]. <paramref name="draws"/> is [leaders, p, 2] (c2, c3).
    /// </summary>
    internal static Tensor<T> MoveChain(Tensor<T> positions, Tensor<T> food, Tensor<T>? draws, double c1)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = positions.Shape[0];
        int p = positions.Shape[1];
        int nLeaders = draws?.Shape[0] ?? 0;

        if (draws is not null && nLeaders > 0)
        {
            var leaderShape = new[] { nLeaders, p };
            var c2 = engine.Reshape(engine.TensorSlice(draws, new[] { 0, 0, 0 }, new[] { nLeaders, p, 1 }), leaderShape);
            var c3 = engine.Reshape(engine.TensorSlice(draws, new[] { 0, 0, 1 }, new[] { nLeaders, p, 1 }), leaderShape);
            var step = engine.TensorMultiplyScalar(c2, ops.FromDouble(c1));
            var sign = engine.TensorWhere(
                engine.TensorLessThan(c3, ops.FromDouble(0.5)),
                FeatureSelectionTensorOps.Filled(leaderShape, ops.One),
                FeatureSelectionTensorOps.Filled(leaderShape, ops.FromDouble(-1)));
            var foodRows = engine.TensorBroadcastTo(engine.Reshape(food, new[] { 1, p }), leaderShape);
            var leaders = PopulationSearchOps.Clamp01(engine.TensorAdd(foodRows, engine.TensorMultiply(sign, step)));
            positions = nLeaders == m
                ? leaders
                : engine.TensorConcatenate(
                    new[] { leaders, engine.TensorSlice(positions, new[] { nLeaders, 0 }, new[] { m - nLeaders, p }) }, 0);
        }

        // Followers: Newton's law of motion with v0 = 0 reduces to the midpoint of each salp and the one ahead
        T half = ops.FromDouble(0.5);
        var ahead = nLeaders > 0 ? PopulationSearchOps.Row(positions, nLeaders - 1) : null;
        var rows = new List<Tensor<T>>(m);
        for (int i = 0; i < m; i++)
        {
            var row = PopulationSearchOps.Row(positions, i);
            if (i >= nLeaders && ahead is not null)
                row = PopulationSearchOps.Clamp01(engine.TensorMultiplyScalar(engine.TensorAdd(row, ahead), half));
            rows.Add(engine.Reshape(row, new[] { 1, p }));
            ahead = row;
        }

        return engine.TensorConcatenate(rows.ToArray(), 0);
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
            throw new InvalidOperationException("SalpSwarm has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SalpSwarm does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SalpSwarm has not been fitted.");

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
