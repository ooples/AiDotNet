using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Genetic;

/// <summary>
/// Ant Colony Optimization based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses ant colony optimization where artificial ants construct solutions by
/// probabilistically selecting features based on pheromone trails.
/// </para>
/// <para><b>For Beginners:</b> Like ants finding food, virtual ants explore
/// feature combinations. Good combinations get "pheromone" deposited, making
/// those features more likely to be selected in future. Over time, pheromones
/// guide ants to good feature subsets.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AntColonySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nAnts;
    private readonly int _nIterations;
    private readonly double _evaporationRate;
    private readonly double _alpha;
    private readonly double _beta;
    private readonly int? _randomState;

    private Vector<T>? _pheromones;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NAnts => _nAnts;
    public int NIterations => _nIterations;
    public Vector<T>? Pheromones => _pheromones;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AntColonySelector(
        int nFeaturesToSelect = 10,
        int nAnts = 20,
        int nIterations = 100,
        double evaporationRate = 0.1,
        double alpha = 1.0,
        double beta = 2.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nAnts = nAnts;
        _nIterations = nIterations;
        _evaporationRate = evaporationRate;
        _alpha = alpha;
        _beta = beta;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AntColonySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // a subset's fitness is the mean absolute correlation of its features with the target
        var correlation = FeatureSelectionTensorOps.AbsoluteCorrelation(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.AsTensor(target));
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // heuristic information (feature quality) and its weight heuristic^beta
        T exponentAlpha = NumOps.FromDouble(_alpha);
        var heuristicWeight = Engine.TensorPow(Engine.TensorAddScalar(correlation, NumOps.FromDouble(0.01)), NumOps.FromDouble(_beta));

        // Initialize pheromones
        var pheromones = FilterStatisticTensorOps.FilledLike(correlation, NumOps.One);
        var bestSolution = new Tensor<T>(new[] { p });
        var bestFitness = new Tensor<T>(new[] { NumOps.MinValue }, new[] { 1 });
        T keep = NumOps.FromDouble(1 - _evaporationRate);

        // ACO iterations
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Each ant constructs a solution
            var weights = Engine.TensorMultiply(Engine.TensorPow(pheromones, exponentAlpha), heuristicWeight);
            var solutions = new Tensor<T>[_nAnts];
            for (int ant = 0; ant < _nAnts; ant++)
                solutions[ant] = ConstructSolution(rand, weights, numToSelect, p);
            var masks = Engine.TensorStack(solutions, 0);
            var fitness = FeatureSelectionTensorOps.MeanOverMask(masks, correlation);

            // the first ant with this iteration's best fitness replaces the best solution when it improves on it
            int leader = FilterStatisticTensorOps.FirstArgMax(fitness);
            var leaderFitness = FeatureSelectionTensorOps.Entries(fitness, new[] { leader });
            if (Engine.TensorGreaterThan(leaderFitness, bestFitness).Cast<int>().ToArray()[0] == 1)
            {
                bestFitness = leaderFitness;
                bestSolution = solutions[leader];
            }

            // Evaporate pheromones, deposit each ant's fitness on its features, then reinforce the best solution
            pheromones = Engine.TensorMultiplyScalar(pheromones, keep);
            pheromones = Engine.TensorAdd(pheromones, Engine.Reshape(
                Engine.TensorMatMul(Engine.TensorTranspose(masks), Engine.Reshape(fitness, new[] { _nAnts, 1 })), new[] { p }));
            pheromones = Engine.TensorAdd(pheromones, Engine.TensorMultiply(bestSolution, Engine.TensorBroadcastTo(bestFitness, new[] { p })));
        }

        _selectedIndices = FilterStatisticTensorOps.MaskPositions(bestSolution);

        // Normalize by the largest value when it is positive
        var largest = PreprocessingTensorOps.ColumnMax(Engine.Reshape(pheromones, new[] { p, 1 }));
        if (Engine.TensorGreaterThan(largest, NumOps.Zero).Cast<int>().ToArray()[0] == 1)
            pheromones = Engine.TensorDivide(pheromones, Engine.TensorBroadcastTo(largest, new[] { p }));

        _pheromones = pheromones.ToVector();
        IsFitted = true;
    }

    // One ant: numToSelect roulette-wheel picks without replacement, each with probability proportional to the
    // weight pheromone^alpha * heuristic^beta among the features still available (one uniform draw per pick).
    private Tensor<T> ConstructSolution(Random rand, Tensor<T> weights, int numToSelect, int p)
    {
        var draws = new double[numToSelect];
        for (int k = 0; k < numToSelect; k++)
            draws[k] = rand.NextDouble();
        var uniforms = new Tensor<double>(draws, new[] { numToSelect }).Cast<T>().ToArray();

        T one = NumOps.One;
        T zero = NumOps.Zero;
        var available = FilterStatisticTensorOps.FilledLike(weights, one);
        foreach (T u in uniforms)
        {
            var candidates = Engine.TensorMultiply(weights, available);
            var cumulative = Engine.TensorCumSum(candidates, 0);
            var threshold = Engine.TensorMultiplyScalar(Engine.ReduceSum(candidates, new[] { 0 }, true), u);
            var reached = Engine.TensorMultiply(
                PreprocessingTensorOps.OneMinus(Engine.TensorLessThan(cumulative, Engine.TensorBroadcastTo(threshold, new[] { p }))),
                available);

            // the first available feature whose cumulative weight reaches the draw, else the first available one
            var pick = FilterStatisticTensorOps.MaskPositions(reached);
            int selected = pick.Length > 0 ? pick[0] : FilterStatisticTensorOps.MaskPositions(available)[0];
            available = Engine.TensorSubtract(available, FeatureSelectionTensorOps.ScatterToFeatures(
                FilterStatisticTensorOps.Filled(new[] { 1 }, one), new[] { selected }, p, zero));
        }

        return PreprocessingTensorOps.OneMinus(available);
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
            throw new InvalidOperationException("AntColonySelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AntColonySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AntColonySelector has not been fitted.");

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
