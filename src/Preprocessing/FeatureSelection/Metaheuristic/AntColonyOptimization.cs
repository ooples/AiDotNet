using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Ant Colony Optimization (ACO) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// ACO is inspired by how ants find shortest paths using pheromone trails.
/// Each ant constructs a feature subset, and pheromone levels are updated
/// based on the quality of the solutions found.
/// </para>
/// <para><b>For Beginners:</b> Imagine ants leaving scent trails as they explore.
/// Better paths get more scent, attracting more ants. Similarly, good feature
/// combinations get stronger "pheromone" signals, guiding the search toward
/// optimal feature subsets.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AntColonyOptimization<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nAnts;
    private readonly int _nIterations;
    private readonly double _alpha; // Pheromone importance
    private readonly double _beta;  // Heuristic importance
    private readonly double _rho;   // Evaporation rate
    private readonly int? _randomState;

    private Vector<T>? _pheromones;
    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NAnts => _nAnts;
    public int NIterations => _nIterations;
    public Vector<T>? Pheromones => _pheromones;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AntColonyOptimization(
        int nFeaturesToSelect = 10,
        int nAnts = 20,
        int nIterations = 50,
        double alpha = 1.0,
        double beta = 2.0,
        double rho = 0.1,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nAnts < 1)
            throw new ArgumentException("Number of ants must be at least 1.", nameof(nAnts));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nAnts = nAnts;
        _nIterations = nIterations;
        _alpha = alpha;
        _beta = beta;
        _rho = rho;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AntColonyOptimization requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var engine = Engine;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Initialize pheromone trails
        var pheromones = FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One);

        // Compute heuristic information (correlation-based)
        var heuristic = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.FromDouble(0.01));
        _featureScores = heuristic.ToVector();

        int[]? bestSubset = null;
        Tensor<T>? bestFitness = null;
        T alpha = NumOps.FromDouble(_alpha);
        T beta = NumOps.FromDouble(_beta);
        T retention = NumOps.FromDouble(1 - _rho);
        T minimumPheromone = NumOps.FromDouble(0.01);
        var desirability = engine.TensorPower(heuristic, beta);

        for (int iter = 0; iter < _nIterations; iter++)
        {
            // Each ant constructs a solution with the transition weights tau^alpha * eta^beta of this iteration
            var weights = engine.TensorMultiply(engine.TensorPower(pheromones, alpha), desirability);
            var antSubsets = new int[_nAnts][];
            var masks = new Tensor<T>[_nAnts];
            for (int ant = 0; ant < _nAnts; ant++)
            {
                antSubsets[ant] = ConstructSolution(weights, random);
                masks[ant] = engine.Reshape(PopulationSearchOps.IndexMask<T>(antSubsets[ant], p), new[] { 1, p });
            }

            // Fitness: mean heuristic value of the chosen features
            var antMasks = engine.TensorConcatenate(masks, 0);
            var antFitness = PopulationSearchOps.SubsetFitness(antMasks, heuristic, 0, NumOps.Zero, NumOps.Zero);

            int iterBest = PopulationSearchOps.ArgMax(antFitness);
            var iterBestFitness = PopulationSearchOps.At(antFitness, iterBest);
            if (bestFitness is null || PopulationSearchOps.Greater(iterBestFitness, bestFitness))
            {
                bestFitness = iterBestFitness;
                bestSubset = (int[])antSubsets[iterBest].Clone();
            }

            // Evaporation, deposit of each ant's fitness on its features, and the minimum pheromone level
            var deposit = engine.Reshape(
                engine.TensorMatMul(engine.TensorTranspose(antMasks), engine.Reshape(antFitness, new[] { _nAnts, 1 })),
                new[] { p });
            pheromones = engine.TensorMax(
                engine.TensorAdd(engine.TensorMultiplyScalar(pheromones, retention), deposit), minimumPheromone);
        }

        _pheromones = pheromones.ToVector();

        // Use best found subset
        if (bestSubset is not null)
        {
            _selectedIndices = bestSubset.OrderBy(x => x).ToArray();
        }
        else
        {
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
        }

        IsFitted = true;
    }

    /// <summary>
    /// One ant's tour: features are added one at a time by roulette over the unselected features, with probability
    /// proportional to <paramref name="weights"/>; when every remaining weight is zero the feature is drawn
    /// uniformly.
    /// </summary>
    private int[] ConstructSolution(Tensor<T> weights, Random random)
    {
        var engine = Engine;
        int p = weights.Length;
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        var selected = new int[numToSelect];
        var available = FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One);
        var zero = new Tensor<T>(new[] { 1 });

        for (int step = 0; step < numToSelect; step++)
        {
            var masked = engine.TensorMultiply(weights, available);
            var total = engine.ReduceSum(masked, new[] { 0 }, true);
            var availablePositions = FeatureSelectionTensorOps.MaskPositions(available);

            int chosen;
            if (PopulationSearchOps.Greater(total, zero))
            {
                // r = u * total; the feature is the first available one whose cumulative weight reaches r
                var r = engine.TensorMultiply(PopulationSearchOps.Uniform<T>(random, 1), total);
                var cumulative = engine.TensorCumSum(masked, 0);
                var below = engine.TensorMultiply(
                    engine.TensorLessThan(cumulative, engine.TensorBroadcastTo(r, new[] { p })), available);
                int rank = Math.Min(engine.TensorCountNonzero(below), availablePositions.Length - 1);
                chosen = availablePositions[rank];
            }
            else
            {
                chosen = availablePositions[random.Next(availablePositions.Length)];
            }

            selected[step] = chosen;
            available = engine.TensorSubtract(available, FeatureSelectionTensorOps.OneHot<T>(chosen, p));
        }

        return selected;
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
            throw new InvalidOperationException("AntColonyOptimization has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AntColonyOptimization does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AntColonyOptimization has not been fitted.");

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
