using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Grey Wolf Optimizer (GWO) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// GWO mimics the leadership hierarchy and hunting mechanism of grey wolves.
/// Alpha, beta, and delta wolves guide the pack toward optimal solutions.
/// </para>
/// <para><b>For Beginners:</b> Imagine a wolf pack hunting. The leaders (alpha,
/// beta, delta) guide the others toward prey. In optimization, this translates
/// to the best solutions guiding the search. Wolves encircle prey and then
/// attack - similarly, we narrow in on good feature subsets.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GreyWolfOptimizer<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int MaxIterations => _maxIterations;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GreyWolfOptimizer(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int maxIterations = 50,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 4)
            throw new ArgumentException("Population size must be at least 4.", nameof(populationSize));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _maxIterations = maxIterations;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GreyWolfOptimizer requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int targetCount = Math.Min(_nFeaturesToSelect, p);

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var correlations = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);

        // Initialize population (continuous positions)
        var positions = PopulationSearchOps.Uniform<T>(random, _populationSize, p);
        var fitness = PopulationSearchOps.ThresholdFitness(positions, correlations, targetCount, 0.1);

        // Alpha, beta and delta: the three best solutions found so far
        var (leaders, leaderFitness) = UpdateLeaders(null, null, positions, fitness);

        // Main loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double a = 2 - iter * 2.0 / _maxIterations;

            // Per wolf and dimension: r1, r2 for alpha, then beta, then delta, in the order of the scalar draws
            var draws = PopulationSearchOps.Uniform<T>(random, _populationSize, p, 6);
            positions = Hunt(positions, leaders, draws, a);
            fitness = PopulationSearchOps.ThresholdFitness(positions, correlations, targetCount, 0.1);
            (leaders, leaderFitness) = UpdateLeaders(leaders, leaderFitness, positions, fitness);
        }

        // Extract best solution
        var alpha = PopulationSearchOps.Row(leaders, 0);
        _featureScores = alpha.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(alpha, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// Mirjalili's position update: for each leader L in (alpha, beta, delta), A = 2 a r1 - a, C = 2 r2,
    /// D = |C L - X| and X_L = L - A D; the new position is (X_alpha + X_beta + X_delta) / 3, clamped to [0, 1].
    /// <paramref name="draws"/> is [m, p, 6] with (r1, r2) for alpha, beta and delta in that order.
    /// </summary>
    internal static Tensor<T> Hunt(Tensor<T> positions, Tensor<T> leaders, Tensor<T> draws, double a)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = positions.Shape[0];
        int p = positions.Shape[1];
        var shape = new[] { m, p };
        Tensor<T>? sum = null;
        for (int leader = 0; leader < 3; leader++)
        {
            var r1 = engine.Reshape(engine.TensorSlice(draws, new[] { 0, 0, 2 * leader }, new[] { m, p, 1 }), shape);
            var r2 = engine.Reshape(engine.TensorSlice(draws, new[] { 0, 0, 2 * leader + 1 }, new[] { m, p, 1 }), shape);
            var coefficientA = PopulationSearchOps.Affine(r1, 2 * a, -a);
            var coefficientC = engine.TensorMultiplyScalar(r2, ops.FromDouble(2));
            var position = engine.TensorBroadcastTo(
                engine.Reshape(PopulationSearchOps.Row(leaders, leader), new[] { 1, p }), shape);
            var distance = engine.TensorAbs(engine.TensorSubtract(engine.TensorMultiply(coefficientC, position), positions));
            var estimate = engine.TensorSubtract(position, engine.TensorMultiply(coefficientA, distance));
            sum = sum is null ? estimate : engine.TensorAdd(sum, estimate);
        }

        return PopulationSearchOps.Clamp01(engine.TensorDivideScalar(sum ?? positions, ops.FromDouble(3)));
    }

    /// <summary>
    /// The three best solutions found so far: the previous leaders (kept first on ties) together with the current
    /// population, ranked by fitness. Returns the leaders [3, p] and their fitness [3].
    /// </summary>
    internal static (Tensor<T> Leaders, Tensor<T> Fitness) UpdateLeaders(
        Tensor<T>? leaders, Tensor<T>? leaderFitness, Tensor<T> positions, Tensor<T> fitness)
    {
        var engine = AiDotNetEngine.Current;
        var pool = leaders is null ? positions : engine.TensorConcatenate(new[] { leaders, positions }, 0);
        var poolFitness = leaderFitness is null ? fitness : engine.TensorConcatenate(new[] { leaderFitness, fitness }, 0);
        var top = FeatureSelectionTensorOps.StableOrder(poolFitness, descending: true).Take(3).ToArray();
        var topFitness = PopulationSearchOps.Rows(engine.Reshape(poolFitness, new[] { poolFitness.Length, 1 }), top);
        return (PopulationSearchOps.Rows(pool, top), engine.Reshape(topFitness, new[] { top.Length }));
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
            throw new InvalidOperationException("GreyWolfOptimizer has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GreyWolfOptimizer does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GreyWolfOptimizer has not been fitted.");

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
