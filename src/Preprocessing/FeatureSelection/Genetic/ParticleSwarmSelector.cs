using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Genetic;

/// <summary>
/// Particle Swarm Optimization based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses particle swarm optimization to find optimal feature subsets where particles
/// represent feature combinations moving through the solution space.
/// </para>
/// <para><b>For Beginners:</b> Imagine a swarm of birds searching for food. Each bird
/// (particle) explores feature combinations, remembers its best finds, and learns from
/// the swarm's best discoveries. Over time, they converge on good feature subsets.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ParticleSwarmSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _swarmSize;
    private readonly int _nIterations;
    private readonly double _inertia;
    private readonly double _cognitive;
    private readonly double _social;
    private readonly int? _randomState;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int SwarmSize => _swarmSize;
    public int NIterations => _nIterations;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ParticleSwarmSelector(
        int nFeaturesToSelect = 10,
        int swarmSize = 30,
        int nIterations = 100,
        double inertia = 0.7,
        double cognitive = 1.5,
        double social = 1.5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _swarmSize = swarmSize;
        _nIterations = nIterations;
        _inertia = inertia;
        _cognitive = cognitive;
        _social = social;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ParticleSwarmSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var correlation = FeatureScoringTensorOps.AbsoluteCorrelation(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.AsTensor(target));
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Initialize particles (positions as probabilities): per particle and feature, a position then a velocity draw
        var positionDraws = new double[_swarmSize * p];
        var velocityDraws = new double[_swarmSize * p];
        for (int k = 0; k < positionDraws.Length; k++)
        {
            positionDraws[k] = rand.NextDouble();
            velocityDraws[k] = rand.NextDouble();
        }

        var shape = new[] { _swarmSize, p };
        var positions = new Tensor<double>(positionDraws, shape).Cast<T>();
        var velocities = Engine.TensorMultiplyScalar(
            Engine.TensorSubtractScalar(new Tensor<double>(velocityDraws, shape).Cast<T>(), NumOps.FromDouble(0.5)), NumOps.FromDouble(0.1));
        var personalBest = positions;
        var personalBestFitness = FilterStatisticTensorOps.Filled(new[] { _swarmSize }, NumOps.MinValue);
        var globalBest = new Tensor<T>(new[] { p });
        var globalBestFitness = new Tensor<T>(new[] { NumOps.MinValue }, new[] { 1 });
        // the importances (each iteration's fitness-weighted feature counts, summed over the iterations) are summed
        // in double whatever T is, and normalised there: a running sum in Half stops taking a contribution c once it
        // passes 2048 c, which capped the frequently kept features and skewed their normalised ratios
        var importances = new Tensor<double>(new[] { p });
        T inertia = NumOps.FromDouble(_inertia);
        T cognitive = NumOps.FromDouble(_cognitive);
        T social = NumOps.FromDouble(_social);
        T minusOne = NumOps.FromDouble(-1);
        T zero = NumOps.Zero;
        T one = NumOps.One;
        var r1Draws = new double[_swarmSize * p];
        var r2Draws = new double[_swarmSize * p];

        // PSO iterations
        for (int iter = 0; iter < _nIterations; iter++)
        {
            // each particle's subset is its numToSelect largest positions
            var particleMasks = new Tensor<T>[_swarmSize];
            for (int i = 0; i < _swarmSize; i++)
                particleMasks[i] = FeatureScoringTensorOps.TopKMask(
                    Engine.Reshape(FeatureScoringTensorOps.Rows(positions, new[] { i }), new[] { p }), numToSelect);
            var masks = Engine.TensorStack(particleMasks, 0);
            var fitness = FeatureScoringTensorOps.MeanOverMask(masks, correlation);

            // Update personal bests
            var improved = Engine.TensorGreaterThan(fitness, personalBestFitness);
            personalBestFitness = Engine.TensorWhere(improved, fitness, personalBestFitness);
            personalBest = Engine.TensorWhere(FilterStatisticTensorOps.ColumnBroadcast(improved, p), positions, personalBest);

            // Update global best: the first particle with the best fitness, when it improves on the global best
            int leader = FilterStatisticTensorOps.FirstArgMax(fitness);
            var leaderFitness = FeatureScoringTensorOps.Entries(fitness, new[] { leader });
            if (Engine.TensorGreaterThan(leaderFitness, globalBestFitness).Cast<int>().ToArray()[0] == 1)
            {
                globalBestFitness = leaderFitness;
                globalBest = Engine.Reshape(FeatureScoringTensorOps.Rows(positions, new[] { leader }), new[] { p });
            }

            // Track feature importance
            importances = Engine.TensorAdd(importances, Engine.Reshape(
                Engine.TensorMatMul(Engine.TensorTranspose(masks), Engine.Reshape(fitness, new[] { _swarmSize, 1 })), new[] { p })
                .Cast<double>());

            // Update velocities and positions (per particle and feature, r1 is drawn before r2)
            for (int k = 0; k < r1Draws.Length; k++)
            {
                r1Draws[k] = rand.NextDouble();
                r2Draws[k] = rand.NextDouble();
            }

            var r1 = new Tensor<double>((double[])r1Draws.Clone(), shape).Cast<T>();
            var r2 = new Tensor<double>((double[])r2Draws.Clone(), shape).Cast<T>();
            var toPersonal = Engine.TensorMultiply(Engine.TensorMultiplyScalar(r1, cognitive), Engine.TensorSubtract(personalBest, positions));
            var toGlobal = Engine.TensorMultiply(Engine.TensorMultiplyScalar(r2, social),
                Engine.TensorSubtract(FilterStatisticTensorOps.RowBroadcast(globalBest, _swarmSize), positions));
            velocities = Engine.TensorClamp(
                Engine.TensorAdd(Engine.TensorAdd(Engine.TensorMultiplyScalar(velocities, inertia), toPersonal), toGlobal), minusOne, one);
            positions = Engine.TensorClamp(Engine.TensorAdd(positions, velocities), zero, one);
        }

        // Normalize by the largest value when it is positive
        double largest = Engine.TensorMaxValue(importances);
        if (largest > 0)
            importances = Engine.TensorDivideScalar(importances, largest);

        _featureImportances = importances.Cast<T>().ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(globalBest, numToSelect);
        IsFitted = true;
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
            throw new InvalidOperationException("ParticleSwarmSelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ParticleSwarmSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ParticleSwarmSelector has not been fitted.");

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
