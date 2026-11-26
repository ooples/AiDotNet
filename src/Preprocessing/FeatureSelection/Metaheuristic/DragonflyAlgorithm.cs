using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Dragonfly Algorithm (DA) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Dragonfly Algorithm is inspired by the static and dynamic swarming behaviors
/// of dragonflies. It uses five main principles: separation, alignment, cohesion,
/// attraction towards food, and distraction from enemies.
/// </para>
/// <para><b>For Beginners:</b> This algorithm mimics how dragonflies swarm. They
/// avoid collisions (separation), fly in the same direction (alignment), stay close
/// to the group (cohesion), move toward food (good solutions), and flee from enemies
/// (bad solutions). This balance of exploration and exploitation finds good features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DragonflyAlgorithm<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
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

    public DragonflyAlgorithm(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int nIterations = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 5)
            throw new ArgumentException("Population size must be at least 5.", nameof(populationSize));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DragonflyAlgorithm requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize dragonfly positions and step vectors in the search space [0, 1]
        var positions = PopulationSearchOps.Uniform<T>(random, _populationSize, p);
        var steps = PopulationSearchOps.Uniform<T>(random, _populationSize, p);

        Tensor<T>? food = null;
        Tensor<T>? foodFitness = null;
        Tensor<T>? enemy = null;
        Tensor<T>? enemyFitness = null;

        // Main loop (Mirjalili 2016)
        for (int iter = 1; iter <= _nIterations; iter++)
        {
            var weights = SwarmWeights.ForIteration(iter, _nIterations, PopulationSearchOps.Uniform<T>(random, 4));

            // Food: the best position found so far; enemy: the worst
            var fitness = PopulationSearchOps.ThresholdFitness(positions, scores, _nFeaturesToSelect, 0.1);
            int best = PopulationSearchOps.ArgMax(fitness);
            int worst = PopulationSearchOps.ArgMin(fitness);
            if (foodFitness is null || PopulationSearchOps.Greater(PopulationSearchOps.At(fitness, best), foodFitness))
            {
                foodFitness = PopulationSearchOps.At(fitness, best);
                food = PopulationSearchOps.Row(positions, best);
            }

            if (enemyFitness is null || PopulationSearchOps.Greater(enemyFitness, PopulationSearchOps.At(fitness, worst)))
            {
                enemyFitness = PopulationSearchOps.At(fitness, worst);
                enemy = PopulationSearchOps.Row(positions, worst);
            }

            if (food is null || enemy is null)
                throw new InvalidOperationException("The food and enemy positions must be set after the first evaluation.");

            for (int i = 0; i < _populationSize; i++)
                MoveDragonfly(ref positions, ref steps, i, food, enemy, weights, random);
        }

        var finalFitness = PopulationSearchOps.ThresholdFitness(positions, scores, _nFeaturesToSelect, 0.1);
        int finalBest = PopulationSearchOps.ArgMax(finalFitness);
        var foodPosition = food is not null && foodFitness is not null
            && !PopulationSearchOps.Greater(PopulationSearchOps.At(finalFitness, finalBest), foodFitness)
                ? food
                : PopulationSearchOps.Row(positions, finalBest);

        // Extract selected features from the food position
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(
            Engine.TensorMultiply(foodPosition, scores), _nFeaturesToSelect);

        IsFitted = true;
    }

    /// <summary>Maximum step length in each dimension, a tenth of the search range [0, 1].</summary>
    internal const double MaxStep = 0.1;

    /// <summary>
    /// The per-iteration weights of DA: neighbourhood radius r = 1/4 + 2 t / T, inertia w = 0.9 - 0.5 t / T,
    /// c' = max(0.1 - 0.2 t / T, 0), and s = 2 rand c', a = 2 rand c', c = 2 rand c', f = 2 rand, e = c'.
    /// </summary>
    internal sealed class SwarmWeights
    {
        public double Radius { get; private set; }

        public double Inertia { get; private set; }

        public Tensor<T> Separation { get; private set; } = new Tensor<T>(new[] { 1 });

        public Tensor<T> Alignment { get; private set; } = new Tensor<T>(new[] { 1 });

        public Tensor<T> Cohesion { get; private set; } = new Tensor<T>(new[] { 1 });

        public Tensor<T> Food { get; private set; } = new Tensor<T>(new[] { 1 });

        public double Enemy { get; private set; }

        /// <summary>The weights of iteration <paramref name="t"/> of <paramref name="iterations"/>; <paramref name="draws"/> holds the s, a, c and f uniforms.</summary>
        public static SwarmWeights ForIteration(int t, int iterations, Tensor<T> draws)
        {
            double progress = (double)t / iterations;
            double c = Math.Max(0.1 - 0.2 * progress, 0);
            return new SwarmWeights
            {
                Radius = 0.25 + 2 * progress,
                Inertia = 0.9 - 0.5 * progress,
                Separation = PopulationSearchOps.Affine(PopulationSearchOps.At(draws, 0), 2 * c, 0),
                Alignment = PopulationSearchOps.Affine(PopulationSearchOps.At(draws, 1), 2 * c, 0),
                Cohesion = PopulationSearchOps.Affine(PopulationSearchOps.At(draws, 2), 2 * c, 0),
                Food = PopulationSearchOps.Affine(PopulationSearchOps.At(draws, 3), 2, 0),
                Enemy = c,
            };
        }
    }

    /// <summary>
    /// Moves dragonfly <paramref name="i"/> as in DA. Neighbours are the dragonflies whose distance is at most r and
    /// nonzero in every dimension. S = -sum(X_j - X), A = mean(dX_j),
    /// C = mean(X_j) - X (A = dX and C = 0 with at most one neighbour, as in DA.m), F = food - X and E = enemy + X when food and enemy are within r in
    /// every dimension (else 0). When the food is out of reach, a dragonfly with more than one neighbour steps
    /// dX = w dX + rand A + rand C + rand S (a fresh rand per dimension), and one without does a Levy flight X = X + Levy .* X with dX = 0;
    /// otherwise dX = s S + a A + c C + f F + e E + w dX. Steps are clamped to [-0.1, 0.1] and positions to [0, 1].
    /// </summary>
    internal static void MoveDragonfly(
        ref Tensor<T> positions, ref Tensor<T> steps, int i, Tensor<T> food, Tensor<T> enemy, SwarmWeights weights, Random random)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = positions.Shape[0];
        int p = positions.Shape[1];
        T radius = ops.FromDouble(weights.Radius);
        T zero = ops.Zero;

        var x = PopulationSearchOps.Row(positions, i);
        var step = PopulationSearchOps.Row(steps, i);
        var distances = engine.TensorAbs(engine.TensorSubtract(
            positions, engine.TensorBroadcastTo(engine.Reshape(x, new[] { 1, p }), new[] { m, p })));
        var outside = engine.ReduceSum(engine.TensorGreaterThan(distances, radius), new[] { 1 }, false);
        var coincident = engine.ReduceSum(engine.TensorEquals(distances, zero), new[] { 1 }, false);
        var neighbour = engine.TensorMultiply(engine.TensorEquals(outside, zero), engine.TensorEquals(coincident, zero));
        int neighbourCount = engine.TensorCountNonzero(neighbour);

        var neighbourRow = engine.Reshape(neighbour, new[] { 1, m });
        var positionSum = engine.Reshape(engine.TensorMatMul(neighbourRow, positions), new[] { p });
        var stepSum = engine.Reshape(engine.TensorMatMul(neighbourRow, steps), new[] { p });
        T count = ops.FromDouble(Math.Max(neighbourCount, 1));
        var separation = engine.TensorNegate(engine.TensorSubtract(
            positionSum, engine.TensorMultiplyScalar(x, ops.FromDouble(neighbourCount))));
        // DA.m averages the neighbours only when there is more than one; a lone neighbour leaves A = dX and C = 0
        var alignment = neighbourCount > 1 ? engine.TensorDivideScalar(stepSum, count) : step;
        var cohesion = neighbourCount > 1
            ? engine.TensorSubtract(engine.TensorDivideScalar(positionSum, count), x)
            : new Tensor<T>(new[] { p });

        bool foodNear = !PopulationSearchOps.Any(engine.TensorGreaterThan(engine.TensorAbs(engine.TensorSubtract(x, food)), radius));
        bool enemyNear = !PopulationSearchOps.Any(engine.TensorGreaterThan(engine.TensorAbs(engine.TensorSubtract(x, enemy)), radius));
        var attraction = foodNear ? engine.TensorSubtract(food, x) : new Tensor<T>(new[] { p });
        var distraction = enemyNear ? engine.TensorAdd(enemy, x) : new Tensor<T>(new[] { p });
        var inertia = engine.TensorMultiplyScalar(step, ops.FromDouble(weights.Inertia));

        Tensor<T> newStep;
        Tensor<T> newPosition;
        if (!foodNear && neighbourCount <= 1)
        {
            // Levy flight of DA: 0.01 r1 sigma / |r2|^(1/beta) with uniform r1, r2 and beta = 3/2
            var levyDraws = PopulationSearchOps.Uniform<T>(random, 2, p);
            var r1 = PopulationSearchOps.Row(levyDraws, 0);
            var r2 = PopulationSearchOps.Row(levyDraws, 1);
            var levy = engine.TensorMultiplyScalar(
                engine.TensorDivide(r1, engine.TensorPower(engine.TensorAbs(r2), ops.FromDouble(1 / 1.5))),
                ops.FromDouble(0.01 * LevySigma));
            newPosition = engine.TensorAdd(x, engine.TensorMultiply(levy, x));
            newStep = new Tensor<T>(new[] { p });
        }
        else
        {
            Tensor<T> sum;
            if (!foodNear)
            {
                // DA.m draws rand for A, C and S per dimension: [p, 3] in that order
                var r = PopulationSearchOps.Uniform<T>(random, p, 3);
                Tensor<T> Column(int k) => engine.Reshape(engine.TensorSlice(r, new[] { 0, k }, new[] { p, 1 }), new[] { p });
                sum = engine.TensorAdd(
                    engine.TensorAdd(
                        engine.TensorMultiply(alignment, Column(0)),
                        engine.TensorMultiply(cohesion, Column(1))),
                    engine.TensorMultiply(separation, Column(2)));
            }
            else
            {
                sum = engine.TensorAdd(
                    engine.TensorAdd(
                        engine.TensorAdd(
                            PopulationSearchOps.Scale(separation, weights.Separation),
                            PopulationSearchOps.Scale(alignment, weights.Alignment)),
                        engine.TensorAdd(
                            PopulationSearchOps.Scale(cohesion, weights.Cohesion),
                            PopulationSearchOps.Scale(attraction, weights.Food))),
                    engine.TensorMultiplyScalar(distraction, ops.FromDouble(weights.Enemy)));
            }

            newStep = engine.TensorClamp(engine.TensorAdd(sum, inertia), ops.FromDouble(-MaxStep), ops.FromDouble(MaxStep));
            newPosition = engine.TensorAdd(x, newStep);
        }

        positions = PopulationSearchOps.ReplaceRow(positions, i, PopulationSearchOps.Clamp01(newPosition));
        steps = PopulationSearchOps.ReplaceRow(steps, i, newStep);
    }

    /// <summary>Mantegna's sigma for beta = 3/2: (Gamma(5/2) sin(3 pi / 4) / (Gamma(5/4) 3/2 2^(1/4)))^(2/3).</summary>
    private static readonly double LevySigma = Math.Pow(
        1.329340388179137 * Math.Sin(Math.PI * 0.75) / (0.906402477055477 * 1.5 * Math.Pow(2, 0.25)), 1 / 1.5);

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
            throw new InvalidOperationException("DragonflyAlgorithm has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DragonflyAlgorithm does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DragonflyAlgorithm has not been fitted.");

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
