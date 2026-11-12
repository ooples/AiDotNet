using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Whale Optimization Algorithm (WOA) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// WOA mimics the bubble-net hunting strategy of humpback whales. It uses
/// encircling prey, bubble-net attacking, and random search phases.
/// </para>
/// <para><b>For Beginners:</b> Humpback whales hunt by swimming in spirals,
/// creating a "net" of bubbles around fish. This algorithm mimics that spiral
/// approach to find good solutions. It balances exploring new areas (random
/// search) with refining promising ones (spiral update).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WhaleOptimization<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _spiralConstant;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int MaxIterations => _maxIterations;
    public double SpiralConstant => _spiralConstant;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public WhaleOptimization(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int maxIterations = 50,
        double spiralConstant = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (populationSize < 2)
            throw new ArgumentException("Population size must be at least 2.", nameof(populationSize));
        if (maxIterations < 1)
            throw new ArgumentException("Max iterations must be at least 1.", nameof(maxIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _maxIterations = maxIterations;
        _spiralConstant = spiralConstant;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "WhaleOptimization requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize population
        var positions = PopulationSearchOps.Uniform<T>(random, _populationSize, p);
        var fitness = PopulationSearchOps.ThresholdFitness(positions, correlations, targetCount, 0.1);

        // The leader is the best solution found so far
        int bestIdx = PopulationSearchOps.ArgMax(fitness);
        var bestFitness = PopulationSearchOps.At(fitness, bestIdx);
        var bestPos = PopulationSearchOps.Row(positions, bestIdx);

        var randomWhales = new int[p];

        // Main loop (Mirjalili and Lewis 2016)
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // a falls linearly from 2 to 0, a2 from -1 to -2; the leader stays fixed while the pod moves
            double a = 2 - iter * 2.0 / _maxIterations;
            double a2 = -1 - (double)iter / _maxIterations;
            for (int i = 0; i < _populationSize; i++)
            {
                // r (for A), r2 (for C), the spiral draw and the strategy draw
                var draws = PopulationSearchOps.Uniform<T>(random, 4);
                Tensor<T>? randomWhale = null;
                if (Explores(draws, a))
                {
                    for (int j = 0; j < p; j++)
                        randomWhales[j] = random.Next(_populationSize);
                    randomWhale = PopulationSearchOps.GatherPerColumn(positions, randomWhales);
                }

                var moved = Move(PopulationSearchOps.Row(positions, i), bestPos, randomWhale, draws, a, a2, _spiralConstant);
                positions = PopulationSearchOps.ReplaceRow(positions, i, moved);
            }

            fitness = PopulationSearchOps.ThresholdFitness(positions, correlations, targetCount, 0.1);
            int iterBest = PopulationSearchOps.ArgMax(fitness);
            var iterBestFitness = PopulationSearchOps.At(fitness, iterBest);
            if (PopulationSearchOps.Greater(iterBestFitness, bestFitness))
            {
                bestFitness = iterBestFitness;
                bestPos = PopulationSearchOps.Row(positions, iterBest);
            }
        }

        _featureScores = bestPos.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(bestPos, numToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// True when a whale searches for prey around a random whale: the strategy draw is below 0.5 and |A| &gt;= 1, with
    /// A = 2 a r - a. <paramref name="draws"/> is (r, r2, spiral draw, strategy draw).
    /// </summary>
    internal static bool Explores(Tensor<T> draws, double a)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var strategy = PopulationSearchOps.At(draws, 3);
        var coefficientA = PopulationSearchOps.Affine(PopulationSearchOps.At(draws, 0), 2 * a, -a);
        return PopulationSearchOps.Any(engine.TensorLessThan(strategy, ops.FromDouble(0.5)))
            && !PopulationSearchOps.Any(engine.TensorLessThan(engine.TensorAbs(coefficientA), ops.One));
    }

    /// <summary>
    /// One whale move. With A = 2 a r - a, C = 2 r2 and l = (a2 - 1) u + 1: when the strategy draw is below 0.5,
    /// X = L - A |C L - X| where L is the leader (|A| &lt; 1, encircling) or <paramref name="randomWhale"/> (search for
    /// prey); otherwise the bubble-net spiral X = |X* - X| e^(b l) cos(2 pi l) + X*. The result is clamped to [0, 1].
    /// </summary>
    internal static Tensor<T> Move(
        Tensor<T> position, Tensor<T> leader, Tensor<T>? randomWhale, Tensor<T> draws, double a, double a2, double b)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var strategy = PopulationSearchOps.At(draws, 3);
        Tensor<T> moved;
        if (PopulationSearchOps.Any(engine.TensorLessThan(strategy, ops.FromDouble(0.5))))
        {
            var coefficientA = PopulationSearchOps.Affine(PopulationSearchOps.At(draws, 0), 2 * a, -a);
            var coefficientC = engine.TensorMultiplyScalar(PopulationSearchOps.At(draws, 1), ops.FromDouble(2));
            var reference = randomWhale ?? leader;
            var distance = engine.TensorAbs(engine.TensorSubtract(PopulationSearchOps.Scale(reference, coefficientC), position));
            moved = engine.TensorSubtract(reference, PopulationSearchOps.Scale(distance, coefficientA));
        }
        else
        {
            var l = PopulationSearchOps.Affine(PopulationSearchOps.At(draws, 2), a2 - 1, 1);
            var spiral = engine.TensorMultiply(
                engine.TensorExp(engine.TensorMultiplyScalar(l, ops.FromDouble(b))),
                engine.TensorCos(engine.TensorMultiplyScalar(l, ops.FromDouble(2 * Math.PI))));
            var distance = engine.TensorAbs(engine.TensorSubtract(leader, position));
            moved = engine.TensorAdd(PopulationSearchOps.Scale(distance, spiral), leader);
        }

        return PopulationSearchOps.Clamp01(moved);
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
            throw new InvalidOperationException("WhaleOptimization has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("WhaleOptimization does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WhaleOptimization has not been fitted.");

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
