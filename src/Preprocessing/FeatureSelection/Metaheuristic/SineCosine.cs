using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Sine Cosine Algorithm (SCA) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Sine Cosine Algorithm uses sine and cosine functions to create a mathematical
/// model for optimization. Agents oscillate around the best solution, with the
/// oscillation range decreasing over iterations.
/// </para>
/// <para><b>For Beginners:</b> Imagine solutions swinging back and forth around the
/// best known answer, like a pendulum. Early on, the swings are big (exploring widely),
/// but they get smaller over time (focusing on the best area). The sine and cosine
/// functions create this natural oscillating behavior.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SineCosine<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nIterations;
    private readonly double _a; // Amplitude parameter
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int NIterations => _nIterations;
    public double Amplitude => _a;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SineCosine(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int nIterations = 100,
        double a = 2.0,
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
        _a = a;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SineCosine requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize positions
        var positions = PopulationSearchOps.Uniform<T>(random, _populationSize, p);
        var fitness = PopulationSearchOps.ThresholdFitness(positions, scores, _nFeaturesToSelect, 0.1);

        // The destination is the best solution found so far
        int bestIdx = PopulationSearchOps.ArgMax(fitness);
        var bestFitness = PopulationSearchOps.At(fitness, bestIdx);
        var bestPosition = PopulationSearchOps.Row(positions, bestIdx);

        // Main loop (Mirjalili 2016): every agent moves against the same destination, then the destination is updated
        for (int iter = 0; iter < _nIterations; iter++)
        {
            double r1 = ExplorationScale(iter, _nIterations, _a);

            // Per agent and dimension: r2, r3 and r4, in the order of the scalar draws
            var draws = PopulationSearchOps.Uniform<T>(random, _populationSize, p, 3);
            positions = Move(positions, bestPosition, draws, r1);
            fitness = PopulationSearchOps.ThresholdFitness(positions, scores, _nFeaturesToSelect, 0.1);

            int iterBest = PopulationSearchOps.ArgMax(fitness);
            var iterBestFitness = PopulationSearchOps.At(fitness, iterBest);
            if (PopulationSearchOps.Greater(iterBestFitness, bestFitness))
            {
                bestFitness = iterBestFitness;
                bestPosition = PopulationSearchOps.Row(positions, iterBest);
            }
        }

        // Extract selected features from best position
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(
            Engine.TensorMultiply(bestPosition, scores), _nFeaturesToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// r1 = a - t a / T of Mirjalili (2016) for the 1-based iteration t = <paramref name="iter"/> + 1, so it falls
    /// linearly to exactly 0 on the last of the <paramref name="iterations"/> updates (SCA.m counts the
    /// initialisation as iteration 1 and its updates as t = 2 .. T).
    /// </summary>
    internal static double ExplorationScale(int iter, int iterations, double a) => a - (iter + 1) * a / iterations;

    /// <summary>
    /// The sine-cosine update X = X + r1 sin(r2) |r3 P - X| when r4 &lt; 0.5 and X + r1 cos(r2) |r3 P - X| otherwise,
    /// with r2 = 2 pi u, r3 = 2 u', clamped to [0, 1]. <paramref name="draws"/> is [m, p, 3] with the uniforms for
    /// r2, r3 and r4.
    /// </summary>
    internal static Tensor<T> Move(Tensor<T> positions, Tensor<T> destination, Tensor<T> draws, double r1)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int m = positions.Shape[0];
        int p = positions.Shape[1];
        var shape = new[] { m, p };
        var r2 = engine.TensorMultiplyScalar(
            engine.Reshape(engine.TensorSlice(draws, new[] { 0, 0, 0 }, new[] { m, p, 1 }), shape), ops.FromDouble(2 * Math.PI));
        var r3 = engine.TensorMultiplyScalar(
            engine.Reshape(engine.TensorSlice(draws, new[] { 0, 0, 1 }, new[] { m, p, 1 }), shape), ops.FromDouble(2));
        var r4 = engine.Reshape(engine.TensorSlice(draws, new[] { 0, 0, 2 }, new[] { m, p, 1 }), shape);

        var target = engine.TensorBroadcastTo(engine.Reshape(destination, new[] { 1, p }), shape);
        var distance = engine.TensorAbs(engine.TensorSubtract(engine.TensorMultiply(r3, target), positions));
        var wave = engine.TensorWhere(
            engine.TensorLessThan(r4, ops.FromDouble(0.5)), engine.TensorSin(r2), engine.TensorCos(r2));
        var step = engine.TensorMultiplyScalar(engine.TensorMultiply(wave, distance), ops.FromDouble(r1));
        return PopulationSearchOps.Clamp01(engine.TensorAdd(positions, step));
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
            throw new InvalidOperationException("SineCosine has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SineCosine does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SineCosine has not been fitted.");

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
