using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Metaheuristic;

/// <summary>
/// Bat Algorithm for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Bat Algorithm mimics the echolocation behavior of bats. Bats adjust their
/// pulse frequency and loudness as they approach prey, which translates to
/// exploration-exploitation balance in optimization.
/// </para>
/// <para><b>For Beginners:</b> Bats use sound pulses to find food. When far from
/// prey, they use loud, low-frequency sounds. As they get closer, they use
/// quieter, higher-frequency sounds. This algorithm mimics that behavior -
/// exploring broadly at first, then focusing on promising areas.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BatAlgorithm<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _frequencyMin;
    private readonly double _frequencyMax;
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

    public BatAlgorithm(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int maxIterations = 50,
        double frequencyMin = 0.0,
        double frequencyMax = 2.0,
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
        _frequencyMin = frequencyMin;
        _frequencyMax = frequencyMax;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BatAlgorithm requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        int targetCount = Math.Min(_nFeaturesToSelect, p);
        var engine = Engine;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var correlations = PopulationSearchOps.AbsTargetCorrelations(data, target, NumOps.Zero);
        Func<Tensor<T>, Tensor<T>> evaluate = rows => PopulationSearchOps.ThresholdFitness(rows, correlations, targetCount, 0.1);

        // Initialize bats: per bat its position, then its loudness A0 in [0, 2) and pulse rate r0 in [0, 1)
        var init = PopulationSearchOps.Uniform<T>(random, _populationSize, p + 2);
        var swarm = new BatSwarm(
            engine.TensorSlice(init, new[] { 0, 0 }, new[] { _populationSize, p }),
            engine.TensorMultiplyScalar(
                engine.Reshape(engine.TensorSlice(init, new[] { 0, p }, new[] { _populationSize, 1 }), new[] { _populationSize }),
                NumOps.FromDouble(2)),
            engine.Reshape(engine.TensorSlice(init, new[] { 0, p + 1 }, new[] { _populationSize, 1 }), new[] { _populationSize }),
            evaluate);

        // r_i^t = r_i^0 (1 - exp(-gamma t)) for t = 1, 2, ... (t counts the iterations done, so r never starts at 0)
        var pulseFactors = PulseRateFactors(_maxIterations, Gamma);

        for (int iter = 0; iter < _maxIterations; iter++)
        {
            var pulseFactor = PopulationSearchOps.At(pulseFactors, iter);
            for (int i = 0; i < _populationSize; i++)
            {
                var frequencyDraw = PopulationSearchOps.Uniform<T>(random, 1);
                var pulseDraw = PopulationSearchOps.Uniform<T>(random, 1);
                Tensor<T>? walk = PopulationSearchOps.Greater(pulseDraw, PopulationSearchOps.At(swarm.PulseRate, i))
                    ? PopulationSearchOps.Affine(PopulationSearchOps.Uniform<T>(random, p), 2, -1)
                    : null;
                var acceptDraw = PopulationSearchOps.Uniform<T>(random, 1);
                swarm.Fly(i, frequencyDraw, walk, acceptDraw, _frequencyMin, _frequencyMax, pulseFactor);
            }
        }

        _featureScores = swarm.Best.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(swarm.Best, numToSelect);

        IsFitted = true;
    }

    /// <summary>Loudness decay alpha of Yang (2010).</summary>
    internal const double Alpha = 0.9;

    /// <summary>Pulse-rate growth gamma of Yang (2010).</summary>
    internal const double Gamma = 0.9;

    /// <summary>The pulse-rate factors 1 - exp(-gamma t) for t = 1 .. <paramref name="iterations"/>.</summary>
    internal static Tensor<T> PulseRateFactors(int iterations, double gamma)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var t = engine.TensorArange<T>(1, iterations + 1, 1);
        return FeatureSelectionTensorOps.OneMinus(engine.TensorExp(engine.TensorMultiplyScalar(t, ops.FromDouble(-gamma))));
    }

    /// <summary>
    /// The bat population of Yang (2010): positions, velocities, fitness, loudness A_i, pulse rates r_i (and their
    /// initial values r_i^0), and the best solution x*.
    /// </summary>
    internal sealed class BatSwarm
    {
        private readonly Func<Tensor<T>, Tensor<T>> _evaluate;

        public BatSwarm(Tensor<T> positions, Tensor<T> loudness, Tensor<T> initialPulseRate, Func<Tensor<T>, Tensor<T>> evaluate)
        {
            _evaluate = evaluate;
            Positions = positions;
            Velocities = new Tensor<T>(positions.Shape.ToArray());
            Fitness = evaluate(positions);
            Loudness = loudness;
            InitialPulseRate = initialPulseRate;
            PulseRate = initialPulseRate;
            int best = PopulationSearchOps.ArgMax(Fitness);
            Best = PopulationSearchOps.Row(positions, best);
            BestFitness = PopulationSearchOps.At(Fitness, best);
        }

        public Tensor<T> Positions { get; private set; }

        public Tensor<T> Velocities { get; private set; }

        public Tensor<T> Fitness { get; private set; }

        public Tensor<T> Loudness { get; private set; }

        public Tensor<T> InitialPulseRate { get; }

        public Tensor<T> PulseRate { get; private set; }

        public Tensor<T> Best { get; private set; }

        public Tensor<T> BestFitness { get; private set; }

        /// <summary>
        /// One move of bat <paramref name="i"/>: f_i = fmin + (fmax - fmin) beta, v_i += (x_i - x*) f_i, candidate
        /// x_i + v_i; or, when <paramref name="walk"/> (epsilon in [-1, 1]^d) is given because the pulse draw
        /// exceeded r_i, the local solution x* + epsilon * mean(A). Candidates are clamped to [0, 1]. The candidate
        /// replaces x_i only when it is better and the acceptance draw is below A_i; then A_i = alpha A_i and
        /// r_i = r_i^0 (1 - exp(-gamma t)). A rejected candidate leaves x_i, A_i and r_i as they were. x* tracks the
        /// best candidate seen.
        /// </summary>
        public void Fly(
            int i, Tensor<T> frequencyDraw, Tensor<T>? walk, Tensor<T> acceptDraw, double frequencyMin, double frequencyMax,
            Tensor<T> pulseFactor)
        {
            var engine = AiDotNetEngine.Current;
            var ops = MathHelper.GetNumericOperations<T>();
            int p = Positions.Shape[1];

            var frequency = PopulationSearchOps.Affine(frequencyDraw, frequencyMax - frequencyMin, frequencyMin);
            var position = PopulationSearchOps.Row(Positions, i);
            var velocity = engine.TensorAdd(
                PopulationSearchOps.Row(Velocities, i),
                PopulationSearchOps.Scale(engine.TensorSubtract(position, Best), frequency));
            Velocities = PopulationSearchOps.ReplaceRow(Velocities, i, velocity);

            var candidate = walk is null
                ? engine.TensorAdd(position, velocity)
                : engine.TensorAdd(Best, engine.TensorMultiplyScalar(walk, engine.TensorMean(Loudness)));
            candidate = PopulationSearchOps.Clamp01(candidate);
            var candidateFitness = _evaluate(engine.Reshape(candidate, new[] { 1, p }));

            if (PopulationSearchOps.Greater(PopulationSearchOps.At(Loudness, i), acceptDraw)
                && PopulationSearchOps.Greater(candidateFitness, PopulationSearchOps.At(Fitness, i)))
            {
                Positions = PopulationSearchOps.ReplaceRow(Positions, i, candidate);
                Fitness = PopulationSearchOps.ReplaceAt(Fitness, i, candidateFitness);
                Loudness = PopulationSearchOps.ReplaceAt(
                    Loudness, i, engine.TensorMultiplyScalar(PopulationSearchOps.At(Loudness, i), ops.FromDouble(Alpha)));
                PulseRate = PopulationSearchOps.ReplaceAt(
                    PulseRate, i, engine.TensorMultiply(PopulationSearchOps.At(InitialPulseRate, i), pulseFactor));
            }

            if (PopulationSearchOps.Greater(candidateFitness, BestFitness))
            {
                Best = candidate;
                BestFitness = candidateFitness;
            }
        }
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
            throw new InvalidOperationException("BatAlgorithm has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BatAlgorithm does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BatAlgorithm has not been fitted.");

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
