using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Particle Swarm Optimization for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Particle Swarm Optimization (PSO) simulates a swarm of particles moving through
/// the search space. Each particle represents a candidate feature subset and moves
/// based on its own best position and the swarm's best position.
/// </para>
/// <para><b>For Beginners:</b> Imagine a flock of birds searching for food. Each bird
/// remembers where it found the best food, and the flock shares information about
/// the best location found by any bird. They balance exploring new areas with
/// returning to known good spots. Here, "food" is the best feature subset.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ParticleSwarmFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nParticles;
    private readonly int _nIterations;
    private readonly double _w; // Inertia weight
    private readonly double _c1; // Cognitive coefficient
    private readonly double _c2; // Social coefficient
    private readonly Func<Matrix<T>, Vector<T>, int[], T>? _scorer;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double InertiaWeight => _w;
    public double CognitiveCoefficient => _c1;
    public double SocialCoefficient => _c2;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ParticleSwarmFS(
        int nFeaturesToSelect = 10,
        int nParticles = 30,
        int nIterations = 100,
        double inertiaWeight = 0.7,
        double cognitiveCoefficient = 1.5,
        double socialCoefficient = 1.5,
        Func<Matrix<T>, Vector<T>, int[], T>? scorer = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nParticles = nParticles;
        _nIterations = nIterations;
        _w = inertiaWeight;
        _c1 = cognitiveCoefficient;
        _c2 = socialCoefficient;
        _scorer = scorer;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ParticleSwarmFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        var scorer = _scorer ?? DefaultScorer;

        // Initialize particles: positions uniform on [0, 1), velocities uniform on [-0.05, 0.05), drawn from the
        // selector's generator in master's order
        var positions = new Tensor<T>[_nParticles];
        var velocities = new Tensor<T>[_nParticles];
        var pBest = new Tensor<T>[_nParticles]; // Personal best positions
        var pBestFitness = new T[_nParticles];
        var gBest = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero); // Global best position
        T gBestFitness = NumOps.FromDouble(double.NegativeInfinity);
        T inertia = NumOps.FromDouble(_w);
        T cognitive = NumOps.FromDouble(_c1);
        T social = NumOps.FromDouble(_c2);
        T zero = NumOps.Zero;
        T one = NumOps.One;
        T minusOne = NumOps.FromDouble(-1);
        T velocityScale = NumOps.FromDouble(0.1);
        T velocityOffset = NumOps.FromDouble(-0.05);

        // Initialize
        for (int i = 0; i < _nParticles; i++)
        {
            // Per dimension: the position draw, then the velocity draw (master's order)
            var draws = WrapperScoringOps.Draws<T>(random, 2 * p);
            positions[i] = WrapperScoringOps.Strided(draws, 2, 0);
            velocities[i] = WrapperScoringOps.Affine(WrapperScoringOps.Strided(draws, 2, 1), velocityScale, velocityOffset);
            pBest[i] = positions[i];

            pBestFitness[i] = EvaluatePosition(data, target, positions[i], scorer);

            if (WrapperScoringOps.IsAbove(pBestFitness[i], gBestFitness))
            {
                gBestFitness = pBestFitness[i];
                gBest = positions[i];
            }
        }

        // PSO main loop
        for (int iter = 0; iter < _nIterations; iter++)
        {
            for (int i = 0; i < _nParticles; i++)
            {
                // Update velocity: w v + c1 r1 (pBest - x) + c2 r2 (gBest - x), clamped to [-1, 1]
                // Per dimension: r1, then r2 (master's order)
                var draws = WrapperScoringOps.Draws<T>(random, 2 * p);
                var r1 = WrapperScoringOps.Strided(draws, 2, 0);
                var r2 = WrapperScoringOps.Strided(draws, 2, 1);
                var velocity = Engine.TensorAdd(Engine.TensorMultiplyScalar(velocities[i], inertia), Engine.TensorAdd(
                    Engine.TensorMultiplyScalar(Engine.TensorMultiply(r1, Engine.TensorSubtract(pBest[i], positions[i])), cognitive),
                    Engine.TensorMultiplyScalar(Engine.TensorMultiply(r2, Engine.TensorSubtract(gBest, positions[i])), social)));
                velocities[i] = Engine.TensorClamp(velocity, minusOne, one);

                // Update position, clamped to [0, 1]
                positions[i] = Engine.TensorClamp(Engine.TensorAdd(positions[i], velocities[i]), zero, one);

                // Evaluate new position
                T fitness = EvaluatePosition(data, target, positions[i], scorer);

                // Update personal best
                if (WrapperScoringOps.IsAbove(fitness, pBestFitness[i]))
                {
                    pBestFitness[i] = fitness;
                    pBest[i] = positions[i];

                    // Update global best
                    if (WrapperScoringOps.IsAbove(fitness, gBestFitness))
                    {
                        gBestFitness = fitness;
                        gBest = positions[i];
                    }
                }
            }
        }

        // Extract selected features from global best
        _featureScores = gBest.ToVector();
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(gBest, numToSelect);

        IsFitted = true;
    }

    private T EvaluatePosition(Matrix<T> data, Vector<T> target, Tensor<T> position,
        Func<Matrix<T>, Vector<T>, int[], T> scorer)
    {
        // Convert continuous position to binary selection (threshold = 0.5)
        var selected = PreprocessingTensorOps.MaskPositions(Engine.TensorGreaterThan(position, NumOps.FromDouble(0.5)));

        // If no features selected, take top by position value
        if (selected.Length == 0)
            selected = new[] { PreprocessingTensorOps.FirstArgMax(position) };

        return scorer(data, target, selected);
    }

    private T DefaultScorer(Matrix<T> data, Vector<T> target, int[] featureIndices)
    {
        if (featureIndices.Length == 0)
            return NumOps.FromDouble(double.NegativeInfinity);

        // Correlation-based score: mean |Pearson r| with the target
        return WrapperScoringOps.MeanAbsTargetCorrelation(data, target, featureIndices);
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
            throw new InvalidOperationException("ParticleSwarmFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ParticleSwarmFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ParticleSwarmFS has not been fitted.");

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
