using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Bat Algorithm for feature selection optimization.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Bat Algorithm metaheuristic inspired by the echolocation behavior of bats.
/// Bats emit pulses and adjust their frequency, loudness, and pulse rate to hunt prey.
/// </para>
/// <para><b>For Beginners:</b> Bats use echolocation to find food. In this algorithm,
/// each bat represents a feature subset. Bats fly toward better solutions (prey)
/// by adjusting their "frequency" (search behavior) and "loudness" (exploration vs
/// exploitation balance).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BatAlgorithmFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _maxIterations;
    private readonly double _fMin;  // Minimum frequency
    private readonly double _fMax;  // Maximum frequency
    private readonly double _loudness;  // Initial loudness
    private readonly double _pulseRate;  // Initial pulse rate
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>, int[], T>? _fitnessFunction;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BatAlgorithmFS(
        int nFeaturesToSelect = 10,
        int populationSize = 30,
        int maxIterations = 50,
        double fMin = 0.0,
        double fMax = 2.0,
        double loudness = 0.5,
        double pulseRate = 0.5,
        Func<Matrix<T>, Vector<T>, int[], T>? fitnessFunction = null,
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
        _maxIterations = maxIterations;
        _fMin = fMin;
        _fMax = fMax;
        _loudness = loudness;
        _pulseRate = pulseRate;
        _fitnessFunction = fitnessFunction;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BatAlgorithmFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize bats: positions uniform on [0, 1), drawn from the selector's generator in master's order;
        // zero velocities
        var positions = new Tensor<T>[_populationSize];
        var velocities = new Tensor<T>[_populationSize];
        var fitness = new T[_populationSize];
        var loudness = new T[_populationSize];
        var pulseRate = new T[_populationSize];
        T initialLoudness = NumOps.FromDouble(_loudness);
        T initialPulseRate = NumOps.FromDouble(_pulseRate);

        for (int i = 0; i < _populationSize; i++)
        {
            var draws = WrapperScoringOps.NextDoubles(random, p);
            positions[i] = WrapperScoringOps.FromDoubles<T>(p, j => draws[j]);
            velocities[i] = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
            fitness[i] = EvaluateSolution(positions[i], data, target);
            loudness[i] = initialLoudness;
            pulseRate[i] = initialPulseRate;
        }

        // Find best bat (the first best)
        int bestIdx = PreprocessingTensorOps.FirstArgMax(new Tensor<T>(fitness, new[] { _populationSize }));
        var bestPosition = positions[bestIdx];
        T bestFitness = fitness[bestIdx];
        T zero = NumOps.Zero;
        T one = NumOps.One;
        T step = NumOps.FromDouble(0.01);
        T loudnessDecay = NumOps.FromDouble(0.9);

        // Optimization loop
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            T increasedPulseRate = WrapperScoringOps.Scalar<T>(_pulseRate * (1 - Math.Exp(-0.9 * iter)));
            for (int i = 0; i < _populationSize; i++)
            {
                // Generate new frequency
                T freq = WrapperScoringOps.Scalar<T>(_fMin + (_fMax - _fMin) * random.NextDouble());

                // Update velocity and position
                velocities[i] = Engine.TensorAdd(velocities[i],
                    Engine.TensorMultiplyScalar(Engine.TensorSubtract(positions[i], bestPosition), freq));
                var moved = Engine.TensorAdd(positions[i], velocities[i]);

                // Local search around the best bat in the dimensions whose draw exceeds the pulse rate:
                // best + 0.01 u loudness with u uniform on [-1, 1)
                // (master's draw order: per dimension the pulse draw, then the walk draw only when it fires)
                var local = new List<int>();
                var walkDraws = new double[p];
                for (int j = 0; j < p; j++)
                {
                    if (WrapperScoringOps.IsAbove(WrapperScoringOps.Scalar<T>(random.NextDouble()), pulseRate[i]))
                    {
                        local.Add(j);
                        walkDraws[j] = 2 * random.NextDouble() - 1;
                    }
                }

                if (local.Count > 0)
                {
                    var walk = Engine.TensorAdd(bestPosition, Engine.TensorMultiplyScalar(
                        WrapperScoringOps.FromDoubles<T>(p, j => walkDraws[j]), WrapperScoringOps.Multiply(step, loudness[i])));
                    moved = Engine.TensorWhere(WrapperScoringOps.Membership<T>(local.ToArray(), p), walk, moved);
                }

                positions[i] = Engine.TensorClamp(moved, zero, one);

                // Evaluate new solution
                T newFitness = EvaluateSolution(positions[i], data, target);

                // Accept if improved and random condition
                if (WrapperScoringOps.IsAbove(newFitness, fitness[i])
                    && WrapperScoringOps.IsBelow(WrapperScoringOps.Scalar<T>(random.NextDouble()), loudness[i]))
                {
                    fitness[i] = newFitness;
                    loudness[i] = WrapperScoringOps.Multiply(loudness[i], loudnessDecay);  // Decrease loudness
                    pulseRate[i] = increasedPulseRate;  // Increase pulse rate
                }

                // Update best
                if (WrapperScoringOps.IsAbove(fitness[i], bestFitness))
                {
                    bestFitness = fitness[i];
                    bestPosition = positions[i];
                }
            }
        }

        // Calculate feature importances from best solution
        _featureImportances = bestPosition.ToVector();

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(bestPosition, nToSelect);

        IsFitted = true;
    }

    private T EvaluateSolution(Tensor<T> position, Matrix<T> data, Vector<T> target)
    {
        var selectedFeatures = PreprocessingTensorOps.MaskPositions(Engine.TensorGreaterThan(position, NumOps.FromDouble(0.5)));

        if (selectedFeatures.Length == 0)
            return NumOps.Zero;

        if (_fitnessFunction is not null)
            return _fitnessFunction(data, target, selectedFeatures);

        return DefaultFitness(data, target, selectedFeatures);
    }

    private T DefaultFitness(Matrix<T> data, Vector<T> target, int[] features)
    {
        // Correlation-based fitness: mean |Pearson r| with the target, minus 0.01 per selected feature
        return NumOps.Subtract(WrapperScoringOps.MeanAbsTargetCorrelation(data, target, features),
            NumOps.FromDouble(0.01 * features.Length));
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
            throw new InvalidOperationException("BatAlgorithmFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BatAlgorithmFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BatAlgorithmFS has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
