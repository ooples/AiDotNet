using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Differential Evolution for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Differential Evolution (DE) is an evolutionary optimization algorithm that uses
/// mutation based on the difference between population members. For feature selection,
/// real-valued vectors are converted to binary feature masks using thresholding.
/// </para>
/// <para><b>For Beginners:</b> DE evolves a population of candidate solutions. To create
/// new solutions, it takes differences between existing ones and adds them to others.
/// This creates mutations that explore the search space efficiently. For feature selection,
/// these continuous values are converted to yes/no decisions about each feature.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DifferentialEvolutionFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _mutationFactor; // F
    private readonly double _crossoverRate; // CR
    private readonly Func<Matrix<T>, Vector<T>, int[], T>? _scorer;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double MutationFactor => _mutationFactor;
    public double CrossoverRate => _crossoverRate;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DifferentialEvolutionFS(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int nGenerations = 100,
        double mutationFactor = 0.8,
        double crossoverRate = 0.7,
        Func<Matrix<T>, Vector<T>, int[], T>? scorer = null,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nGenerations = nGenerations;
        _mutationFactor = mutationFactor;
        _crossoverRate = crossoverRate;
        _scorer = scorer;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DifferentialEvolutionFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize population with random real-valued vectors in [0, 1), drawn from the selector's generator
        // in master's order
        var population = new Tensor<T>[_populationSize];
        var fitness = new T[_populationSize];

        for (int i = 0; i < _populationSize; i++)
        {
            population[i] = WrapperScoringOps.Draws<T>(random, p);
            fitness[i] = EvaluateVector(data, target, population[i], p, scorer);
        }

        int bestIdx = PreprocessingTensorOps.FirstArgMax(new Tensor<T>(fitness, new[] { _populationSize }));
        T bestFitness = fitness[bestIdx];
        T mutationFactor = NumOps.FromDouble(_mutationFactor);
        T zero = NumOps.Zero;
        T one = NumOps.One;

        // DE evolution
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            for (int i = 0; i < _populationSize; i++)
            {
                // Select three random individuals different from i
                var candidates = Enumerable.Range(0, _populationSize)
                    .Where(x => x != i)
                    .OrderBy(_ => random.Next())
                    .Take(3)
                    .ToArray();

                int r1 = candidates[0];
                int r2 = candidates[1];
                int r3 = candidates[2];

                // Mutation: v = r1 + F * (r2 - r3), clipped to [0, 1]
                var mutant = Engine.TensorClamp(Engine.TensorAdd(population[r1],
                    Engine.TensorMultiplyScalar(Engine.TensorSubtract(population[r2], population[r3]), mutationFactor)), zero, one);

                // Crossover: each dimension comes from the mutant with probability CR, and dimension jRand always does
                int jRand = random.Next(p); // Ensure at least one dimension from mutant
                var fromMutant = new List<int>();
                for (int j = 0; j < p; j++)
                {
                    if (random.NextDouble() < _crossoverRate || j == jRand)
                        fromMutant.Add(j);
                }

                var trial = Engine.TensorWhere(WrapperScoringOps.Membership<T>(fromMutant.ToArray(), p), mutant, population[i]);

                // Selection
                T trialFitness = EvaluateVector(data, target, trial, p, scorer);
                if (WrapperScoringOps.IsAtLeast(trialFitness, fitness[i]))
                {
                    population[i] = trial;
                    fitness[i] = trialFitness;

                    if (WrapperScoringOps.IsAbove(trialFitness, bestFitness))
                    {
                        bestFitness = trialFitness;
                        bestIdx = i;
                    }
                }
            }
        }

        // Extract best solution
        var best = population[bestIdx];
        _featureScores = best.ToVector();

        // Convert to binary selection
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(best, numToSelect);

        IsFitted = true;
    }

    private T EvaluateVector(Matrix<T> data, Vector<T> target, Tensor<T> vector,
        int p, Func<Matrix<T>, Vector<T>, int[], T> scorer)
    {
        // Select features with value > 0.5 (or top k if fewer)
        var selected = PreprocessingTensorOps.MaskPositions(Engine.TensorGreaterThan(vector, NumOps.FromDouble(0.5)));

        // If too few, add the highest scoring ones (by decreasing value, the lowest index first among ties)
        if (selected.Length == 0)
        {
            selected = PreprocessingTensorOps.StableOrder(vector, descending: true)
                .Take(Math.Min(_nFeaturesToSelect, p))
                .ToArray();
        }

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
            throw new InvalidOperationException("DifferentialEvolutionFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DifferentialEvolutionFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DifferentialEvolutionFS has not been fitted.");

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
