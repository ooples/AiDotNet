using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Genetic;

/// <summary>
/// Differential Evolution based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses differential evolution algorithm to optimize feature subsets by combining
/// differences between population members to explore the solution space.
/// </para>
/// <para><b>For Beginners:</b> Differential evolution creates new solutions by taking
/// the difference between existing solutions and adding it to another. This helps
/// explore feature combinations more efficiently than random mutation.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DifferentialEvolutionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _mutationFactor;
    private readonly double _crossoverRate;
    private readonly int? _randomState;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int PopulationSize => _populationSize;
    public int NGenerations => _nGenerations;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DifferentialEvolutionSelector(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int nGenerations = 100,
        double mutationFactor = 0.8,
        double crossoverRate = 0.7,
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
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "DifferentialEvolutionSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Initialize population: random keys in [0, 1), the subset is the numToSelect largest keys
        var draws = new double[_populationSize * p];
        for (int k = 0; k < draws.Length; k++)
            draws[k] = rand.NextDouble();
        var initial = new Tensor<double>(draws, new[] { _populationSize, p }).Cast<T>();
        var population = new Tensor<T>[_populationSize];
        var fitness = new Tensor<T>[_populationSize];
        for (int i = 0; i < _populationSize; i++)
        {
            population[i] = Engine.Reshape(FeatureScoringTensorOps.Rows(initial, new[] { i }), new[] { p });
            fitness[i] = Fitness(population[i], correlation, numToSelect);
        }

        // the importances (each generation's fitness-weighted feature counts, summed over the generations) are summed
        // in double whatever T is, and normalised there: a running sum in Half stops taking a contribution c once it
        // passes 2048 c, which capped the frequently kept features and skewed their normalised ratios
        var importances = new Tensor<double>(new[] { p });
        T mutationFactor = NumOps.FromDouble(_mutationFactor);
        T zero = NumOps.Zero;
        T one = NumOps.One;

        // DE iterations
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            var generation = new Tensor<double>(new[] { p });
            for (int i = 0; i < _populationSize; i++)
            {
                // Select three random distinct individuals
                int a, b, c;
                do { a = rand.Next(_populationSize); } while (a == i);
                do { b = rand.Next(_populationSize); } while (b == i || b == a);
                do { c = rand.Next(_populationSize); } while (c == i || c == a || c == b);

                // Mutation and crossover: a coordinate crosses over with probability crossoverRate, and jRand always
                int jRand = rand.Next(p);
                var crossOver = new int[p];
                for (int j = 0; j < p; j++)
                    crossOver[j] = rand.NextDouble() < _crossoverRate || j == jRand ? 1 : 0;
                var mutant = Engine.TensorClamp(Engine.TensorAdd(population[a], Engine.TensorMultiplyScalar(
                    Engine.TensorSubtract(population[b], population[c]), mutationFactor)), zero, one);
                var trial = Engine.TensorWhere(new Tensor<int>(crossOver, new[] { p }).Cast<T>(), mutant, population[i]);

                // Selection
                var trialFitness = Fitness(trial, correlation, numToSelect);
                if (Engine.TensorGreaterThan(trialFitness, fitness[i]).Cast<int>().ToArray()[0] == 1)
                {
                    population[i] = trial;
                    fitness[i] = trialFitness;
                }

                // Track feature importance
                generation = Engine.TensorAdd(generation, Engine.TensorMultiply(
                    FeatureScoringTensorOps.TopKMask(population[i], numToSelect), Engine.TensorBroadcastTo(fitness[i], new[] { p }))
                    .Cast<double>());
            }

            importances = Engine.TensorAdd(importances, generation);
        }

        // Find best solution (the first with the highest fitness)
        int bestIdx = FilterStatisticTensorOps.FirstArgMax(Engine.TensorConcatenate(fitness, 0));
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(population[bestIdx], numToSelect);

        // Normalize by the largest value when it is positive
        double largest = Engine.TensorMaxValue(importances);
        if (largest > 0)
            importances = Engine.TensorDivideScalar(importances, largest);

        _featureImportances = importances.Cast<T>().ToVector();
        IsFitted = true;
    }

    // fitness ([1]) of the subset made of the numToSelect largest keys
    private static Tensor<T> Fitness(Tensor<T> keys, Tensor<T> correlation, int numToSelect)
    {
        int p = keys.Length;
        var mask = AiDotNetEngine.Current.Reshape(FeatureScoringTensorOps.TopKMask(keys, numToSelect), new[] { 1, p });
        return FeatureScoringTensorOps.MeanOverMask(mask, correlation);
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
            throw new InvalidOperationException("DifferentialEvolutionSelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DifferentialEvolutionSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DifferentialEvolutionSelector has not been fitted.");

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
