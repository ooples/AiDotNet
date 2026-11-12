using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Genetic;

/// <summary>
/// Genetic Algorithm based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses a genetic algorithm to evolve feature subsets, optimizing for
/// predictive performance through selection, crossover, and mutation.
/// </para>
/// <para><b>For Beginners:</b> Like natural evolution, this creates a population
/// of different feature combinations, keeps the best ones, combines them to make
/// new combinations, and occasionally mutates them. Over generations, it finds
/// good feature subsets.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GeneticAlgorithmSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _mutationRate;
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

    public GeneticAlgorithmSelector(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int nGenerations = 100,
        double mutationRate = 0.1,
        double crossoverRate = 0.8,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _populationSize = populationSize;
        _nGenerations = nGenerations;
        _mutationRate = mutationRate;
        _crossoverRate = crossoverRate;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GeneticAlgorithmSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var correlation = FeatureSelectionTensorOps.AbsoluteCorrelation(
            FilterStatisticTensorOps.AsTensor(data), FilterStatisticTensorOps.AsTensor(target));
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Initialize population
        var population = new bool[_populationSize][];
        for (int i = 0; i < _populationSize; i++)
        {
            population[i] = new bool[p];
            var indices = Enumerable.Range(0, p).OrderBy(_ => rand.Next()).Take(numToSelect).ToList();
            foreach (int idx in indices)
                population[i][idx] = true;
        }

        var importances = new Tensor<T>(new[] { p });

        // Evolution
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            // Evaluate fitness, and add each member's fitness to the importance of the features it keeps
            var masks = MaskTensor(population, p);
            var fitness = FeatureSelectionTensorOps.MeanOverMask(masks, correlation);
            importances = Engine.TensorAdd(importances, Engine.Reshape(
                Engine.TensorMatMul(Engine.TensorTranspose(masks), Engine.Reshape(fitness, new[] { _populationSize, 1 })), new[] { p }));
            var rank = FeatureSelectionTensorOps.LessCounts(fitness);

            // Selection (tournament)
            var newPopulation = new bool[_populationSize][];
            for (int i = 0; i < _populationSize; i++)
            {
                int idx1 = rand.Next(_populationSize);
                int idx2 = rand.Next(_populationSize);
                newPopulation[i] = (bool[])population[rank[idx1] > rank[idx2] ? idx1 : idx2].Clone();
            }

            // Crossover
            for (int i = 0; i < _populationSize - 1; i += 2)
            {
                if (rand.NextDouble() < _crossoverRate)
                {
                    int crossPoint = rand.Next(p);
                    for (int j = crossPoint; j < p; j++)
                        (newPopulation[i][j], newPopulation[i + 1][j]) = (newPopulation[i + 1][j], newPopulation[i][j]);
                }
            }

            // Mutation
            for (int i = 0; i < _populationSize; i++)
            {
                for (int j = 0; j < p; j++)
                {
                    if (rand.NextDouble() < _mutationRate)
                        newPopulation[i][j] = !newPopulation[i][j];
                }
            }

            population = newPopulation;
        }

        // Normalize by the largest value when it is positive
        var largest = PreprocessingTensorOps.ColumnMax(Engine.Reshape(importances, new[] { p, 1 }));
        if (Engine.TensorGreaterThan(largest, NumOps.Zero).Cast<int>().ToArray()[0] == 1)
            importances = Engine.TensorDivide(importances, Engine.TensorBroadcastTo(largest, new[] { p }));

        _featureImportances = importances.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importances, numToSelect);
        IsFitted = true;
    }

    // 0/1 [m, p] tensor of a population of feature masks
    private static Tensor<T> MaskTensor(bool[][] population, int p)
    {
        var flat = new int[population.Length * p];
        for (int i = 0; i < population.Length; i++)
        {
            for (int j = 0; j < p; j++)
                flat[i * p + j] = population[i][j] ? 1 : 0;
        }

        return new Tensor<int>(flat, new[] { population.Length, p }).Cast<T>();
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
            throw new InvalidOperationException("GeneticAlgorithmSelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GeneticAlgorithmSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GeneticAlgorithmSelector has not been fitted.");

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
