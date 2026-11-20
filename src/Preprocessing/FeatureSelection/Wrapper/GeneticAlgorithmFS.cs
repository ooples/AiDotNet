using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Wrapper;

/// <summary>
/// Genetic Algorithm for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Genetic Algorithms (GA) evolve a population of candidate feature subsets using
/// biological-inspired operators: selection, crossover, and mutation. Better solutions
/// are more likely to survive and produce offspring, gradually improving the population.
/// </para>
/// <para><b>For Beginners:</b> Think of it as natural selection for features. Each
/// individual is a binary string (1=include feature, 0=exclude). The fittest individuals
/// (best feature subsets) "breed" to create new solutions. Over generations, the
/// population evolves toward better feature combinations.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GeneticAlgorithmFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _populationSize;
    private readonly int _nGenerations;
    private readonly double _crossoverRate;
    private readonly double _mutationRate;
    private readonly int _eliteSize;
    private readonly Func<Matrix<T>, Vector<T>, int[], T>? _scorer;
    private readonly int? _randomState;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double CrossoverRate => _crossoverRate;
    public double MutationRate => _mutationRate;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GeneticAlgorithmFS(
        int nFeaturesToSelect = 10,
        int populationSize = 50,
        int nGenerations = 100,
        double crossoverRate = 0.8,
        double mutationRate = 0.1,
        int eliteSize = 2,
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
        _crossoverRate = crossoverRate;
        _mutationRate = mutationRate;
        _eliteSize = eliteSize;
        _scorer = scorer;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "GeneticAlgorithmFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        int numToSelect = Math.Min(_nFeaturesToSelect, p);

        // Initialize population with random binary chromosomes
        var population = new bool[_populationSize][];
        var fitness = new T[_populationSize];

        for (int i = 0; i < _populationSize; i++)
        {
            population[i] = new bool[p];
            // Initialize with approximately _nFeaturesToSelect features
            var indices = Enumerable.Range(0, p)
                .OrderBy(_ => random.Next())
                .Take(numToSelect)
                .ToList();
            foreach (int idx in indices)
                population[i][idx] = true;

            fitness[i] = EvaluateChromosome(data, target, population[i], scorer);
        }

        // Evolution loop
        for (int gen = 0; gen < _nGenerations; gen++)
        {
            // Sort by fitness (descending)
            var sortedIndices = PreprocessingTensorOps.StableOrder(
                new Tensor<T>(fitness, new[] { _populationSize }), descending: true);

            var newPopulation = new bool[_populationSize][];
            var newFitness = new T[_populationSize];

            // Elitism: copy best individuals
            for (int i = 0; i < _eliteSize && i < _populationSize; i++)
            {
                newPopulation[i] = (bool[])population[sortedIndices[i]].Clone();
                newFitness[i] = fitness[sortedIndices[i]];
            }

            // Fill rest of population with offspring
            for (int i = _eliteSize; i < _populationSize; i++)
            {
                // Tournament selection for parents
                int parent1 = TournamentSelection(fitness, random);
                int parent2 = TournamentSelection(fitness, random);

                bool[] offspring;
                if (random.NextDouble() < _crossoverRate)
                    offspring = Crossover(population[parent1], population[parent2], random);
                else
                    offspring = (bool[])population[parent1].Clone();

                // Mutation
                Mutate(offspring, random);

                // Ensure at least one feature is selected
                if (!offspring.Any(x => x))
                    offspring[random.Next(p)] = true;

                newPopulation[i] = offspring;
                newFitness[i] = EvaluateChromosome(data, target, offspring, scorer);
            }

            population = newPopulation;
            fitness = newFitness;
        }

        // Find best solution (the first best)
        var fitnessTensor = new Tensor<T>(fitness, new[] { _populationSize });
        int bestIdx = PreprocessingTensorOps.FirstArgMax(fitnessTensor);

        // Compute feature frequency scores: the summed fitness of the chromosomes that select each feature
        var featureScores = WrapperScoringOps.ScoreSums(
            population.Select(WrapperScoringOps.Positions).ToList(), fitnessTensor, p);
        _featureScores = featureScores.ToVector();

        // Select from best chromosome, respecting _nFeaturesToSelect
        var bestChrom = population[bestIdx];
        var selectedFromBest = new List<int>();
        for (int j = 0; j < p; j++)
        {
            if (bestChrom[j])
                selectedFromBest.Add(j);
        }

        // If too many, take top by feature score
        if (selectedFromBest.Count > numToSelect)
        {
            var candidates = selectedFromBest.ToArray();
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(
                WrapperScoringOps.Gather(featureScores, candidates), numToSelect)
                .Select(position => candidates[position])
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = [.. selectedFromBest.OrderBy(x => x)];
        }

        IsFitted = true;
    }

    private int TournamentSelection(T[] fitness, Random random, int tournamentSize = 3)
    {
        int best = random.Next(_populationSize);
        for (int i = 1; i < tournamentSize; i++)
        {
            int competitor = random.Next(_populationSize);
            if (WrapperScoringOps.IsAbove(fitness[competitor], fitness[best]))
                best = competitor;
        }
        return best;
    }

    private bool[] Crossover(bool[] parent1, bool[] parent2, Random random)
    {
        int p = parent1.Length;
        var offspring = new bool[p];

        // Two-point crossover
        int point1 = random.Next(p);
        int point2 = random.Next(p);
        if (point1 > point2)
            (point1, point2) = (point2, point1);

        for (int j = 0; j < p; j++)
        {
            if (j >= point1 && j <= point2)
                offspring[j] = parent2[j];
            else
                offspring[j] = parent1[j];
        }

        return offspring;
    }

    private void Mutate(bool[] chromosome, Random random)
    {
        for (int j = 0; j < chromosome.Length; j++)
        {
            if (random.NextDouble() < _mutationRate)
                chromosome[j] = !chromosome[j];
        }
    }

    private T EvaluateChromosome(Matrix<T> data, Vector<T> target, bool[] chromosome,
        Func<Matrix<T>, Vector<T>, int[], T> scorer)
    {
        var selected = new List<int>();
        for (int j = 0; j < chromosome.Length; j++)
        {
            if (chromosome[j])
                selected.Add(j);
        }

        if (selected.Count == 0)
            return NumOps.FromDouble(double.NegativeInfinity);

        return scorer(data, target, [.. selected]);
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
            throw new InvalidOperationException("GeneticAlgorithmFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GeneticAlgorithmFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GeneticAlgorithmFS has not been fitted.");

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
