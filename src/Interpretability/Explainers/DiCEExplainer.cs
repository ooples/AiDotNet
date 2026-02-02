using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Interpretability.Explainers;

/// <summary>
/// DiCE (Diverse Counterfactual Explanations) explainer using genetic algorithm-based search.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DiCE generates MULTIPLE diverse counterfactual explanations,
/// not just one. This is much more useful in practice because:
///
/// 1. <b>Multiple options:</b> "You could get a loan by EITHER increasing income OR
///    reducing debt" instead of just one path
/// 2. <b>Diversity:</b> The counterfactuals are intentionally different from each other
/// 3. <b>Actionability:</b> Respects constraints like "you can't become younger"
/// 4. <b>Realism:</b> Changes are minimal and realistic
///
/// <b>How DiCE works:</b>
/// Uses a genetic algorithm to evolve a population of candidate counterfactuals,
/// optimizing for:
/// - <b>Validity:</b> Must achieve the target prediction
/// - <b>Proximity:</b> Stay close to the original instance
/// - <b>Sparsity:</b> Change as few features as possible
/// - <b>Diversity:</b> Counterfactuals should differ from each other
///
/// <b>Example output:</b>
/// Original: Loan denied (income=$40k, debt=$30k, employed=No)
/// Counterfactual 1: Loan approved (income=$50k, debt=$30k, employed=No)
/// Counterfactual 2: Loan approved (income=$40k, debt=$20k, employed=No)
/// Counterfactual 3: Loan approved (income=$40k, debt=$30k, employed=Yes)
///
/// Each shows a DIFFERENT way to get approved!
/// </para>
/// </remarks>
public class DiCEExplainer<T> : ILocalExplainer<T, DiCEExplanation<T>>, IGPUAcceleratedExplainer<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly Func<Matrix<T>, Vector<T>> _predictFunction;
    private readonly int _numFeatures;
    private readonly int _numCounterfactuals;
    private readonly int _populationSize;
    private readonly int _maxGenerations;
    private readonly double _mutationRate;
    private readonly double _crossoverRate;
    private readonly double _diversityWeight;
    private readonly double _proximityWeight;
    private readonly double _sparsityWeight;
    private readonly double _targetThreshold;
    private readonly int? _randomState;
    private readonly string[]? _featureNames;
    private readonly T[]? _featureMins;
    private readonly T[]? _featureMaxs;
    private readonly bool[]? _featuresMutable;
    private readonly FeatureType[]? _featureTypes;
    private readonly T[]? _featureRanges;
    private GPUExplainerHelper<T>? _gpuHelper;

    /// <inheritdoc/>
    public string MethodName => "DiCE";

    /// <inheritdoc/>
    public bool SupportsLocalExplanations => true;

    /// <inheritdoc/>
    public bool SupportsGlobalExplanations => false;

    /// <inheritdoc/>
    public bool IsGPUAccelerated => _gpuHelper?.IsGPUEnabled ?? false;

    /// <inheritdoc/>
    public void SetGPUHelper(GPUExplainerHelper<T>? helper)
    {
        _gpuHelper = helper;
    }

    /// <summary>
    /// Initializes a new DiCE explainer.
    /// </summary>
    /// <param name="predictFunction">Model prediction function.</param>
    /// <param name="numFeatures">Number of input features.</param>
    /// <param name="numCounterfactuals">Number of diverse counterfactuals to generate (default: 4).</param>
    /// <param name="populationSize">Genetic algorithm population size (default: 50).</param>
    /// <param name="maxGenerations">Maximum generations (default: 100).</param>
    /// <param name="diversityWeight">Weight for diversity objective (default: 1.0).</param>
    /// <param name="proximityWeight">Weight for proximity objective (default: 1.0).</param>
    /// <param name="sparsityWeight">Weight for sparsity objective (default: 0.5).</param>
    /// <param name="targetThreshold">Threshold for classification (default: 0.5).</param>
    /// <param name="mutationRate">Probability of mutation (default: 0.1).</param>
    /// <param name="crossoverRate">Probability of crossover (default: 0.7).</param>
    /// <param name="featureNames">Optional feature names.</param>
    /// <param name="featureMins">Optional minimum values for features.</param>
    /// <param name="featureMaxs">Optional maximum values for features.</param>
    /// <param name="featuresMutable">Which features can be changed.</param>
    /// <param name="featureTypes">Type of each feature (continuous, categorical, etc.).</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b>
    /// - <b>numCounterfactuals:</b> How many different explanations you want (3-5 is typical)
    /// - <b>diversityWeight:</b> Higher = more different counterfactuals
    /// - <b>proximityWeight:</b> Higher = smaller changes from original
    /// - <b>sparsityWeight:</b> Higher = fewer features changed
    /// - <b>featuresMutable:</b> Mark age, race, etc. as immutable
    /// </para>
    /// </remarks>
    public DiCEExplainer(
        Func<Matrix<T>, Vector<T>> predictFunction,
        int numFeatures,
        int numCounterfactuals = 4,
        int populationSize = 50,
        int maxGenerations = 100,
        double diversityWeight = 1.0,
        double proximityWeight = 1.0,
        double sparsityWeight = 0.5,
        double targetThreshold = 0.5,
        double mutationRate = 0.1,
        double crossoverRate = 0.7,
        string[]? featureNames = null,
        T[]? featureMins = null,
        T[]? featureMaxs = null,
        bool[]? featuresMutable = null,
        FeatureType[]? featureTypes = null,
        int? randomState = null)
    {
        _predictFunction = predictFunction ?? throw new ArgumentNullException(nameof(predictFunction));
        _numFeatures = numFeatures;
        _numCounterfactuals = numCounterfactuals;
        _populationSize = populationSize;
        _maxGenerations = maxGenerations;
        _diversityWeight = diversityWeight;
        _proximityWeight = proximityWeight;
        _sparsityWeight = sparsityWeight;
        _targetThreshold = targetThreshold;
        _mutationRate = mutationRate;
        _crossoverRate = crossoverRate;
        _featureNames = featureNames;
        _featureMins = featureMins;
        _featureMaxs = featureMaxs;
        _featuresMutable = featuresMutable;
        _featureTypes = featureTypes;
        _randomState = randomState;

        // Compute feature ranges for normalization
        _featureRanges = ComputeFeatureRanges();
    }

    /// <summary>
    /// Generates diverse counterfactual explanations for an instance.
    /// </summary>
    /// <param name="instance">The input to explain.</param>
    /// <returns>DiCE explanation with multiple diverse counterfactuals.</returns>
    public DiCEExplanation<T> Explain(Vector<T> instance)
    {
        // Get original prediction and determine target
        var instanceMatrix = CreateSingleRowMatrix(instance);
        var originalPred = _predictFunction(instanceMatrix)[0];
        double originalValue = NumOps.ToDouble(originalPred);

        // Target is the opposite class
        double targetValue = originalValue >= _targetThreshold ? 0.0 : 1.0;

        return ExplainWithTarget(instance, targetValue);
    }

    /// <summary>
    /// Generates diverse counterfactual explanations for a specific target.
    /// </summary>
    /// <param name="instance">The input to explain.</param>
    /// <param name="targetValue">The desired prediction value.</param>
    /// <returns>DiCE explanation with multiple diverse counterfactuals.</returns>
    public DiCEExplanation<T> ExplainWithTarget(Vector<T> instance, double targetValue)
    {
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Get original prediction
        var instanceMatrix = CreateSingleRowMatrix(instance);
        var originalPred = _predictFunction(instanceMatrix)[0];

        // Initialize population
        var population = InitializePopulation(instance, rand);

        // Genetic algorithm main loop
        for (int gen = 0; gen < _maxGenerations; gen++)
        {
            // Evaluate fitness of all individuals
            var fitness = EvaluatePopulation(population, instance, targetValue);

            // Check early stopping: if we have enough valid diverse counterfactuals
            var validCount = CountValidCounterfactuals(population, targetValue);
            if (validCount >= _numCounterfactuals)
            {
                // Check if diversity is sufficient
                var topCfs = SelectTopCounterfactuals(population, fitness, instance, targetValue);
                if (ComputeSetDiversity(topCfs, instance) > 0.5)
                {
                    break;
                }
            }

            // Selection
            var selected = TournamentSelection(population, fitness, rand);

            // Crossover
            var offspring = ApplyCrossover(selected, instance, rand);

            // Mutation
            ApplyMutation(offspring, instance, rand);

            // Elitism: keep best individuals
            population = CombineWithElitism(population, offspring, fitness, instance, targetValue);
        }

        // Select final diverse counterfactuals
        var finalFitness = EvaluatePopulation(population, instance, targetValue);
        var counterfactuals = SelectDiverseCounterfactuals(population, finalFitness, instance, targetValue);

        // Create explanations
        var cfExplanations = new List<SingleCounterfactual<T>>();
        foreach (var cf in counterfactuals)
        {
            var cfMatrix = CreateSingleRowMatrix(cf);
            var cfPred = _predictFunction(cfMatrix);
            cfExplanations.Add(CreateSingleCounterfactual(instance, cf, cfPred[0]));
        }

        return new DiCEExplanation<T>(
            originalInstance: instance,
            originalPrediction: originalPred,
            targetPrediction: NumOps.FromDouble(targetValue),
            counterfactuals: cfExplanations,
            featureNames: _featureNames);
    }

    /// <inheritdoc/>
    public DiCEExplanation<T>[] ExplainBatch(Matrix<T> instances)
    {
        var results = new DiCEExplanation<T>[instances.Rows];
        for (int i = 0; i < instances.Rows; i++)
        {
            results[i] = Explain(instances.GetRow(i));
        }
        return results;
    }

    /// <summary>
    /// Initializes the population with random perturbations.
    /// </summary>
    private List<Vector<T>> InitializePopulation(Vector<T> original, Random rand)
    {
        var population = new List<Vector<T>>();

        for (int i = 0; i < _populationSize; i++)
        {
            var individual = new T[_numFeatures];

            for (int f = 0; f < _numFeatures; f++)
            {
                if (!IsMutable(f))
                {
                    individual[f] = original[f];
                    continue;
                }

                // Random perturbation
                double origVal = NumOps.ToDouble(original[f]);
                double range = GetFeatureRange(f);
                double perturbation = (rand.NextDouble() * 2 - 1) * range * 0.5;
                double newVal = ClipToRange(origVal + perturbation, f);

                individual[f] = NumOps.FromDouble(newVal);
            }

            population.Add(new Vector<T>(individual));
        }

        return population;
    }

    /// <summary>
    /// Evaluates fitness of all individuals in the population.
    /// </summary>
    private double[] EvaluatePopulation(List<Vector<T>> population, Vector<T> original, double targetValue)
    {
        var fitness = new double[population.Count];

        for (int i = 0; i < population.Count; i++)
        {
            fitness[i] = ComputeFitness(population[i], original, targetValue, population);
        }

        return fitness;
    }

    /// <summary>
    /// Computes fitness for a single individual.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Fitness combines multiple objectives:
    /// - Validity: Does it achieve the target prediction?
    /// - Proximity: How close is it to the original?
    /// - Sparsity: How few features changed?
    /// - Diversity: How different is it from other counterfactuals?
    /// </para>
    /// </remarks>
    private double ComputeFitness(Vector<T> individual, Vector<T> original, double targetValue, List<Vector<T>> population)
    {
        // Validity: Does it achieve the target?
        var pred = _predictFunction(CreateSingleRowMatrix(individual))[0];
        double predValue = NumOps.ToDouble(pred);
        double validityLoss = ComputeValidityLoss(predValue, targetValue);

        // Proximity: Distance from original
        double proximity = ComputeNormalizedDistance(original, individual);

        // Sparsity: Number of features changed
        double sparsity = ComputeSparsity(original, individual);

        // Diversity: Average distance from other individuals
        double diversity = ComputeDiversityScore(individual, population, original);

        // Combined fitness (lower is better for losses)
        double fitness = -validityLoss * 10  // Strong penalty for not achieving target
                        - _proximityWeight * proximity
                        - _sparsityWeight * sparsity
                        + _diversityWeight * diversity;  // Reward diversity

        return fitness;
    }

    /// <summary>
    /// Computes validity loss (how far from target prediction).
    /// </summary>
    private double ComputeValidityLoss(double predicted, double target)
    {
        bool reachedTarget = (target >= _targetThreshold && predicted >= _targetThreshold) ||
                            (target < _targetThreshold && predicted < _targetThreshold);

        if (reachedTarget)
        {
            return 0;
        }

        // Distance from target threshold
        if (target >= _targetThreshold)
        {
            return Math.Max(0, _targetThreshold - predicted);
        }
        else
        {
            return Math.Max(0, predicted - _targetThreshold);
        }
    }

    /// <summary>
    /// Computes normalized distance from original.
    /// </summary>
    private double ComputeNormalizedDistance(Vector<T> original, Vector<T> counterfactual)
    {
        double sumSq = 0;
        int changed = 0;

        for (int f = 0; f < _numFeatures; f++)
        {
            double origVal = NumOps.ToDouble(original[f]);
            double cfVal = NumOps.ToDouble(counterfactual[f]);
            double range = GetFeatureRange(f);

            if (Math.Abs(origVal - cfVal) > 1e-6)
            {
                double normalizedDiff = (origVal - cfVal) / (range + 1e-6);
                sumSq += normalizedDiff * normalizedDiff;
                changed++;
            }
        }

        return Math.Sqrt(sumSq);
    }

    /// <summary>
    /// Computes sparsity (fraction of features changed).
    /// </summary>
    private double ComputeSparsity(Vector<T> original, Vector<T> counterfactual)
    {
        int changed = 0;
        int mutableCount = 0;

        for (int f = 0; f < _numFeatures; f++)
        {
            if (!IsMutable(f)) continue;
            mutableCount++;

            double origVal = NumOps.ToDouble(original[f]);
            double cfVal = NumOps.ToDouble(counterfactual[f]);

            if (Math.Abs(origVal - cfVal) > 1e-6)
            {
                changed++;
            }
        }

        return mutableCount > 0 ? (double)changed / mutableCount : 0;
    }

    /// <summary>
    /// Computes diversity score for an individual.
    /// </summary>
    private double ComputeDiversityScore(Vector<T> individual, List<Vector<T>> population, Vector<T> original)
    {
        if (population.Count <= 1) return 1.0;

        double totalDist = 0;
        int count = 0;

        foreach (var other in population)
        {
            if (other == individual) continue;
            totalDist += ComputeNormalizedDistance(individual, other);
            count++;
        }

        return count > 0 ? totalDist / count : 0;
    }

    /// <summary>
    /// Computes diversity of a set of counterfactuals.
    /// </summary>
    private double ComputeSetDiversity(List<Vector<T>> counterfactuals, Vector<T> original)
    {
        if (counterfactuals.Count <= 1) return 0;

        double totalDist = 0;
        int pairs = 0;

        for (int i = 0; i < counterfactuals.Count; i++)
        {
            for (int j = i + 1; j < counterfactuals.Count; j++)
            {
                totalDist += ComputeNormalizedDistance(counterfactuals[i], counterfactuals[j]);
                pairs++;
            }
        }

        return pairs > 0 ? totalDist / pairs : 0;
    }

    /// <summary>
    /// Tournament selection for genetic algorithm.
    /// </summary>
    private List<Vector<T>> TournamentSelection(List<Vector<T>> population, double[] fitness, Random rand)
    {
        var selected = new List<Vector<T>>();
        int tournamentSize = 3;

        for (int i = 0; i < _populationSize; i++)
        {
            int best = rand.Next(population.Count);
            double bestFitness = fitness[best];

            for (int t = 1; t < tournamentSize; t++)
            {
                int candidate = rand.Next(population.Count);
                if (fitness[candidate] > bestFitness)
                {
                    best = candidate;
                    bestFitness = fitness[candidate];
                }
            }

            selected.Add(population[best].Clone());
        }

        return selected;
    }

    /// <summary>
    /// Applies crossover between selected individuals.
    /// </summary>
    private List<Vector<T>> ApplyCrossover(List<Vector<T>> selected, Vector<T> original, Random rand)
    {
        var offspring = new List<Vector<T>>();

        for (int i = 0; i < selected.Count - 1; i += 2)
        {
            if (rand.NextDouble() < _crossoverRate && i + 1 < selected.Count)
            {
                var (child1, child2) = SinglePointCrossover(selected[i], selected[i + 1], original, rand);
                offspring.Add(child1);
                offspring.Add(child2);
            }
            else
            {
                offspring.Add(selected[i].Clone());
                if (i + 1 < selected.Count)
                    offspring.Add(selected[i + 1].Clone());
            }
        }

        return offspring;
    }

    /// <summary>
    /// Single-point crossover between two parents.
    /// </summary>
    private (Vector<T>, Vector<T>) SinglePointCrossover(Vector<T> parent1, Vector<T> parent2, Vector<T> original, Random rand)
    {
        int crossoverPoint = rand.Next(1, _numFeatures);
        var child1 = new T[_numFeatures];
        var child2 = new T[_numFeatures];

        for (int f = 0; f < _numFeatures; f++)
        {
            if (!IsMutable(f))
            {
                child1[f] = original[f];
                child2[f] = original[f];
            }
            else if (f < crossoverPoint)
            {
                child1[f] = parent1[f];
                child2[f] = parent2[f];
            }
            else
            {
                child1[f] = parent2[f];
                child2[f] = parent1[f];
            }
        }

        return (new Vector<T>(child1), new Vector<T>(child2));
    }

    /// <summary>
    /// Applies mutation to offspring.
    /// </summary>
    private void ApplyMutation(List<Vector<T>> offspring, Vector<T> original, Random rand)
    {
        foreach (var individual in offspring)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                if (!IsMutable(f)) continue;
                if (rand.NextDouble() >= _mutationRate) continue;

                double range = GetFeatureRange(f);
                double origVal = NumOps.ToDouble(individual[f]);
                double perturbation = (rand.NextDouble() * 2 - 1) * range * 0.2;
                double newVal = ClipToRange(origVal + perturbation, f);

                individual[f] = NumOps.FromDouble(newVal);
            }
        }
    }

    /// <summary>
    /// Combines parents and offspring with elitism.
    /// </summary>
    private List<Vector<T>> CombineWithElitism(List<Vector<T>> parents, List<Vector<T>> offspring, double[] parentFitness, Vector<T> original, double targetValue)
    {
        // Keep top 10% of parents (elitism)
        int eliteCount = Math.Max(1, _populationSize / 10);
        var indices = Enumerable.Range(0, parents.Count)
            .OrderByDescending(i => parentFitness[i])
            .Take(eliteCount)
            .ToList();

        var nextGen = new List<Vector<T>>();
        foreach (var i in indices)
        {
            nextGen.Add(parents[i].Clone());
        }

        // Fill rest with offspring
        var offspringFitness = EvaluatePopulation(offspring, original, targetValue);
        var offspringIndices = Enumerable.Range(0, offspring.Count)
            .OrderByDescending(i => offspringFitness[i])
            .Take(_populationSize - eliteCount)
            .ToList();

        foreach (var i in offspringIndices)
        {
            nextGen.Add(offspring[i]);
        }

        return nextGen;
    }

    /// <summary>
    /// Counts valid counterfactuals that reach the target.
    /// </summary>
    private int CountValidCounterfactuals(List<Vector<T>> population, double targetValue)
    {
        int count = 0;
        foreach (var individual in population)
        {
            var pred = _predictFunction(CreateSingleRowMatrix(individual))[0];
            double predValue = NumOps.ToDouble(pred);
            bool valid = (targetValue >= _targetThreshold && predValue >= _targetThreshold) ||
                        (targetValue < _targetThreshold && predValue < _targetThreshold);
            if (valid) count++;
        }
        return count;
    }

    /// <summary>
    /// Selects top counterfactuals based on fitness.
    /// </summary>
    private List<Vector<T>> SelectTopCounterfactuals(List<Vector<T>> population, double[] fitness, Vector<T> original, double targetValue)
    {
        // Filter to valid counterfactuals
        var validIndices = new List<int>();
        for (int i = 0; i < population.Count; i++)
        {
            var pred = _predictFunction(CreateSingleRowMatrix(population[i]))[0];
            double predValue = NumOps.ToDouble(pred);
            bool valid = (targetValue >= _targetThreshold && predValue >= _targetThreshold) ||
                        (targetValue < _targetThreshold && predValue < _targetThreshold);
            if (valid) validIndices.Add(i);
        }

        if (validIndices.Count == 0)
        {
            // No valid counterfactuals, return best overall
            return Enumerable.Range(0, population.Count)
                .OrderByDescending(i => fitness[i])
                .Take(_numCounterfactuals)
                .Select(i => population[i])
                .ToList();
        }

        // Sort valid by fitness and return top
        return validIndices
            .OrderByDescending(i => fitness[i])
            .Take(_numCounterfactuals)
            .Select(i => population[i])
            .ToList();
    }

    /// <summary>
    /// Selects diverse counterfactuals using greedy selection.
    /// </summary>
    private List<Vector<T>> SelectDiverseCounterfactuals(List<Vector<T>> population, double[] fitness, Vector<T> original, double targetValue)
    {
        // Get valid counterfactuals sorted by fitness
        var candidates = new List<(int Index, double Fitness)>();
        for (int i = 0; i < population.Count; i++)
        {
            var pred = _predictFunction(CreateSingleRowMatrix(population[i]))[0];
            double predValue = NumOps.ToDouble(pred);
            bool valid = (targetValue >= _targetThreshold && predValue >= _targetThreshold) ||
                        (targetValue < _targetThreshold && predValue < _targetThreshold);
            if (valid)
            {
                candidates.Add((i, fitness[i]));
            }
        }

        if (candidates.Count == 0)
        {
            // No valid, return best overall
            return Enumerable.Range(0, population.Count)
                .OrderByDescending(i => fitness[i])
                .Take(_numCounterfactuals)
                .Select(i => population[i])
                .ToList();
        }

        // Greedy selection for diversity
        var selected = new List<Vector<T>>();
        var candidateList = candidates.OrderByDescending(c => c.Fitness).ToList();

        // Add best one first
        selected.Add(population[candidateList[0].Index]);
        candidateList.RemoveAt(0);

        // Greedily add most diverse remaining
        while (selected.Count < _numCounterfactuals && candidateList.Count > 0)
        {
            double maxMinDist = -1;
            int bestIdx = 0;

            for (int c = 0; c < candidateList.Count; c++)
            {
                double minDist = double.MaxValue;
                foreach (var s in selected)
                {
                    double dist = ComputeNormalizedDistance(population[candidateList[c].Index], s);
                    if (dist < minDist) minDist = dist;
                }

                if (minDist > maxMinDist)
                {
                    maxMinDist = minDist;
                    bestIdx = c;
                }
            }

            selected.Add(population[candidateList[bestIdx].Index]);
            candidateList.RemoveAt(bestIdx);
        }

        return selected;
    }

    /// <summary>
    /// Creates a single counterfactual result.
    /// </summary>
    private SingleCounterfactual<T> CreateSingleCounterfactual(Vector<T> original, Vector<T> counterfactual, T prediction)
    {
        var changes = new List<FeatureChange<T>>();

        for (int f = 0; f < _numFeatures; f++)
        {
            double origVal = NumOps.ToDouble(original[f]);
            double cfVal = NumOps.ToDouble(counterfactual[f]);

            if (Math.Abs(origVal - cfVal) > 1e-6)
            {
                changes.Add(new FeatureChange<T>(
                    featureIndex: f,
                    featureName: _featureNames?[f] ?? $"Feature_{f}",
                    originalValue: original[f],
                    counterfactualValue: counterfactual[f]));
            }
        }

        return new SingleCounterfactual<T>(
            counterfactualInstance: counterfactual,
            prediction: prediction,
            changes: changes,
            distance: ComputeNormalizedDistance(original, counterfactual));
    }

    // Helper methods

    private bool IsMutable(int featureIndex)
    {
        return _featuresMutable == null || _featuresMutable[featureIndex];
    }

    private double GetFeatureRange(int featureIndex)
    {
        if (_featureRanges != null && featureIndex < _featureRanges.Length)
        {
            return NumOps.ToDouble(_featureRanges[featureIndex]);
        }
        return 1.0;
    }

    private T[] ComputeFeatureRanges()
    {
        if (_featureMins == null || _featureMaxs == null)
        {
            return new T[_numFeatures];
        }

        var ranges = new T[_numFeatures];
        for (int f = 0; f < _numFeatures; f++)
        {
            ranges[f] = NumOps.Subtract(_featureMaxs[f], _featureMins[f]);
        }
        return ranges;
    }

    private double ClipToRange(double value, int featureIndex)
    {
        double min = _featureMins != null && featureIndex < _featureMins.Length
            ? NumOps.ToDouble(_featureMins[featureIndex]) : double.MinValue;
        double max = _featureMaxs != null && featureIndex < _featureMaxs.Length
            ? NumOps.ToDouble(_featureMaxs[featureIndex]) : double.MaxValue;
        return Math.Max(min, Math.Min(max, value));
    }

    private Matrix<T> CreateSingleRowMatrix(Vector<T> v)
    {
        return new Matrix<T>(new[] { v });
    }
}

/// <summary>
/// Type of feature for counterfactual constraints.
/// </summary>
public enum FeatureType
{
    /// <summary>Continuous numeric feature.</summary>
    Continuous,
    /// <summary>Categorical feature (discrete values).</summary>
    Categorical,
    /// <summary>Integer feature.</summary>
    Integer,
    /// <summary>Binary feature (0 or 1).</summary>
    Binary
}

/// <summary>
/// Represents a single feature change in a counterfactual.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class FeatureChange<T>
{
    /// <summary>Index of the feature that changed.</summary>
    public int FeatureIndex { get; }

    /// <summary>Name of the feature that changed.</summary>
    public string FeatureName { get; }

    /// <summary>Original value.</summary>
    public T OriginalValue { get; }

    /// <summary>Counterfactual value.</summary>
    public T CounterfactualValue { get; }

    /// <summary>Initializes a new feature change.</summary>
    public FeatureChange(int featureIndex, string featureName, T originalValue, T counterfactualValue)
    {
        FeatureIndex = featureIndex;
        FeatureName = featureName;
        OriginalValue = originalValue;
        CounterfactualValue = counterfactualValue;
    }

    /// <summary>Returns string representation.</summary>
    public override string ToString()
    {
        return $"{FeatureName}: {OriginalValue} → {CounterfactualValue}";
    }
}

/// <summary>
/// Represents a single counterfactual explanation.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class SingleCounterfactual<T>
{
    /// <summary>The counterfactual instance.</summary>
    public Vector<T> CounterfactualInstance { get; }

    /// <summary>Prediction for the counterfactual.</summary>
    public T Prediction { get; }

    /// <summary>Features that were changed.</summary>
    public IReadOnlyList<FeatureChange<T>> Changes { get; }

    /// <summary>Normalized distance from original.</summary>
    public double Distance { get; }

    /// <summary>Number of features changed.</summary>
    public int Sparsity => Changes.Count;

    /// <summary>Initializes a new single counterfactual.</summary>
    public SingleCounterfactual(Vector<T> counterfactualInstance, T prediction, List<FeatureChange<T>> changes, double distance)
    {
        CounterfactualInstance = counterfactualInstance;
        Prediction = prediction;
        Changes = changes;
        Distance = distance;
    }

    /// <summary>Returns string representation.</summary>
    public override string ToString()
    {
        return $"CF (pred={Prediction}, changes={Sparsity}, dist={Distance:F4}): " +
               string.Join(", ", Changes.Select(c => c.ToString()));
    }
}

/// <summary>
/// DiCE explanation containing multiple diverse counterfactuals.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class DiCEExplanation<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>Original input instance.</summary>
    public Vector<T> OriginalInstance { get; }

    /// <summary>Original prediction.</summary>
    public T OriginalPrediction { get; }

    /// <summary>Target prediction value.</summary>
    public T TargetPrediction { get; }

    /// <summary>Diverse counterfactual explanations.</summary>
    public IReadOnlyList<SingleCounterfactual<T>> Counterfactuals { get; }

    /// <summary>Feature names.</summary>
    public string[]? FeatureNames { get; }

    /// <summary>Number of counterfactuals generated.</summary>
    public int Count => Counterfactuals.Count;

    /// <summary>Average number of features changed across counterfactuals.</summary>
    public double AverageSparsity => Counterfactuals.Count > 0
        ? Counterfactuals.Average(cf => cf.Sparsity) : 0;

    /// <summary>Average distance from original across counterfactuals.</summary>
    public double AverageDistance => Counterfactuals.Count > 0
        ? Counterfactuals.Average(cf => cf.Distance) : 0;

    /// <summary>Initializes a new DiCE explanation.</summary>
    public DiCEExplanation(
        Vector<T> originalInstance,
        T originalPrediction,
        T targetPrediction,
        List<SingleCounterfactual<T>> counterfactuals,
        string[]? featureNames = null)
    {
        OriginalInstance = originalInstance;
        OriginalPrediction = originalPrediction;
        TargetPrediction = targetPrediction;
        Counterfactuals = counterfactuals;
        FeatureNames = featureNames;
    }

    /// <summary>Gets the counterfactual with fewest changes.</summary>
    public SingleCounterfactual<T>? GetSparsest()
    {
        return Counterfactuals.OrderBy(cf => cf.Sparsity).FirstOrDefault();
    }

    /// <summary>Gets the counterfactual closest to original.</summary>
    public SingleCounterfactual<T>? GetClosest()
    {
        return Counterfactuals.OrderBy(cf => cf.Distance).FirstOrDefault();
    }

    /// <summary>Gets features that appear in multiple counterfactuals.</summary>
    public IEnumerable<(string FeatureName, int Count)> GetCommonChanges()
    {
        var counts = new Dictionary<string, int>();

        foreach (var cf in Counterfactuals)
        {
            foreach (var change in cf.Changes)
            {
                if (!counts.ContainsKey(change.FeatureName))
                    counts[change.FeatureName] = 0;
                counts[change.FeatureName]++;
            }
        }

        return counts
            .OrderByDescending(kv => kv.Value)
            .Select(kv => (kv.Key, kv.Value));
    }

    /// <summary>Returns string representation.</summary>
    public override string ToString()
    {
        var lines = new List<string>
        {
            $"DiCE Explanation: {Count} diverse counterfactuals",
            $"  Original prediction: {OriginalPrediction}",
            $"  Target prediction: {TargetPrediction}",
            $"  Average sparsity: {AverageSparsity:F2} features changed",
            $"  Average distance: {AverageDistance:F4}",
            ""
        };

        for (int i = 0; i < Counterfactuals.Count; i++)
        {
            lines.Add($"  Counterfactual {i + 1}: {Counterfactuals[i]}");
        }

        var common = GetCommonChanges().Take(5).ToList();
        if (common.Count > 0)
        {
            lines.Add("");
            lines.Add($"  Most common changes: {string.Join(", ", common.Select(c => $"{c.FeatureName}({c.Count})"))}");
        }

        return string.Join(Environment.NewLine, lines);
    }
}
