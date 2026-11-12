using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

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
        Guard.NotNull(predictFunction);
        _predictFunction = predictFunction;
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

        // Target is the opposite class
        T targetValue = NumOps.GreaterThanOrEquals(originalPred, NumOps.FromDouble(_targetThreshold)) ? NumOps.Zero : NumOps.One;

        return ExplainWithTarget(instance, targetValue);
    }

    /// <summary>
    /// Generates diverse counterfactual explanations for a specific target.
    /// </summary>
    /// <param name="instance">The input to explain.</param>
    /// <param name="targetValue">The desired prediction value.</param>
    /// <returns>DiCE explanation with multiple diverse counterfactuals.</returns>
    public DiCEExplanation<T> ExplainWithTarget(Vector<T> instance, T targetValue)
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
            // Evaluate fitness of all individuals (ranked: equal fitness, equal rank)
            var fitness = ExplainerTensorOps<T>.DenseRanks(EvaluatePopulation(population, instance, targetValue));

            // Check early stopping: if we have enough valid diverse counterfactuals
            var validCount = CountValidCounterfactuals(population, targetValue);
            if (validCount >= _numCounterfactuals)
            {
                // Check if diversity is sufficient
                var topCfs = SelectTopCounterfactuals(population, fitness, instance, targetValue);
                if (HasSetDiversityAbove(topCfs, 0.5))
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
        var finalFitness = ExplainerTensorOps<T>.DenseRanks(EvaluatePopulation(population, instance, targetValue));
        var counterfactuals = SelectDiverseCounterfactuals(population, finalFitness, instance, targetValue);

        // Create explanations
        var cfExplanations = new List<SingleCounterfactual<T>>();
        if (counterfactuals.Count > 0)
        {
            // The model's first output for each counterfactual (one prediction call each, in order)
            var cfPredictions = Predict(counterfactuals).ToVector();
            cfExplanations.AddRange(counterfactuals.Zip(cfPredictions,
                (cf, prediction) => CreateSingleCounterfactual(instance, cf, prediction)));
        }

        return new DiCEExplanation<T>(
            originalInstance: instance,
            originalPrediction: originalPred,
            targetPrediction: targetValue,
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
    private Tensor<T> EvaluatePopulation(List<Vector<T>> population, Vector<T> original, T targetValue)
    {
        var engine = AiDotNetEngine.Current;
        int count = population.Count;
        if (count == 0)
        {
            return new Tensor<T>(new[] { 0 });
        }

        // Validity: Does it achieve the target?
        var validityLoss = ComputeValidityLoss(Predict(population), targetValue);

        // Proximity: Distance from original
        var proximity = PairwiseDistances(new List<Vector<T>> { original }, population).Reshape(new[] { count });

        // Sparsity: Fraction of mutable features changed
        var sparsity = ComputeSparsity(original, population);

        // Diversity: Average distance from the other individuals
        var diversity = count <= 1
            ? engine.TensorFullLike(proximity, NumOps.One)
            : engine.TensorDivideScalar(
                engine.ReduceSum(PairwiseDistances(population, population), new[] { 1 }, false), NumOps.FromDouble(count - 1));

        // Combined fitness (lower is better for losses)
        var fitness = engine.TensorMultiplyScalar(validityLoss, NumOps.FromDouble(-10)); // Strong penalty for not achieving target
        fitness = engine.TensorSubtract(fitness, engine.TensorMultiplyScalar(proximity, NumOps.FromDouble(_proximityWeight)));
        fitness = engine.TensorSubtract(fitness, engine.TensorMultiplyScalar(sparsity, NumOps.FromDouble(_sparsityWeight)));
        return engine.TensorAdd(fitness, engine.TensorMultiplyScalar(diversity, NumOps.FromDouble(_diversityWeight))); // Reward diversity
    }

    /// <summary>
    /// The model's first output for each individual, one prediction call per individual.
    /// </summary>
    private Tensor<T> Predict(List<Vector<T>> population)
    {
        var predictions = population
            .Select(individual => _predictFunction(CreateSingleRowMatrix(individual)).GetSubVector(0, 1))
            .ToList();
        return AiDotNetEngine.Current.TensorSliceAxis(ExplainerTensorOps<T>.Stack(predictions), 1, 0);
    }

    /// <summary>
    /// Computes validity loss (how far from target prediction).
    /// </summary>
    private Tensor<T> ComputeValidityLoss(Tensor<T> predicted, T target)
    {
        // Distance from target threshold, zero once the prediction is on the target's side
        var engine = AiDotNetEngine.Current;
        T threshold = NumOps.FromDouble(_targetThreshold);
        var gap = NumOps.GreaterThanOrEquals(target, threshold)
            ? engine.TensorSubtract(engine.TensorFullLike(predicted, threshold), predicted)
            : engine.TensorSubtractScalar(predicted, threshold);
        return engine.TensorMax(gap, NumOps.Zero);
    }

    /// <summary>
    /// Whether each prediction reaches the target's side of the threshold.
    /// </summary>
    private bool[] IsValid(Tensor<T> predicted, T target)
    {
        var engine = AiDotNetEngine.Current;
        T threshold = NumOps.FromDouble(_targetThreshold);
        var valid = NumOps.GreaterThanOrEquals(target, threshold)
            ? engine.TensorGreaterEqual(predicted, engine.TensorFullLike(predicted, threshold))
            : engine.TensorLessThan(predicted, threshold);
        return valid.Cast<int>().ToArray().Select(flag => flag != 0).ToArray();
    }

    /// <summary>
    /// Computes normalized distance from original.
    /// </summary>
    private Tensor<T> PairwiseDistances(List<Vector<T>> first, List<Vector<T>> second)
    {
        // sqrt(sum over changed features (|diff| > 1e-6) of (diff / (range + 1e-6))^2), as a [first, second] tensor
        var engine = AiDotNetEngine.Current;
        int rows = first.Count;
        int columns = second.Count;
        var shape = new[] { rows, columns, _numFeatures };
        var a = engine.TensorBroadcastTo(ExplainerTensorOps<T>.Stack(first).Reshape(new[] { rows, 1, _numFeatures }), shape);
        var b = engine.TensorBroadcastTo(ExplainerTensorOps<T>.Stack(second).Reshape(new[] { 1, columns, _numFeatures }), shape);
        var diff = engine.TensorSubtract(a, b);

        var scale = engine.TensorAddScalar(NormalizationRanges().AsTensor(), NumOps.FromDouble(1e-6));
        var normalized = engine.TensorDivide(diff, engine.TensorBroadcastTo(scale.Reshape(new[] { 1, 1, _numFeatures }), shape));
        var changed = engine.TensorGreaterThan(engine.TensorAbs(diff), NumOps.FromDouble(1e-6));
        var sumSq = engine.ReduceSum(engine.TensorMultiply(changed, engine.TensorSquare(normalized)), new[] { 2 }, false);
        return engine.TensorSqrt(sumSq);
    }

    /// <summary>
    /// The normalized distance between two instances.
    /// </summary>
    private T ComputeNormalizedDistance(Vector<T> original, Vector<T> counterfactual)
    {
        return PairwiseDistances(new List<Vector<T>> { original }, new List<Vector<T>> { counterfactual }).GetFlat(0);
    }

    /// <summary>
    /// The per-feature normalization ranges (1 for features without a known range).
    /// </summary>
    private Vector<T> NormalizationRanges()
    {
        var engine = AiDotNetEngine.Current;
        if (_featureRanges == null || _featureRanges.Length == 0)
        {
            return engine.Fill(_numFeatures, NumOps.One);
        }

        var known = new Vector<T>(_featureRanges);
        if (known.Length >= _numFeatures)
        {
            return known.GetSubVector(0, _numFeatures);
        }

        return engine.TensorConcatenate(
            new[] { known.AsTensor(), engine.Fill(_numFeatures - known.Length, NumOps.One).AsTensor() }, 0).ToVector();
    }

    /// <summary>
    /// Computes sparsity (fraction of features changed).
    /// </summary>
    private Tensor<T> ComputeSparsity(Vector<T> original, List<Vector<T>> population)
    {
        var engine = AiDotNetEngine.Current;
        var mutable = Enumerable.Range(0, _numFeatures).Select(IsMutable).ToArray();
        int mutableCount = mutable.Count(isMutable => isMutable);
        if (mutableCount == 0)
        {
            return new Tensor<T>(new[] { population.Count });
        }

        // Fraction of the mutable features changed by more than 1e-6
        var individuals = ExplainerTensorOps<T>.Stack(population);
        var originals = engine.TensorBroadcastTo(ExplainerTensorOps<T>.RowTensor(original.GetSubVector(0, _numFeatures)), individuals.Shape.ToArray());
        var changed = engine.TensorGreaterThan(engine.TensorAbs(engine.TensorSubtract(originals, individuals)), NumOps.FromDouble(1e-6));
        var mutableMask = engine.TensorBroadcastTo(
            new Tensor<int>(mutable.Select(isMutable => isMutable ? 1 : 0).ToArray(), new[] { 1, _numFeatures }).Cast<T>(), individuals.Shape.ToArray());
        var changedCount = engine.ReduceSum(engine.TensorMultiply(changed, mutableMask), new[] { 1 }, false);
        return engine.TensorDivideScalar(changedCount, NumOps.FromDouble(mutableCount));
    }

    /// <summary>
    /// Computes diversity of a set of counterfactuals.
    /// </summary>
    private bool HasSetDiversityAbove(List<Vector<T>> counterfactuals, double threshold)
    {
        int n = counterfactuals.Count;
        if (n <= 1) return false;

        // Mean distance over the pairs i < j (the distance matrix is symmetric with a zero diagonal)
        T totalDist = NumOps.Divide(AiDotNetEngine.Current.TensorSum(PairwiseDistances(counterfactuals, counterfactuals)), NumOps.FromDouble(2));
        T meanDist = NumOps.Divide(totalDist, NumOps.FromDouble(n * (n - 1) / 2));
        return NumOps.GreaterThan(meanDist, NumOps.FromDouble(threshold));
    }

    /// <summary>
    /// Tournament selection for genetic algorithm.
    /// </summary>
    private List<Vector<T>> TournamentSelection(List<Vector<T>> population, int[] fitness, Random rand)
    {
        var selected = new List<Vector<T>>();
        int tournamentSize = 3;

        for (int i = 0; i < _populationSize; i++)
        {
            int best = rand.Next(population.Count);
            int bestFitness = fitness[best];

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
        var engine = AiDotNetEngine.Current;

        // Immutable features keep the original value; mutable ones swap parents at the crossover point
        var immutable = Enumerable.Range(0, _numFeatures).Select(f => !IsMutable(f)).ToArray();
        var beforePoint = Enumerable.Range(0, _numFeatures).Select(f => f < crossoverPoint).ToArray();
        var first = parent1.GetSubVector(0, _numFeatures).AsTensor();
        var second = parent2.GetSubVector(0, _numFeatures).AsTensor();
        var originalValues = original.GetSubVector(0, _numFeatures).AsTensor();

        var child1 = engine.TensorWhere(immutable, originalValues, engine.TensorWhere(beforePoint, first, second));
        var child2 = engine.TensorWhere(immutable, originalValues, engine.TensorWhere(beforePoint, second, first));
        return (child1.ToVector(), child2.ToVector());
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
    private List<Vector<T>> CombineWithElitism(List<Vector<T>> parents, List<Vector<T>> offspring, int[] parentFitness, Vector<T> original, T targetValue)
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
        var offspringFitness = ExplainerTensorOps<T>.DenseRanks(EvaluatePopulation(offspring, original, targetValue));
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
    private int CountValidCounterfactuals(List<Vector<T>> population, T targetValue)
    {
        if (population.Count == 0) return 0;
        return IsValid(Predict(population), targetValue).Count(valid => valid);
    }

    /// <summary>
    /// Selects top counterfactuals based on fitness.
    /// </summary>
    private List<Vector<T>> SelectTopCounterfactuals(List<Vector<T>> population, int[] fitness, Vector<T> original, T targetValue)
    {
        // Filter to valid counterfactuals
        var valid = population.Count > 0 ? IsValid(Predict(population), targetValue) : Array.Empty<bool>();
        var validIndices = Enumerable.Range(0, population.Count).Where(i => valid[i]).ToList();

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
    private List<Vector<T>> SelectDiverseCounterfactuals(List<Vector<T>> population, int[] fitness, Vector<T> original, T targetValue)
    {
        // Get valid counterfactuals sorted by fitness
        var valid = population.Count > 0 ? IsValid(Predict(population), targetValue) : Array.Empty<bool>();
        var candidates = Enumerable.Range(0, population.Count).Where(i => valid[i]).ToList();

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
        var candidateList = candidates.OrderByDescending(i => fitness[i]).Select(i => population[i]).ToList();

        // Add best one first
        var selected = new List<Vector<T>> { candidateList[0] };
        var remaining = candidateList.Skip(1).ToList();
        if (selected.Count >= _numCounterfactuals || remaining.Count == 0)
        {
            return selected;
        }

        // Greedily add the remaining candidate farthest from its nearest selected one
        var engine = AiDotNetEngine.Current;
        var distances = PairwiseDistances(remaining, candidateList);
        // Nearest-selected distance per candidate; a NaN distance is skipped (MaxValue until a number arrives)
        var minDistances = NanSkippingMin(
            engine.TensorFullLike(engine.TensorSliceAxis(distances, 1, 0), NumOps.MaxValue), engine.TensorSliceAxis(distances, 1, 0));
        var taken = new bool[remaining.Count];
        T negativeOne = NumOps.Negate(NumOps.One);
        var unavailable = engine.TensorFullLike(minDistances, negativeOne);
        while (selected.Count < _numCounterfactuals && selected.Count < candidateList.Count)
        {
            // The farthest untaken candidate (scan seeded with -1); with none above -1, the first untaken one
            int bestIdx = ExplainerTensorOps<T>.FirstArgMaxAbove(
                engine.TensorWhere(taken, unavailable, minDistances).ToVector(), negativeOne);
            if (bestIdx < 0)
            {
                bestIdx = Array.IndexOf(taken, false);
            }
            selected.Add(remaining[bestIdx]);
            taken[bestIdx] = true;
            minDistances = NanSkippingMin(minDistances, engine.TensorSliceAxis(distances, 1, bestIdx + 1));
        }

        return selected;
    }

    /// <summary>
    /// Element-wise min(running, next) that keeps the running value where next is NaN (running is never NaN).
    /// </summary>
    private static Tensor<T> NanSkippingMin(Tensor<T> running, Tensor<T> next)
    {
        var engine = AiDotNetEngine.Current;
        return engine.TensorWhere(engine.TensorIsNan(next), running, engine.TensorMin(running, next));
    }

    /// <summary>
    /// Creates a single counterfactual result.
    /// </summary>
    private SingleCounterfactual<T> CreateSingleCounterfactual(Vector<T> original, Vector<T> counterfactual, T prediction)
    {
        var engine = AiDotNetEngine.Current;
        var originalValues = original.GetSubVector(0, _numFeatures);
        var counterfactualValues = counterfactual.GetSubVector(0, _numFeatures);
        var diff = engine.Subtract(originalValues, counterfactualValues);
        var changedFeatures = engine.TensorGreaterThan(engine.Abs(diff).AsTensor(), NumOps.FromDouble(1e-6))
            .ToVector().NonZeroIndices().ToList();

        var changes = changedFeatures
            .Zip(originalValues.GetElements(changedFeatures), (feature, value) => (feature, value))
            .Zip(counterfactualValues.GetElements(changedFeatures), (entry, value) => new FeatureChange<T>(
                featureIndex: entry.feature,
                featureName: _featureNames?[entry.feature] ?? $"Feature_{entry.feature}",
                originalValue: entry.value,
                counterfactualValue: value))
            .ToList();

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

    private T[]? ComputeFeatureRanges()
    {
        // Without bounds there is no known range: the null ranges fall back to 1 (GetFeatureRange /
        // NormalizationRanges). An all-zero array here would zero every perturbation and freeze the search.
        if (_featureMins == null || _featureMaxs == null)
        {
            return null;
        }

        var engine = AiDotNetEngine.Current;
        var maxs = new Vector<T>(_featureMaxs).GetSubVector(0, _numFeatures);
        var mins = new Vector<T>(_featureMins).GetSubVector(0, _numFeatures);
        return engine.Subtract(maxs, mins).ToArray();
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
    public T Distance { get; }

    /// <summary>Number of features changed.</summary>
    public int Sparsity => Changes.Count;

    /// <summary>Initializes a new single counterfactual.</summary>
    public SingleCounterfactual(Vector<T> counterfactualInstance, T prediction, List<FeatureChange<T>> changes, T distance)
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
    public T AverageDistance => Counterfactuals.Count > 0
        ? AiDotNetEngine.Current.Mean(new Vector<T>(Counterfactuals.Select(cf => cf.Distance))) : NumOps.Zero;

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
