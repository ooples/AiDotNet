using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.Templates;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PromptEngineering.Optimization;

/// <summary>
/// Optimizer that uses genetic algorithms to evolve better prompts.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring.</typeparam>
/// <remarks>
/// <para>
/// This optimizer mimics natural evolution: prompts are mutated, crossed over,
/// and selected based on fitness (performance scores).
/// </para>
/// <para><b>For Beginners:</b> Evolves prompts like nature evolves species.
///
/// Example:
/// <code>
/// var optimizer = new GeneticOptimizer&lt;double&gt;(
///     populationSize: 20,
///     mutationRate: 0.1
/// );
///
/// var optimized = optimizer.Optimize(
///     initialPrompt: "Classify sentiment:",
///     evaluationFunction: prompt => EvaluateAccuracy(prompt),
///     maxIterations: 100
/// );
/// </code>
///
/// How it works:
/// - Start with variations of the initial prompt (population)
/// - Evaluate how well each performs (fitness)
/// - Select the best performers
/// - Combine them (crossover) and introduce random changes (mutation)
/// - Repeat for many generations
/// </para>
/// </remarks>
public class GeneticOptimizer<T> : PromptOptimizerBase<T>
{
    private readonly int _populationSize;
    private readonly double _mutationRate;
    private readonly double _crossoverRate;
    private readonly int _eliteCount;
    private readonly Random _random;

    private readonly List<string> _prefixes;
    private readonly List<string> _suffixes;
    private readonly List<string> _connectors;

    /// <summary>
    /// Initializes a new instance of the GeneticOptimizer class.
    /// </summary>
    /// <param name="populationSize">Number of prompts in each generation.</param>
    /// <param name="mutationRate">Probability of mutation (0.0 to 1.0).</param>
    /// <param name="crossoverRate">Probability of crossover (0.0 to 1.0).</param>
    /// <param name="eliteCount">Number of top performers to keep unchanged.</param>
    /// <param name="seed">Random seed for reproducibility (null for random).</param>
    public GeneticOptimizer(
        int populationSize = 20,
        double mutationRate = 0.1,
        double crossoverRate = 0.7,
        int eliteCount = 2,
        int? seed = null)
    {
        _populationSize = Math.Max(4, populationSize);
        _mutationRate = Math.Max(0.0, Math.Min(1.0, mutationRate));
        _crossoverRate = Math.Max(0.0, Math.Min(1.0, crossoverRate));
        _eliteCount = Math.Min(eliteCount, _populationSize / 2);
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        // Default genetic building blocks
        _prefixes = new List<string>
        {
            "",
            "Please ",
            "Carefully ",
            "Think step-by-step and ",
            "Analyze and ",
            "Consider all aspects and ",
            "Using your expertise, ",
            "Thoughtfully "
        };

        _suffixes = new List<string>
        {
            "",
            "\n\nProvide your answer clearly.",
            "\n\nExplain your reasoning.",
            "\n\nBe concise and specific.",
            "\n\nAnswer in detail.",
            "\n\nShow your work.",
            "\n\nGive a comprehensive response."
        };

        _connectors = new List<string>
        {
            " ",
            ". Then ",
            ", and then ",
            ". After that, ",
            ". Next, "
        };
    }

    /// <summary>
    /// Adds custom genetic building blocks for mutation.
    /// </summary>
    /// <param name="prefixes">Prefix variations to add.</param>
    /// <param name="suffixes">Suffix variations to add.</param>
    /// <param name="connectors">Connector variations to add.</param>
    public void AddBuildingBlocks(
        IEnumerable<string>? prefixes = null,
        IEnumerable<string>? suffixes = null,
        IEnumerable<string>? connectors = null)
    {
        if (prefixes is not null) _prefixes.AddRange(prefixes);
        if (suffixes is not null) _suffixes.AddRange(suffixes);
        if (connectors is not null) _connectors.AddRange(connectors);
    }

    /// <summary>
    /// Optimizes using genetic algorithm.
    /// </summary>
    protected override IPromptTemplate OptimizeCore(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations)
    {
        // Initialize population
        var population = InitializePopulation(initialPrompt);
        var fitness = EvaluatePopulation(population, evaluationFunction);

        // Track best
        int bestIndex = FindBestIndex(fitness);
        string bestPrompt = population[bestIndex];
        T bestScore = fitness[bestIndex];

        RecordIteration(0, bestPrompt, bestScore);

        int generations = maxIterations / _populationSize;
        int iteration = 1;

        for (int gen = 0; gen < generations && iteration < maxIterations; gen++)
        {
            // Selection, crossover, mutation
            var newPopulation = Evolve(population, fitness);
            population = newPopulation;
            fitness = EvaluatePopulation(population, evaluationFunction);

            // Track best of this generation
            bestIndex = FindBestIndex(fitness);
            if (NumOps.GreaterThan(fitness[bestIndex], bestScore))
            {
                bestScore = fitness[bestIndex];
                bestPrompt = population[bestIndex];
            }

            RecordIteration(iteration, population[bestIndex], fitness[bestIndex]);
            iteration++;
        }

        return new SimplePromptTemplate(bestPrompt);
    }

    /// <summary>
    /// Optimizes using genetic algorithm asynchronously.
    /// </summary>
    protected override async Task<IPromptTemplate> OptimizeCoreAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations,
        CancellationToken cancellationToken)
    {
        // Initialize population
        var population = InitializePopulation(initialPrompt);
        var fitness = await EvaluatePopulationAsync(population, evaluationFunction, cancellationToken)
            .ConfigureAwait(false);

        // Track best
        int bestIndex = FindBestIndex(fitness);
        string bestPrompt = population[bestIndex];
        T bestScore = fitness[bestIndex];

        RecordIteration(0, bestPrompt, bestScore);

        int generations = maxIterations / _populationSize;
        int iteration = 1;

        for (int gen = 0; gen < generations && iteration < maxIterations; gen++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Selection, crossover, mutation
            var newPopulation = Evolve(population, fitness);
            population = newPopulation;
            fitness = await EvaluatePopulationAsync(population, evaluationFunction, cancellationToken)
                .ConfigureAwait(false);

            // Track best of this generation
            bestIndex = FindBestIndex(fitness);
            if (NumOps.GreaterThan(fitness[bestIndex], bestScore))
            {
                bestScore = fitness[bestIndex];
                bestPrompt = population[bestIndex];
            }

            RecordIteration(iteration, population[bestIndex], fitness[bestIndex]);
            iteration++;
        }

        return new SimplePromptTemplate(bestPrompt);
    }

    private List<string> InitializePopulation(string initialPrompt)
    {
        var population = new List<string> { initialPrompt };

        while (population.Count < _populationSize)
        {
            population.Add(Mutate(initialPrompt));
        }

        return population;
    }

    private T[] EvaluatePopulation(List<string> population, Func<string, T> evaluationFunction)
    {
        var fitness = new T[population.Count];
        for (int i = 0; i < population.Count; i++)
        {
            fitness[i] = evaluationFunction(population[i]);
        }
        return fitness;
    }

    private async Task<T[]> EvaluatePopulationAsync(
        List<string> population,
        Func<string, Task<T>> evaluationFunction,
        CancellationToken cancellationToken)
    {
        var tasks = population.Select(p => evaluationFunction(p)).ToArray();
        return await Task.WhenAll(tasks).ConfigureAwait(false);
    }

    private int FindBestIndex(T[] fitness)
    {
        int bestIndex = 0;
        for (int i = 1; i < fitness.Length; i++)
        {
            if (NumOps.GreaterThan(fitness[i], fitness[bestIndex]))
            {
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private List<string> Evolve(List<string> population, T[] fitness)
    {
        var newPopulation = new List<string>();

        // Elitism: keep top performers
        var ranked = population
            .Select((p, i) => (Prompt: p, Fitness: fitness[i]))
            .OrderByDescending(x => x.Fitness, Comparer<T>.Create((a, b) =>
                NumOps.GreaterThan(a, b) ? 1 : (NumOps.LessThan(a, b) ? -1 : 0)))
            .ToList();

        for (int i = 0; i < _eliteCount && i < ranked.Count; i++)
        {
            newPopulation.Add(ranked[i].Prompt);
        }

        // Fill rest with offspring
        while (newPopulation.Count < _populationSize)
        {
            var parent1 = TournamentSelect(population, fitness);
            var parent2 = TournamentSelect(population, fitness);

            string offspring;
            if (_random.NextDouble() < _crossoverRate)
            {
                offspring = Crossover(parent1, parent2);
            }
            else
            {
                offspring = _random.NextDouble() < 0.5 ? parent1 : parent2;
            }

            if (_random.NextDouble() < _mutationRate)
            {
                offspring = Mutate(offspring);
            }

            newPopulation.Add(offspring);
        }

        return newPopulation;
    }

    private string TournamentSelect(List<string> population, T[] fitness, int tournamentSize = 3)
    {
        // Select by index to avoid IndexOf bug when duplicates exist in population
        int bestIdx = _random.Next(population.Count);
        string best = population[bestIdx];
        T bestFitness = fitness[bestIdx];

        for (int i = 1; i < tournamentSize; i++)
        {
            int idx = _random.Next(population.Count);
            if (NumOps.GreaterThan(fitness[idx], bestFitness))
            {
                best = population[idx];
                bestFitness = fitness[idx];
            }
        }

        return best;
    }

    private string Crossover(string parent1, string parent2)
    {
        var words1 = parent1.Split(' ');
        var words2 = parent2.Split(' ');

        var midpoint1 = words1.Length / 2;
        var midpoint2 = words2.Length / 2;

        var offspring = string.Join(" ", words1.Take(midpoint1))
            + " " + string.Join(" ", words2.Skip(midpoint2));

        return offspring.Trim();
    }

    private string Mutate(string prompt)
    {
        var mutationType = _random.Next(4);

        return mutationType switch
        {
            0 => _prefixes[_random.Next(_prefixes.Count)] + prompt,
            1 => prompt + _suffixes[_random.Next(_suffixes.Count)],
            2 => InsertConnector(prompt),
            3 => ShuffleWords(prompt),
            _ => prompt
        };
    }

    private string InsertConnector(string prompt)
    {
        var sentences = prompt.Split(new[] { ". " }, StringSplitOptions.RemoveEmptyEntries);
        if (sentences.Length < 2) return prompt;

        var idx = _random.Next(sentences.Length - 1);
        sentences[idx] = sentences[idx] + _connectors[_random.Next(_connectors.Count)];

        return string.Join(". ", sentences);
    }

    private string ShuffleWords(string prompt)
    {
        var words = prompt.Split(' ').ToList();
        if (words.Count < 4) return prompt;

        // Only shuffle middle portion to preserve structure
        var start = 1;
        var end = words.Count - 1;

        if (end - start > 2)
        {
            var i = _random.Next(start, end);
            var j = _random.Next(start, end);
            (words[i], words[j]) = (words[j], words[i]);
        }

        return string.Join(" ", words);
    }
}
