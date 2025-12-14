using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.Templates;

namespace AiDotNet.PromptEngineering.Optimization;

/// <summary>
/// Optimizer that uses simulated annealing to escape local optima.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring.</typeparam>
/// <remarks>
/// <para>
/// Simulated annealing occasionally accepts worse solutions early in the search
/// (when "temperature" is high), allowing exploration of the solution space.
/// As temperature decreases, it becomes more greedy.
/// </para>
/// <para><b>For Beginners:</b> Like cooling metal - starts flexible, becomes rigid.
///
/// Example:
/// <code>
/// var optimizer = new SimulatedAnnealingOptimizer&lt;double&gt;(
///     initialTemperature: 1.0,
///     coolingRate: 0.95
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
/// - High temperature: Accept many variations, even worse ones
/// - Cooling down: Gradually become more selective
/// - Low temperature: Only accept improvements (like greedy search)
///
/// Benefits:
/// - Escapes local optima by exploring broadly
/// - Proven effective for combinatorial optimization
/// - Balances exploration and exploitation
/// </para>
/// </remarks>
public class SimulatedAnnealingOptimizer<T> : PromptOptimizerBase<T>
{
    private readonly double _initialTemperature;
    private readonly double _coolingRate;
    private readonly double _minTemperature;
    private readonly Random _random;

    private readonly List<string> _modifications;

    /// <summary>
    /// Initializes a new instance of the SimulatedAnnealingOptimizer class.
    /// </summary>
    /// <param name="initialTemperature">Starting temperature (higher = more exploration).</param>
    /// <param name="coolingRate">Rate of cooling (0-1, higher = slower cooling).</param>
    /// <param name="minTemperature">Minimum temperature before stopping.</param>
    /// <param name="seed">Random seed for reproducibility (null for random).</param>
    public SimulatedAnnealingOptimizer(
        double initialTemperature = 1.0,
        double coolingRate = 0.95,
        double minTemperature = 0.01,
        int? seed = null)
    {
        _initialTemperature = Math.Max(0.01, initialTemperature);
        _coolingRate = Math.Max(0.5, Math.Min(0.999, coolingRate));
        _minTemperature = Math.Max(0.001, minTemperature);
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        _modifications = new List<string>
        {
            "prefix_add",
            "suffix_add",
            "word_swap",
            "word_delete",
            "word_insert",
            "case_change",
            "punctuation_modify"
        };
    }

    /// <summary>
    /// Optimizes using simulated annealing.
    /// </summary>
    protected override IPromptTemplate OptimizeCore(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations)
    {
        string currentPrompt = initialPrompt;
        T currentScore = evaluationFunction(initialPrompt);

        string bestPrompt = currentPrompt;
        T bestScore = currentScore;

        double temperature = _initialTemperature;

        RecordIteration(0, currentPrompt, currentScore);

        for (int iteration = 1; iteration < maxIterations && temperature > _minTemperature; iteration++)
        {
            // Generate neighbor
            string candidatePrompt = GenerateNeighbor(currentPrompt);
            T candidateScore = evaluationFunction(candidatePrompt);

            // Calculate acceptance probability
            double scoreDiff = ConvertToDouble(candidateScore) - ConvertToDouble(currentScore);
            bool accept = false;

            if (scoreDiff > 0)
            {
                // Better solution - always accept
                accept = true;
            }
            else
            {
                // Worse solution - accept with probability based on temperature
                double acceptanceProbability = Math.Exp(scoreDiff / temperature);
                accept = _random.NextDouble() < acceptanceProbability;
            }

            if (accept)
            {
                currentPrompt = candidatePrompt;
                currentScore = candidateScore;
            }

            // Track best overall
            if (NumOps.GreaterThan(candidateScore, bestScore))
            {
                bestScore = candidateScore;
                bestPrompt = candidatePrompt;
            }

            RecordIteration(iteration, currentPrompt, currentScore);

            // Cool down
            temperature *= _coolingRate;
        }

        return new SimplePromptTemplate(bestPrompt);
    }

    /// <summary>
    /// Optimizes using simulated annealing asynchronously.
    /// </summary>
    protected override async Task<IPromptTemplate> OptimizeCoreAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations,
        CancellationToken cancellationToken)
    {
        string currentPrompt = initialPrompt;
        T currentScore = await evaluationFunction(initialPrompt).ConfigureAwait(false);

        string bestPrompt = currentPrompt;
        T bestScore = currentScore;

        double temperature = _initialTemperature;

        RecordIteration(0, currentPrompt, currentScore);

        for (int iteration = 1; iteration < maxIterations && temperature > _minTemperature; iteration++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Generate neighbor
            string candidatePrompt = GenerateNeighbor(currentPrompt);
            T candidateScore = await evaluationFunction(candidatePrompt).ConfigureAwait(false);

            // Calculate acceptance probability
            double scoreDiff = ConvertToDouble(candidateScore) - ConvertToDouble(currentScore);
            bool accept = false;

            if (scoreDiff > 0)
            {
                accept = true;
            }
            else
            {
                double acceptanceProbability = Math.Exp(scoreDiff / temperature);
                accept = _random.NextDouble() < acceptanceProbability;
            }

            if (accept)
            {
                currentPrompt = candidatePrompt;
                currentScore = candidateScore;
            }

            if (NumOps.GreaterThan(candidateScore, bestScore))
            {
                bestScore = candidateScore;
                bestPrompt = candidatePrompt;
            }

            RecordIteration(iteration, currentPrompt, currentScore);

            temperature *= _coolingRate;
        }

        return new SimplePromptTemplate(bestPrompt);
    }

    private string GenerateNeighbor(string prompt)
    {
        var modification = _modifications[_random.Next(_modifications.Count)];

        return modification switch
        {
            "prefix_add" => AddPrefix(prompt),
            "suffix_add" => AddSuffix(prompt),
            "word_swap" => SwapWords(prompt),
            "word_delete" => DeleteWord(prompt),
            "word_insert" => InsertWord(prompt),
            "case_change" => ChangeCase(prompt),
            "punctuation_modify" => ModifyPunctuation(prompt),
            _ => prompt
        };
    }

    private string AddPrefix(string prompt)
    {
        var prefixes = new[]
        {
            "Please ", "Carefully ", "Think step-by-step and ",
            "Analyze and ", "Consider thoroughly and "
        };

        var prefix = prefixes[_random.Next(prefixes.Length)];
        if (!prompt.StartsWith(prefix))
        {
            return prefix + prompt;
        }
        return prompt;
    }

    private string AddSuffix(string prompt)
    {
        var suffixes = new[]
        {
            "\n\nProvide your answer clearly.",
            "\n\nExplain your reasoning.",
            "\n\nBe concise and specific."
        };

        var suffix = suffixes[_random.Next(suffixes.Length)];
        if (!prompt.EndsWith(suffix.Trim()))
        {
            return prompt + suffix;
        }
        return prompt;
    }

    private string SwapWords(string prompt)
    {
        var words = prompt.Split(' ').ToList();
        if (words.Count < 4) return prompt;

        int i = _random.Next(1, words.Count - 1);
        int j = _random.Next(1, words.Count - 1);
        if (i != j)
        {
            (words[i], words[j]) = (words[j], words[i]);
        }

        return string.Join(" ", words);
    }

    private string DeleteWord(string prompt)
    {
        var words = prompt.Split(' ').ToList();
        if (words.Count < 5) return prompt;

        // Delete a non-essential word
        var deleteIndex = _random.Next(1, words.Count - 1);
        words.RemoveAt(deleteIndex);

        return string.Join(" ", words);
    }

    private string InsertWord(string prompt)
    {
        var words = prompt.Split(' ').ToList();
        var insertions = new[] { "carefully", "thoroughly", "clearly", "precisely", "accurately" };

        var insertion = insertions[_random.Next(insertions.Length)];
        var position = _random.Next(1, words.Count);
        words.Insert(position, insertion);

        return string.Join(" ", words);
    }

    private string ChangeCase(string prompt)
    {
        var words = prompt.Split(' ').ToList();
        if (words.Count < 2) return prompt;

        var index = _random.Next(words.Count);
        words[index] = char.IsUpper(words[index][0])
            ? words[index].ToLowerInvariant()
            : char.ToUpperInvariant(words[index][0]) + words[index].Substring(1);

        return string.Join(" ", words);
    }

    private string ModifyPunctuation(string prompt)
    {
        if (prompt.EndsWith("."))
        {
            return prompt.TrimEnd('.') + "!";
        }
        else if (prompt.EndsWith("!"))
        {
            return prompt.TrimEnd('!') + ".";
        }
        else
        {
            return prompt + ".";
        }
    }

    private double ConvertToDouble(T value)
    {
        return Convert.ToDouble(value);
    }
}
