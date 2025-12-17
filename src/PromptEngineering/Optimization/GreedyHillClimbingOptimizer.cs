using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.Templates;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.PromptEngineering.Optimization;

/// <summary>
/// Simple greedy optimizer that always moves toward better solutions.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring.</typeparam>
/// <remarks>
/// <para>
/// Hill climbing is a simple optimization strategy that always accepts improvements
/// and never accepts worse solutions. Fast but can get stuck in local optima.
/// </para>
/// <para><b>For Beginners:</b> Always takes the best step available.
///
/// Example:
/// <code>
/// var optimizer = new GreedyHillClimbingOptimizer&lt;double&gt;();
///
/// var optimized = optimizer.Optimize(
///     initialPrompt: "Classify sentiment:",
///     evaluationFunction: prompt => EvaluateAccuracy(prompt),
///     maxIterations: 50
/// );
/// </code>
///
/// How it works:
/// - Try a variation of current prompt
/// - If it's better, use it as the new current
/// - If it's worse, try another variation
/// - Repeat until no improvements found
///
/// Benefits:
/// - Simple and fast
/// - Guaranteed to improve (or stay same)
/// - Good for fine-tuning near a good solution
///
/// Limitations:
/// - Can get stuck at local optima
/// - May miss better solutions further away
/// </para>
/// </remarks>
public class GreedyHillClimbingOptimizer<T> : PromptOptimizerBase<T>
{
    private readonly int _maxNoImprovementIterations;
    private readonly Random _random;

    private readonly List<Func<string, string>> _mutations;

    /// <summary>
    /// Initializes a new instance of the GreedyHillClimbingOptimizer class.
    /// </summary>
    /// <param name="maxNoImprovementIterations">Stop after this many iterations without improvement.</param>
    /// <param name="seed">Random seed for reproducibility (null for random).</param>
    public GreedyHillClimbingOptimizer(
        int maxNoImprovementIterations = 10,
        int? seed = null)
    {
        _maxNoImprovementIterations = Math.Max(1, maxNoImprovementIterations);
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

        // Define mutation operations
        _mutations = new List<Func<string, string>>
        {
            AddPrefix,
            AddSuffix,
            RemoveFiller,
            AddClarity,
            ReorderWords,
            AddSpecificity
        };
    }

    /// <summary>
    /// Optimizes using greedy hill climbing.
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

        int noImprovementCount = 0;
        RecordIteration(0, currentPrompt, currentScore);

        for (int iteration = 1; iteration < maxIterations && noImprovementCount < _maxNoImprovementIterations; iteration++)
        {
            // Generate a neighbor
            var mutation = _mutations[_random.Next(_mutations.Count)];
            string candidatePrompt = mutation(currentPrompt);

            // Skip if same as current
            if (candidatePrompt == currentPrompt)
            {
                continue;
            }

            T candidateScore = evaluationFunction(candidatePrompt);
            RecordIteration(iteration, candidatePrompt, candidateScore);

            // Greedy: only accept if better
            if (NumOps.GreaterThan(candidateScore, currentScore))
            {
                currentPrompt = candidatePrompt;
                currentScore = candidateScore;
                noImprovementCount = 0;

                // Update best
                if (NumOps.GreaterThan(currentScore, bestScore))
                {
                    bestScore = currentScore;
                    bestPrompt = currentPrompt;
                }
            }
            else
            {
                noImprovementCount++;
            }
        }

        return new SimplePromptTemplate(bestPrompt);
    }

    /// <summary>
    /// Optimizes using greedy hill climbing asynchronously.
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

        int noImprovementCount = 0;
        RecordIteration(0, currentPrompt, currentScore);

        for (int iteration = 1; iteration < maxIterations && noImprovementCount < _maxNoImprovementIterations; iteration++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var mutation = _mutations[_random.Next(_mutations.Count)];
            string candidatePrompt = mutation(currentPrompt);

            if (candidatePrompt == currentPrompt)
            {
                continue;
            }

            T candidateScore = await evaluationFunction(candidatePrompt).ConfigureAwait(false);
            RecordIteration(iteration, candidatePrompt, candidateScore);

            if (NumOps.GreaterThan(candidateScore, currentScore))
            {
                currentPrompt = candidatePrompt;
                currentScore = candidateScore;
                noImprovementCount = 0;

                if (NumOps.GreaterThan(currentScore, bestScore))
                {
                    bestScore = currentScore;
                    bestPrompt = currentPrompt;
                }
            }
            else
            {
                noImprovementCount++;
            }
        }

        return new SimplePromptTemplate(bestPrompt);
    }

    private string AddPrefix(string prompt)
    {
        var prefixes = new[]
        {
            "Please ", "Carefully ", "Think step-by-step and ",
            "Analyze and ", "Consider and ", "Thoroughly "
        };

        var prefix = prefixes[_random.Next(prefixes.Length)];
        return prompt.StartsWith(prefix) ? prompt : prefix + prompt;
    }

    private string AddSuffix(string prompt)
    {
        var suffixes = new[]
        {
            "\n\nProvide your answer clearly.",
            "\n\nExplain your reasoning step by step.",
            "\n\nBe concise and accurate.",
            "\n\nFormat your response appropriately."
        };

        var suffix = suffixes[_random.Next(suffixes.Length)];
        return prompt.Contains(suffix.Trim()) ? prompt : prompt + suffix;
    }

    private string RemoveFiller(string prompt)
    {
        var fillers = new[] { " very ", " really ", " quite ", " just ", " simply ", " basically " };
        var result = prompt;

        foreach (var filler in fillers)
        {
            if (result.Contains(filler) && _random.NextDouble() < 0.5)
            {
                result = result.Replace(filler, " ");
            }
        }

        return result;
    }

    private string AddClarity(string prompt)
    {
        var clarifiers = new[]
        {
            (":", ": specifically,"),
            (".", ". Make sure to"),
            ("?", "? Please explain"),
            (",", ", clearly,")
        };

        var clarifier = clarifiers[_random.Next(clarifiers.Length)];

        var parts = prompt.Split(new[] { clarifier.Item1 }, StringSplitOptions.None);
        if (parts.Length > 1 && _random.NextDouble() < 0.5)
        {
            var index = _random.Next(parts.Length - 1);
            parts[index] = parts[index] + clarifier.Item2.TrimStart(clarifier.Item1[0]);
            return string.Join(clarifier.Item1, parts);
        }

        return prompt;
    }

    private string ReorderWords(string prompt)
    {
        var words = prompt.Split(' ').ToList();
        if (words.Count < 5) return prompt;

        // Swap two adjacent words (excluding first and last)
        var index = _random.Next(1, words.Count - 2);
        (words[index], words[index + 1]) = (words[index + 1], words[index]);

        return string.Join(" ", words);
    }

    private string AddSpecificity(string prompt)
    {
        var specifics = new[]
        {
            " (be specific) ",
            " (provide details) ",
            " (include examples) ",
            " (be thorough) "
        };

        // Insert at a sentence break
        var sentences = prompt.Split(new[] { ". " }, StringSplitOptions.None);
        if (sentences.Length > 1)
        {
            var index = _random.Next(sentences.Length - 1);
            sentences[index] = sentences[index] + specifics[_random.Next(specifics.Length)].Trim();
            return string.Join(". ", sentences);
        }

        return prompt + specifics[_random.Next(specifics.Length)];
    }
}
