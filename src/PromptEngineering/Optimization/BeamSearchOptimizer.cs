using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.Templates;

namespace AiDotNet.PromptEngineering.Optimization;

/// <summary>
/// Optimizer that uses beam search to explore multiple promising prompt variations.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring.</typeparam>
/// <remarks>
/// <para>
/// Beam search maintains a fixed number of best candidates at each step,
/// exploring variations of all candidates before selecting the top performers.
/// </para>
/// <para><b>For Beginners:</b> Explores multiple paths simultaneously.
///
/// Example:
/// <code>
/// var optimizer = new BeamSearchOptimizer&lt;double&gt;(beamWidth: 5);
///
/// var optimized = optimizer.Optimize(
///     initialPrompt: "Classify sentiment:",
///     evaluationFunction: prompt => EvaluateAccuracy(prompt),
///     maxIterations: 50
/// );
/// </code>
///
/// How it works:
/// - Keep track of top N prompts (the "beam")
/// - Generate variations of all N prompts
/// - Score all variations
/// - Keep only the top N again
/// - Repeat
///
/// Benefits:
/// - More thorough than greedy search
/// - Less likely to get stuck in local optima
/// - Faster than exhaustive search
/// </para>
/// </remarks>
public class BeamSearchOptimizer<T> : PromptOptimizerBase<T>
{
    private readonly int _beamWidth;
    private readonly int _variationsPerCandidate;
    private readonly Random _random;

    private readonly List<string> _prefixes;
    private readonly List<string> _suffixes;
    private readonly List<string> _insertions;

    /// <summary>
    /// Initializes a new instance of the BeamSearchOptimizer class.
    /// </summary>
    /// <param name="beamWidth">Number of candidates to maintain at each step.</param>
    /// <param name="variationsPerCandidate">Number of variations to generate per candidate.</param>
    /// <param name="seed">Random seed for reproducibility (null for random).</param>
    public BeamSearchOptimizer(
        int beamWidth = 5,
        int variationsPerCandidate = 3,
        int? seed = null)
    {
        _beamWidth = Math.Max(1, beamWidth);
        _variationsPerCandidate = Math.Max(1, variationsPerCandidate);
        _random = seed.HasValue ? new Random(seed.Value) : new Random();

        _prefixes = new List<string>
        {
            "Please ",
            "Carefully ",
            "Think step-by-step and ",
            "Analyze and ",
            "Consider thoroughly and ",
            "Using best practices, ",
            "Based on your expertise, "
        };

        _suffixes = new List<string>
        {
            "\n\nProvide your answer clearly.",
            "\n\nExplain your reasoning.",
            "\n\nBe concise and specific.",
            "\n\nAnswer in detail.",
            "\n\nFormat your response appropriately.",
            "\n\nEnsure accuracy in your response."
        };

        _insertions = new List<string>
        {
            "carefully ",
            "thoroughly ",
            "systematically ",
            "precisely ",
            "accurately "
        };
    }

    /// <summary>
    /// Adds custom variations for beam search.
    /// </summary>
    public void AddVariations(
        IEnumerable<string>? prefixes = null,
        IEnumerable<string>? suffixes = null,
        IEnumerable<string>? insertions = null)
    {
        if (prefixes is not null) _prefixes.AddRange(prefixes);
        if (suffixes is not null) _suffixes.AddRange(suffixes);
        if (insertions is not null) _insertions.AddRange(insertions);
    }

    /// <summary>
    /// Optimizes using beam search.
    /// </summary>
    protected override IPromptTemplate OptimizeCore(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations)
    {
        // Initialize beam with initial prompt
        var beam = new List<(string Prompt, T Score)>
        {
            (initialPrompt, evaluationFunction(initialPrompt))
        };

        RecordIteration(0, initialPrompt, beam[0].Score);

        string bestPrompt = initialPrompt;
        T bestScore = beam[0].Score;
        int iteration = 1;

        while (iteration < maxIterations)
        {
            var candidates = new List<(string Prompt, T Score)>();

            // Generate variations for each beam candidate
            foreach (var candidate in beam)
            {
                var variations = GenerateVariations(candidate.Prompt);
                foreach (var variation in variations.Take(_variationsPerCandidate))
                {
                    if (iteration >= maxIterations) break;

                    var score = evaluationFunction(variation);
                    candidates.Add((variation, score));
                    RecordIteration(iteration, variation, score);

                    if (NumOps.GreaterThan(score, bestScore))
                    {
                        bestScore = score;
                        bestPrompt = variation;
                    }

                    iteration++;
                }
            }

            // Select top beam candidates
            beam = candidates
                .OrderByDescending(c => c.Score, Comparer<T>.Create((a, b) =>
                    NumOps.GreaterThan(a, b) ? 1 : (NumOps.LessThan(a, b) ? -1 : 0)))
                .Take(_beamWidth)
                .ToList();

            if (beam.Count == 0) break;
        }

        return new SimplePromptTemplate(bestPrompt);
    }

    /// <summary>
    /// Optimizes using beam search asynchronously.
    /// </summary>
    protected override async Task<IPromptTemplate> OptimizeCoreAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations,
        CancellationToken cancellationToken)
    {
        // Initialize beam with initial prompt
        var initialScore = await evaluationFunction(initialPrompt).ConfigureAwait(false);
        var beam = new List<(string Prompt, T Score)>
        {
            (initialPrompt, initialScore)
        };

        RecordIteration(0, initialPrompt, beam[0].Score);

        string bestPrompt = initialPrompt;
        T bestScore = beam[0].Score;
        int iteration = 1;

        while (iteration < maxIterations)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var candidates = new List<(string Prompt, T Score)>();

            // Generate and evaluate variations
            foreach (var candidate in beam)
            {
                var variations = GenerateVariations(candidate.Prompt).Take(_variationsPerCandidate);

                foreach (var variation in variations)
                {
                    if (iteration >= maxIterations) break;

                    var score = await evaluationFunction(variation).ConfigureAwait(false);
                    candidates.Add((variation, score));
                    RecordIteration(iteration, variation, score);

                    if (NumOps.GreaterThan(score, bestScore))
                    {
                        bestScore = score;
                        bestPrompt = variation;
                    }

                    iteration++;
                }
            }

            // Select top beam candidates
            beam = candidates
                .OrderByDescending(c => c.Score, Comparer<T>.Create((a, b) =>
                    NumOps.GreaterThan(a, b) ? 1 : (NumOps.LessThan(a, b) ? -1 : 0)))
                .Take(_beamWidth)
                .ToList();

            if (beam.Count == 0) break;
        }

        return new SimplePromptTemplate(bestPrompt);
    }

    private IEnumerable<string> GenerateVariations(string prompt)
    {
        var variations = new List<string>();

        // Add prefix variations
        foreach (var prefix in _prefixes.OrderBy(_ => _random.Next()).Take(2))
        {
            if (!prompt.StartsWith(prefix))
            {
                variations.Add(prefix + prompt);
            }
        }

        // Add suffix variations
        foreach (var suffix in _suffixes.OrderBy(_ => _random.Next()).Take(2))
        {
            if (!prompt.EndsWith(suffix.Trim()))
            {
                variations.Add(prompt + suffix);
            }
        }

        // Add word insertion variations
        var words = prompt.Split(' ').ToList();
        if (words.Count > 2)
        {
            var insertPos = _random.Next(1, words.Count);
            var insertion = _insertions[_random.Next(_insertions.Count)];
            words.Insert(insertPos, insertion);
            variations.Add(string.Join(" ", words));
        }

        // Add word removal variation (remove filler words)
        var fillers = new[] { "very", "really", "quite", "just", "simply" };
        var filtered = prompt.Split(' ')
            .Where(w => !fillers.Contains(w.ToLowerInvariant()))
            .ToArray();
        if (filtered.Length < prompt.Split(' ').Length)
        {
            variations.Add(string.Join(" ", filtered));
        }

        return variations.Distinct();
    }
}
