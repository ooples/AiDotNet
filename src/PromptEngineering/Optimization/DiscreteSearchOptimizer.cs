using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.Templates;

namespace AiDotNet.PromptEngineering.Optimization;

/// <summary>
/// Optimizer that uses discrete search to find better prompts by testing variations.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring (must be comparable).</typeparam>
/// <remarks>
/// <para>
/// This optimizer generates variations of prompt components and tests combinations to find
/// the best-performing prompt. It's interpretable and systematic.
/// </para>
/// <para><b>For Beginners:</b> Tries different versions of prompt parts and picks what works best.
///
/// Example:
/// ```csharp
/// var optimizer = new DiscreteSearchOptimizer<double>();
///
/// // Evaluation function that returns accuracy
/// double Evaluate(string prompt)
/// {
///     var correct = 0;
///     foreach (var test in testCases)
///     {
///         var result = model.Generate(prompt + test.Input);
///         if (result == test.Expected) correct++;
///     }
///     return correct / (double)testCases.Count;
/// }
///
/// var optimized = optimizer.Optimize(
///     initialPrompt: "Classify the sentiment:",
///     evaluationFunction: Evaluate,
///     maxIterations: 50
/// );
///
/// // Returns best-performing prompt variation
/// ```
/// </para>
/// </remarks>
public class DiscreteSearchOptimizer<T> : PromptOptimizerBase<T> where T : IComparable<T>
{
    private readonly List<string> _instructionVariations;
    private readonly List<string> _formatVariations;

    /// <summary>
    /// Initializes a new instance of the DiscreteSearchOptimizer class.
    /// </summary>
    public DiscreteSearchOptimizer()
    {
        // Default variations to try
        _instructionVariations = new List<string>
        {
            "",
            "Please ",
            "Carefully ",
            "Think step-by-step and ",
            "Analyze and "
        };

        _formatVariations = new List<string>
        {
            "",
            "\n\nProvide your answer clearly.",
            "\n\nExplain your reasoning.",
            "\n\nBe concise and specific.",
            "\n\nAnswer in detail."
        };
    }

    /// <summary>
    /// Adds custom instruction variations to try.
    /// </summary>
    public void AddInstructionVariations(params string[] variations)
    {
        _instructionVariations.AddRange(variations);
    }

    /// <summary>
    /// Adds custom format variations to try.
    /// </summary>
    public void AddFormatVariations(params string[] variations)
    {
        _formatVariations.AddRange(variations);
    }

    /// <summary>
    /// Optimizes the prompt using discrete search.
    /// </summary>
    protected override IPromptTemplate OptimizeCore(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations)
    {
        string bestPrompt = initialPrompt;
        T bestScore = evaluationFunction(initialPrompt);
        RecordIteration(0, initialPrompt, bestScore);

        int iteration = 1;

        // Try different combinations of variations
        foreach (var instruction in _instructionVariations)
        {
            if (iteration >= maxIterations) break;

            foreach (var format in _formatVariations)
            {
                if (iteration >= maxIterations) break;

                // Create variation
                var candidatePrompt = instruction + initialPrompt + format;

                // Evaluate
                var score = evaluationFunction(candidatePrompt);
                RecordIteration(iteration, candidatePrompt, score);

                // Keep if better
                if (score.CompareTo(bestScore) > 0)
                {
                    bestScore = score;
                    bestPrompt = candidatePrompt;
                }

                iteration++;
            }
        }

        return new SimplePromptTemplate(bestPrompt);
    }

    /// <summary>
    /// Optimizes the prompt asynchronously using discrete search.
    /// </summary>
    protected override async Task<IPromptTemplate> OptimizeCoreAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations,
        CancellationToken cancellationToken)
    {
        string bestPrompt = initialPrompt;
        T bestScore = await evaluationFunction(initialPrompt).ConfigureAwait(false);
        RecordIteration(0, initialPrompt, bestScore);

        int iteration = 1;

        // Try different combinations of variations
        foreach (var instruction in _instructionVariations)
        {
            if (iteration >= maxIterations || cancellationToken.IsCancellationRequested)
                break;

            foreach (var format in _formatVariations)
            {
                if (iteration >= maxIterations || cancellationToken.IsCancellationRequested)
                    break;

                // Create variation
                var candidatePrompt = instruction + initialPrompt + format;

                // Evaluate
                var score = await evaluationFunction(candidatePrompt).ConfigureAwait(false);
                RecordIteration(iteration, candidatePrompt, score);

                // Keep if better
                if (score.CompareTo(bestScore) > 0)
                {
                    bestScore = score;
                    bestPrompt = candidatePrompt;
                }

                iteration++;
            }
        }

        return new SimplePromptTemplate(bestPrompt);
    }
}
