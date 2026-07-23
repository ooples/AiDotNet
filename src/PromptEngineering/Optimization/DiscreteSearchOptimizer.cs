using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.Templates;

namespace AiDotNet.PromptEngineering.Optimization;

/// <summary>
/// Optimizer that uses discrete search to find better prompts by testing variations.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring.</typeparam>
/// <remarks>
/// <para>
/// This optimizer generates variations of prompt components and tests combinations to find
/// the best-performing prompt. It's interpretable and systematic.
/// </para>
/// <para><b>For Beginners:</b> Tries different versions of prompt parts and picks what works best.
///
/// Example:
/// ```csharp
/// var optimizer = new DiscreteSearchOptimizer&lt;double&gt;();
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
public class DiscreteSearchOptimizer<T> : PromptOptimizerBase<T>
{
    private readonly List<string> _instructionVariations;
    private readonly List<string> _formatVariations;

    /// <summary>
    /// Initializes a new instance of the DiscreteSearchOptimizer class with default variations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Creates a new optimizer with a set of default instruction and format variations that are
    /// commonly effective for prompt optimization across various tasks.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This constructor sets up the optimizer with some preset prompt variations that typically
    /// work well. You can add more variations later using AddInstructionVariations and AddFormatVariations.
    /// </para>
    /// </remarks>
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
    /// Initializes a new instance of the DiscreteSearchOptimizer class with custom variations.
    /// </summary>
    /// <param name="instructionVariations">Custom instruction variations to try. If null, uses defaults.</param>
    /// <param name="formatVariations">Custom format variations to try. If null, uses defaults.</param>
    /// <remarks>
    /// <para>
    /// Creates a new optimizer with custom instruction and format variations for prompt optimization.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// Use this constructor when you want to specify your own prompt variations instead of the defaults.
    /// Pass null for either parameter to use the default variations for that category.
    /// </para>
    /// </remarks>
    public DiscreteSearchOptimizer(
        IEnumerable<string>? instructionVariations = null,
        IEnumerable<string>? formatVariations = null)
    {
        // Use provided variations or defaults
        _instructionVariations = instructionVariations?.ToList() ?? new List<string>
        {
            "",
            "Please ",
            "Carefully ",
            "Think step-by-step and ",
            "Analyze and "
        };

        _formatVariations = formatVariations?.ToList() ?? new List<string>
        {
            "",
            "\n\nProvide your answer clearly.",
            "\n\nExplain your reasoning.",
            "\n\nBe concise and specific.",
            "\n\nAnswer in detail."
        };
    }

    /// <summary>
    /// Adds custom instruction variations to try during optimization.
    /// </summary>
    /// <param name="variations">The instruction variations to add.</param>
    /// <remarks>
    /// <para>
    /// Instruction variations are prepended to the prompt. They modify how the prompt begins.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// These are different ways to start your prompt. For example, "Please " or "Carefully "
    /// can change how the model interprets the instruction.
    /// </para>
    /// </remarks>
    public void AddInstructionVariations(params string[] variations)
    {
        if (variations is not null)
        {
            _instructionVariations.AddRange(variations);
        }
    }

    /// <summary>
    /// Adds custom format variations to try during optimization.
    /// </summary>
    /// <param name="variations">The format variations to add.</param>
    /// <remarks>
    /// <para>
    /// Format variations are appended to the prompt. They modify how the prompt ends.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// These are different ways to end your prompt. For example, asking for reasoning
    /// or requesting a specific format can improve output quality.
    /// </para>
    /// </remarks>
    public void AddFormatVariations(params string[] variations)
    {
        if (variations is not null)
        {
            _formatVariations.AddRange(variations);
        }
    }

    /// <summary>
    /// Optimizes the prompt using discrete search.
    /// </summary>
    /// <param name="initialPrompt">The initial prompt to optimize.</param>
    /// <param name="evaluationFunction">Function that evaluates a prompt and returns a score.</param>
    /// <param name="maxIterations">Maximum number of variations to try.</param>
    /// <returns>The best-performing prompt template found.</returns>
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

                // Keep if better (using NumOps for comparison)
                if (NumOps.GreaterThan(score, bestScore))
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
    /// <param name="initialPrompt">The initial prompt to optimize.</param>
    /// <param name="evaluationFunction">Async function that evaluates a prompt and returns a score.</param>
    /// <param name="maxIterations">Maximum number of variations to try.</param>
    /// <param name="cancellationToken">Token to cancel the operation.</param>
    /// <returns>The best-performing prompt template found.</returns>
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

                // Keep if better (using NumOps for comparison)
                if (NumOps.GreaterThan(score, bestScore))
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
