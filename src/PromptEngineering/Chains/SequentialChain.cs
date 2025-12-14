namespace AiDotNet.PromptEngineering.Chains;

/// <summary>
/// Chain that executes steps sequentially, passing output from each step to the next.
/// </summary>
/// <remarks>
/// <para>
/// A sequential chain runs operations in order, where each step's output becomes the next step's input.
/// This is the most common and straightforward chain type.
/// </para>
/// <para><b>For Beginners:</b> Runs steps one after another, like a recipe.
///
/// Example:
/// ```csharp
/// var chain = new SequentialChain<string, string>("ProcessingChain", "Processes text through multiple steps");
///
/// // Add steps
/// chain.AddStep("ExtractKeywords", text => ExtractKeywords(text));
/// chain.AddStep("Summarize", keywords => Summarize(keywords));
/// chain.AddStep("Translate", summary => Translate(summary, "es"));
///
/// // Run chain
/// var result = chain.Run("Long article text...");
/// // Step 1: Extract keywords → "AI, machine learning, neural networks"
/// // Step 2: Summarize keywords → "Article about AI and ML"
/// // Step 3: Translate → "Artículo sobre IA y ML"
/// ```
/// </para>
/// </remarks>
public class SequentialChain<TInput, TOutput> : ChainBase<TInput, TOutput>
{
    private readonly List<ChainStep> _steps;
    private IReadOnlyList<string>? _cachedStepNames;

    /// <summary>
    /// Initializes a new instance of the SequentialChain class.
    /// </summary>
    /// <param name="name">The name of the chain.</param>
    /// <param name="description">The description of the chain.</param>
    public SequentialChain(string name, string description = "")
        : base(name, description)
    {
        _steps = new List<ChainStep>();
    }

    /// <summary>
    /// Adds a step to the chain.
    /// </summary>
    /// <param name="stepName">The name of the step.</param>
    /// <param name="stepFunction">The function to execute for this step.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public SequentialChain<TInput, TOutput> AddStep(string stepName, Func<object, object> stepFunction)
    {
        if (string.IsNullOrWhiteSpace(stepName))
        {
            throw new ArgumentException("Step name cannot be empty.", nameof(stepName));
        }

        if (stepFunction == null)
        {
            throw new ArgumentNullException(nameof(stepFunction));
        }

        _steps.Add(new ChainStep(stepName, stepFunction, null));
        _cachedStepNames = null; // Invalidate cache
        return this;
    }

    /// <summary>
    /// Adds an asynchronous step to the chain.
    /// </summary>
    /// <param name="stepName">The name of the step.</param>
    /// <param name="stepFunction">The async function to execute for this step.</param>
    /// <returns>This chain instance for method chaining.</returns>
    public SequentialChain<TInput, TOutput> AddStepAsync(string stepName, Func<object, CancellationToken, Task<object>> stepFunction)
    {
        if (string.IsNullOrWhiteSpace(stepName))
        {
            throw new ArgumentException("Step name cannot be empty.", nameof(stepName));
        }

        if (stepFunction == null)
        {
            throw new ArgumentNullException(nameof(stepFunction));
        }

        _steps.Add(new ChainStep(stepName, null, stepFunction));
        _cachedStepNames = null; // Invalidate cache
        return this;
    }

    /// <summary>
    /// Gets the current steps in the chain.
    /// </summary>
    /// <remarks>
    /// This property uses lazy initialization with caching to avoid repeated allocations.
    /// The cache is automatically invalidated when steps are added.
    /// </remarks>
    public IReadOnlyList<string> Steps
    {
        get
        {
            if (_cachedStepNames is null)
            {
                _cachedStepNames = _steps.Select(s => s.Name).ToList().AsReadOnly();
            }

            return _cachedStepNames;
        }
    }

    /// <summary>
    /// Executes all steps sequentially.
    /// </summary>
    protected override TOutput RunCore(TInput input)
    {
        object current = input!;

        foreach (var step in _steps)
        {
            if (step.SyncFunction != null)
            {
                current = step.SyncFunction(current);
            }
            else if (step.AsyncFunction != null)
            {
                throw new InvalidOperationException(
                    $"Step '{step.Name}' was registered as async. Use RunAsync when executing chains that include async steps.");
            }
            else
            {
                throw new InvalidOperationException($"Step '{step.Name}' has no function defined.");
            }
        }

        if (current is TOutput output)
        {
            return output;
        }

        throw new InvalidOperationException($"Final output type mismatch. Expected {typeof(TOutput).Name}, got {current?.GetType().Name ?? "null"}");
    }

    /// <summary>
    /// Executes all steps sequentially asynchronously.
    /// </summary>
    protected override async Task<TOutput> RunCoreAsync(TInput input, CancellationToken cancellationToken)
    {
        object current = input!;

        foreach (var step in _steps)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (step.AsyncFunction != null)
            {
                current = await step.AsyncFunction(current, cancellationToken).ConfigureAwait(false);
            }
            else if (step.SyncFunction != null)
            {
                current = step.SyncFunction(current);
            }
            else
            {
                throw new InvalidOperationException($"Step '{step.Name}' has no function defined.");
            }
        }

        if (current is TOutput output)
        {
            return output;
        }

        throw new InvalidOperationException($"Final output type mismatch. Expected {typeof(TOutput).Name}, got {current?.GetType().Name ?? "null"}");
    }

    /// <summary>
    /// Represents a single step in the chain.
    /// </summary>
    private class ChainStep
    {
        public string Name { get; }
        public Func<object, object>? SyncFunction { get; }
        public Func<object, CancellationToken, Task<object>>? AsyncFunction { get; }

        public ChainStep(
            string name,
            Func<object, object>? syncFunction,
            Func<object, CancellationToken, Task<object>>? asyncFunction)
        {
            Name = name;
            SyncFunction = syncFunction;
            AsyncFunction = asyncFunction;
        }
    }
}
