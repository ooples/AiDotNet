using AiDotNet.Interfaces;
using AiDotNet.PromptEngineering.Templates;

namespace AiDotNet.PromptEngineering.Optimization;

/// <summary>
/// Optimizer that combines multiple optimization strategies for better results.
/// </summary>
/// <typeparam name="T">The type of numeric data used for scoring.</typeparam>
/// <remarks>
/// <para>
/// This optimizer runs multiple optimization strategies and combines their results,
/// using voting, averaging, or selection to determine the best prompt.
/// </para>
/// <para><b>For Beginners:</b> Like getting multiple opinions and combining them.
///
/// Example:
/// <code>
/// var optimizer = new EnsembleOptimizer&lt;double&gt;(
///     new DiscreteSearchOptimizer&lt;double&gt;(),
///     new BeamSearchOptimizer&lt;double&gt;(),
///     new GeneticOptimizer&lt;double&gt;()
/// );
///
/// var optimized = optimizer.Optimize(
///     initialPrompt: "Classify sentiment:",
///     evaluationFunction: prompt => EvaluateAccuracy(prompt),
///     maxIterations: 100  // Divided among strategies
/// );
/// </code>
///
/// How it works:
/// - Run multiple optimizers independently
/// - Collect their best results
/// - Pick the overall best (or combine them)
///
/// Benefits:
/// - More robust than single strategy
/// - Different strategies find different optima
/// - Better coverage of solution space
/// </para>
/// </remarks>
public class EnsembleOptimizer<T> : PromptOptimizerBase<T>
{
    private readonly List<IPromptOptimizer<T>> _optimizers;
    private readonly EnsembleStrategy _strategy;

    /// <summary>
    /// Strategy for combining ensemble results.
    /// </summary>
    public enum EnsembleStrategy
    {
        /// <summary>Select the single best result across all optimizers.</summary>
        BestWins,

        /// <summary>Run sequentially, each starting from previous best.</summary>
        Sequential,

        /// <summary>Run in parallel and pick best result.</summary>
        Parallel
    }

    /// <summary>
    /// Initializes a new instance of the EnsembleOptimizer class with specified optimizers.
    /// </summary>
    /// <param name="optimizers">The optimizers to include in the ensemble.</param>
    public EnsembleOptimizer(params IPromptOptimizer<T>[] optimizers)
        : this(EnsembleStrategy.BestWins, optimizers)
    {
    }

    /// <summary>
    /// Initializes a new instance of the EnsembleOptimizer class with strategy and optimizers.
    /// </summary>
    /// <param name="strategy">The strategy for combining results.</param>
    /// <param name="optimizers">The optimizers to include in the ensemble.</param>
    public EnsembleOptimizer(EnsembleStrategy strategy, params IPromptOptimizer<T>[] optimizers)
    {
        _strategy = strategy;
        _optimizers = optimizers?.ToList() ?? new List<IPromptOptimizer<T>>();

        if (_optimizers.Count == 0)
        {
            // Add default optimizers
            _optimizers.Add(new DiscreteSearchOptimizer<T>());
            _optimizers.Add(new BeamSearchOptimizer<T>());
        }
    }

    /// <summary>
    /// Adds an optimizer to the ensemble.
    /// </summary>
    /// <param name="optimizer">The optimizer to add.</param>
    public void AddOptimizer(IPromptOptimizer<T> optimizer)
    {
        if (optimizer is not null)
        {
            _optimizers.Add(optimizer);
        }
    }

    /// <summary>
    /// Creates a default ensemble with common optimization strategies.
    /// </summary>
    /// <returns>An ensemble optimizer with default strategies.</returns>
    public static EnsembleOptimizer<T> CreateDefault()
    {
        return new EnsembleOptimizer<T>(
            EnsembleStrategy.BestWins,
            new DiscreteSearchOptimizer<T>(),
            new BeamSearchOptimizer<T>(beamWidth: 3),
            new SimulatedAnnealingOptimizer<T>()
        );
    }

    /// <summary>
    /// Creates an aggressive ensemble with more optimizers.
    /// </summary>
    /// <returns>An ensemble optimizer with aggressive strategies.</returns>
    public static EnsembleOptimizer<T> CreateAggressive()
    {
        return new EnsembleOptimizer<T>(
            EnsembleStrategy.Sequential,
            new DiscreteSearchOptimizer<T>(),
            new BeamSearchOptimizer<T>(beamWidth: 5),
            new GeneticOptimizer<T>(populationSize: 10),
            new SimulatedAnnealingOptimizer<T>(initialTemperature: 2.0)
        );
    }

    /// <summary>
    /// Optimizes using ensemble strategies.
    /// </summary>
    protected override IPromptTemplate OptimizeCore(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations)
    {
        return _strategy switch
        {
            EnsembleStrategy.Sequential => OptimizeSequential(initialPrompt, evaluationFunction, maxIterations),
            EnsembleStrategy.Parallel => OptimizeParallel(initialPrompt, evaluationFunction, maxIterations),
            _ => OptimizeBestWins(initialPrompt, evaluationFunction, maxIterations)
        };
    }

    /// <summary>
    /// Optimizes using ensemble strategies asynchronously.
    /// </summary>
    protected override async Task<IPromptTemplate> OptimizeCoreAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations,
        CancellationToken cancellationToken)
    {
        return _strategy switch
        {
            EnsembleStrategy.Sequential => await OptimizeSequentialAsync(
                initialPrompt, evaluationFunction, maxIterations, cancellationToken).ConfigureAwait(false),
            EnsembleStrategy.Parallel => await OptimizeParallelAsync(
                initialPrompt, evaluationFunction, maxIterations, cancellationToken).ConfigureAwait(false),
            _ => await OptimizeBestWinsAsync(
                initialPrompt, evaluationFunction, maxIterations, cancellationToken).ConfigureAwait(false)
        };
    }

    private IPromptTemplate OptimizeBestWins(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations)
    {
        var iterationsPerOptimizer = maxIterations / _optimizers.Count;
        var results = new List<(IPromptTemplate Template, T Score)>();
        int iteration = 0;

        foreach (var optimizer in _optimizers)
        {
            var result = optimizer.Optimize(initialPrompt, evaluationFunction, iterationsPerOptimizer);
            var score = evaluationFunction(result.Format(new Dictionary<string, string>()));
            results.Add((result, score));

            // Record history
            foreach (var entry in optimizer.GetOptimizationHistory())
            {
                RecordIteration(iteration++, entry.Prompt, entry.Score);
            }
        }

        // Find best
        var best = results
            .OrderByDescending(r => r.Score, Comparer<T>.Create((a, b) =>
                NumOps.GreaterThan(a, b) ? 1 : (NumOps.LessThan(a, b) ? -1 : 0)))
            .First();

        return best.Template;
    }

    private IPromptTemplate OptimizeSequential(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations)
    {
        var iterationsPerOptimizer = maxIterations / _optimizers.Count;
        string currentPrompt = initialPrompt;
        IPromptTemplate bestTemplate = new SimplePromptTemplate(initialPrompt);
        T bestScore = evaluationFunction(initialPrompt);
        int iteration = 0;

        foreach (var optimizer in _optimizers)
        {
            var result = optimizer.Optimize(currentPrompt, evaluationFunction, iterationsPerOptimizer);
            var renderedPrompt = result.Format(new Dictionary<string, string>());
            var score = evaluationFunction(renderedPrompt);

            // Record history
            foreach (var entry in optimizer.GetOptimizationHistory())
            {
                RecordIteration(iteration++, entry.Prompt, entry.Score);
            }

            if (NumOps.GreaterThan(score, bestScore))
            {
                bestScore = score;
                bestTemplate = result;
            }

            // Use best result as starting point for next optimizer
            currentPrompt = renderedPrompt;
        }

        return bestTemplate;
    }

    private IPromptTemplate OptimizeParallel(
        string initialPrompt,
        Func<string, T> evaluationFunction,
        int maxIterations)
    {
        var iterationsPerOptimizer = maxIterations / _optimizers.Count;

        // Run all optimizers in parallel
        var tasks = _optimizers.Select(opt =>
            Task.Run(() => opt.Optimize(initialPrompt, evaluationFunction, iterationsPerOptimizer)))
            .ToArray();

        Task.WaitAll(tasks);

        var results = tasks.Select(t => t.Result).ToList();

        // Evaluate all results
        var evaluated = results
            .Select(r => (Template: r, Score: evaluationFunction(r.Format(new Dictionary<string, string>()))))
            .OrderByDescending(r => r.Score, Comparer<T>.Create((a, b) =>
                NumOps.GreaterThan(a, b) ? 1 : (NumOps.LessThan(a, b) ? -1 : 0)))
            .First();

        // Record combined history
        int iteration = 0;
        foreach (var optimizer in _optimizers)
        {
            foreach (var entry in optimizer.GetOptimizationHistory())
            {
                RecordIteration(iteration++, entry.Prompt, entry.Score);
            }
        }

        return evaluated.Template;
    }

    private async Task<IPromptTemplate> OptimizeBestWinsAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations,
        CancellationToken cancellationToken)
    {
        var iterationsPerOptimizer = maxIterations / _optimizers.Count;
        var results = new List<(IPromptTemplate Template, T Score)>();
        int iteration = 0;

        foreach (var optimizer in _optimizers)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var result = await optimizer.OptimizeAsync(
                initialPrompt, evaluationFunction, iterationsPerOptimizer, cancellationToken)
                .ConfigureAwait(false);

            var score = await evaluationFunction(result.Format(new Dictionary<string, string>()))
                .ConfigureAwait(false);

            results.Add((result, score));

            foreach (var entry in optimizer.GetOptimizationHistory())
            {
                RecordIteration(iteration++, entry.Prompt, entry.Score);
            }
        }

        var best = results
            .OrderByDescending(r => r.Score, Comparer<T>.Create((a, b) =>
                NumOps.GreaterThan(a, b) ? 1 : (NumOps.LessThan(a, b) ? -1 : 0)))
            .First();

        return best.Template;
    }

    private async Task<IPromptTemplate> OptimizeSequentialAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations,
        CancellationToken cancellationToken)
    {
        var iterationsPerOptimizer = maxIterations / _optimizers.Count;
        string currentPrompt = initialPrompt;
        IPromptTemplate bestTemplate = new SimplePromptTemplate(initialPrompt);
        T bestScore = await evaluationFunction(initialPrompt).ConfigureAwait(false);
        int iteration = 0;

        foreach (var optimizer in _optimizers)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var result = await optimizer.OptimizeAsync(
                currentPrompt, evaluationFunction, iterationsPerOptimizer, cancellationToken)
                .ConfigureAwait(false);

            var renderedPrompt = result.Format(new Dictionary<string, string>());
            var score = await evaluationFunction(renderedPrompt).ConfigureAwait(false);

            foreach (var entry in optimizer.GetOptimizationHistory())
            {
                RecordIteration(iteration++, entry.Prompt, entry.Score);
            }

            if (NumOps.GreaterThan(score, bestScore))
            {
                bestScore = score;
                bestTemplate = result;
            }

            currentPrompt = renderedPrompt;
        }

        return bestTemplate;
    }

    private async Task<IPromptTemplate> OptimizeParallelAsync(
        string initialPrompt,
        Func<string, Task<T>> evaluationFunction,
        int maxIterations,
        CancellationToken cancellationToken)
    {
        var iterationsPerOptimizer = maxIterations / _optimizers.Count;

        var tasks = _optimizers.Select(opt =>
            opt.OptimizeAsync(initialPrompt, evaluationFunction, iterationsPerOptimizer, cancellationToken))
            .ToArray();

        var results = await Task.WhenAll(tasks).ConfigureAwait(false);

        var evaluatedTasks = results.Select(async r =>
        {
            var score = await evaluationFunction(r.Format(new Dictionary<string, string>()))
                .ConfigureAwait(false);
            return (Template: r, Score: score);
        });

        var evaluated = await Task.WhenAll(evaluatedTasks).ConfigureAwait(false);

        var best = evaluated
            .OrderByDescending(r => r.Score, Comparer<T>.Create((a, b) =>
                NumOps.GreaterThan(a, b) ? 1 : (NumOps.LessThan(a, b) ? -1 : 0)))
            .First();

        int iteration = 0;
        foreach (var optimizer in _optimizers)
        {
            foreach (var entry in optimizer.GetOptimizationHistory())
            {
                RecordIteration(iteration++, entry.Prompt, entry.Score);
            }
        }

        return best.Template;
    }
}
