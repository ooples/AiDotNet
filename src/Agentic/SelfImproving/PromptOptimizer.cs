using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// Selects the best system prompt for an agent by measuring candidate prompts against a labeled eval set —
/// a DSPy-like, evaluation-driven prompt search. Each candidate builds an agent, runs the eval cases, and is
/// scored; the highest-scoring prompt wins.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// The optimizer is the <em>selection</em> half; the candidate prompts are the <em>search</em> half and can
/// come from anywhere — a hand-written set, an LLM that proposes variations, or AiDotNet's existing
/// genetic/beam/annealing prompt optimizers feeding their population in. Scoring defaults to a case-insensitive
/// substring match against the expected answer, overridable with a custom scorer (e.g., reusing an
/// <see cref="ITrajectoryEvaluator"/>).
/// </para>
/// <para><b>For Beginners:</b> Instead of guessing which wording works best, you give the optimizer a few
/// prompt options and a set of practice questions with answers. It tries each prompt on every question, counts
/// how many it gets right, and hands back the winner.
/// </para>
/// </remarks>
public sealed class PromptOptimizer<T>
{
    private readonly Func<AgentRunResult, PromptEvalCase, double> _scorer;

    /// <summary>
    /// Initializes a new prompt optimizer.
    /// </summary>
    /// <param name="scorer">
    /// Optional custom scorer for a run against a case (higher is better). <c>null</c> uses a case-insensitive
    /// substring match: 1.0 when the answer contains the expected text, else 0.0.
    /// </param>
    public PromptOptimizer(Func<AgentRunResult, PromptEvalCase, double>? scorer = null)
    {
        _scorer = scorer ?? DefaultScorer;
    }

    /// <summary>
    /// Evaluates each candidate prompt over the eval set and returns the best.
    /// </summary>
    /// <param name="candidatePrompts">The system prompts to compare. Must be non-empty.</param>
    /// <param name="agentFactory">Builds an agent configured with the given system prompt.</param>
    /// <param name="cases">The labeled eval set. Must be non-empty.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>The best prompt and the ranked candidate scores.</returns>
    /// <exception cref="ArgumentNullException">Thrown when any argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="candidatePrompts"/> or <paramref name="cases"/> is empty.</exception>
    public async Task<PromptOptimizationResult> OptimizeAsync(
        IReadOnlyList<string> candidatePrompts,
        Func<string, IAgent<T>> agentFactory,
        IReadOnlyList<PromptEvalCase> cases,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(candidatePrompts);
        Guard.NotNull(agentFactory);
        Guard.NotNull(cases);
        if (candidatePrompts.Count == 0)
        {
            throw new ArgumentException("At least one candidate prompt is required.", nameof(candidatePrompts));
        }

        if (cases.Count == 0)
        {
            throw new ArgumentException("At least one eval case is required.", nameof(cases));
        }

        var scored = new List<ScoredPrompt>(candidatePrompts.Count);
        foreach (var prompt in candidatePrompts)
        {
            Guard.NotNull(prompt);
            var agent = agentFactory(prompt);
            Guard.NotNull(agent);

            var sum = 0.0;
            foreach (var evalCase in cases)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var result = await agent.RunAsync(new[] { ChatMessage.User(evalCase.Input) }, cancellationToken)
                    .ConfigureAwait(false);
                sum += _scorer(result, evalCase);
            }

            scored.Add(new ScoredPrompt(prompt, sum / cases.Count));
        }

        var ranked = scored.OrderByDescending(s => s.Score).ToList();
        var best = ranked[0];
        return new PromptOptimizationResult(best.Prompt, best.Score, ranked);
    }

    private static double DefaultScorer(AgentRunResult result, PromptEvalCase evalCase) =>
        result.FinalText.IndexOf(evalCase.Expected, StringComparison.OrdinalIgnoreCase) >= 0 ? 1.0 : 0.0;
}
