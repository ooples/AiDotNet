namespace AiDotNet.Agentic.SelfImproving;

/// <summary>A candidate prompt paired with its mean score over the eval set.</summary>
public sealed class ScoredPrompt
{
    /// <summary>Initializes a new scored prompt.</summary>
    /// <param name="prompt">The candidate prompt.</param>
    /// <param name="score">Its mean score over the eval set (higher is better).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="prompt"/> is <c>null</c>.</exception>
    public ScoredPrompt(string prompt, double score)
    {
        Guard.NotNull(prompt);
        Prompt = prompt;
        Score = score;
    }

    /// <summary>Gets the candidate prompt.</summary>
    public string Prompt { get; }

    /// <summary>Gets the mean score over the eval set.</summary>
    public double Score { get; }
}

/// <summary>
/// The outcome of prompt optimization: the best-scoring prompt plus the full ranked list of candidates.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After trying each prompt on the practice questions, this tells you which prompt
/// scored best (use that one) and shows every prompt's score so you can see the spread.
/// </para>
/// </remarks>
public sealed class PromptOptimizationResult
{
    /// <summary>Initializes a new result.</summary>
    /// <param name="bestPrompt">The highest-scoring prompt.</param>
    /// <param name="bestScore">The best prompt's mean score.</param>
    /// <param name="candidates">All candidates with their scores (ranked best first).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="bestPrompt"/> or <paramref name="candidates"/> is <c>null</c>.</exception>
    public PromptOptimizationResult(string bestPrompt, double bestScore, IReadOnlyList<ScoredPrompt> candidates)
    {
        Guard.NotNull(bestPrompt);
        Guard.NotNull(candidates);
        BestPrompt = bestPrompt;
        BestScore = bestScore;
        Candidates = candidates;
    }

    /// <summary>Gets the highest-scoring prompt.</summary>
    public string BestPrompt { get; }

    /// <summary>Gets the best prompt's mean score.</summary>
    public double BestScore { get; }

    /// <summary>Gets all candidates with their scores, ranked best first.</summary>
    public IReadOnlyList<ScoredPrompt> Candidates { get; }
}
