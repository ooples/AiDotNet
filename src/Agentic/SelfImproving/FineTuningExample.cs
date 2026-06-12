namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// One supervised fine-tuning example distilled from a high-reward trajectory: the prompt the agent saw and
/// the completion it produced that earned a good score.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A "do it like this" training pair — the question plus a known-good answer —
/// harvested from a run that scored well. Many of these teach a local model to imitate its own best behavior.
/// </para>
/// </remarks>
public sealed class FineTuningExample
{
    /// <summary>
    /// Initializes a new example.
    /// </summary>
    /// <param name="prompt">The input/context the agent was given.</param>
    /// <param name="completion">The (good) completion to learn.</param>
    /// <param name="reward">The reward the originating trajectory earned.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="prompt"/> or <paramref name="completion"/> is <c>null</c>.</exception>
    public FineTuningExample(string prompt, string completion, double reward)
    {
        Guard.NotNull(prompt);
        Guard.NotNull(completion);
        Prompt = prompt;
        Completion = completion;
        Reward = reward;
    }

    /// <summary>Gets the input/context the agent was given.</summary>
    public string Prompt { get; }

    /// <summary>Gets the completion to learn.</summary>
    public string Completion { get; }

    /// <summary>Gets the reward the originating trajectory earned.</summary>
    public double Reward { get; }
}
