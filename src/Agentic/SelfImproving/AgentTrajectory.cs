using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.SelfImproving;

/// <summary>
/// A captured record of one agent run — the structured "trajectory" the self-improving layer learns from.
/// It pairs what the agent did (the full message transcript, the final answer, step count, token usage) with
/// an optional quality <see cref="Reward"/> assigned later by an evaluator.
/// </summary>
/// <remarks>
/// <para>
/// Trajectories are the raw material for every self-improvement mechanism: continuous evaluation scores them,
/// learned routers/tool policies train on them, prompt optimizers measure prompts against them, and
/// reward-filtered trajectories become fine-tuning (LoRA) data. Capturing a run is non-invasive — a
/// <see cref="TracingAgent{T}"/> wraps any agent and records each run without changing its behavior.
/// </para>
/// <para><b>For Beginners:</b> Think of this as the flight recorder for an agent: it saves exactly what was
/// said and done on a run, plus (once graded) how good the outcome was. Collect many of these and the system
/// can learn what works and get better over time.
/// </para>
/// </remarks>
public sealed class AgentTrajectory
{
    /// <summary>
    /// Initializes a new trajectory.
    /// </summary>
    /// <param name="id">The unique id for this trajectory.</param>
    /// <param name="agentName">The name of the agent that produced the run.</param>
    /// <param name="messages">The full conversation transcript the run produced.</param>
    /// <param name="finalText">The agent's final answer.</param>
    /// <param name="iterations">The number of model calls the run took.</param>
    /// <param name="usage">Aggregate token usage, when available.</param>
    /// <param name="reward">An optional quality score (assigned by an evaluator). <c>null</c> until graded.</param>
    /// <param name="metadata">Optional key/value metadata.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="id"/>, <paramref name="agentName"/>, <paramref name="messages"/>, or <paramref name="finalText"/> is <c>null</c>.</exception>
    public AgentTrajectory(
        string id,
        string agentName,
        IReadOnlyList<ChatMessage> messages,
        string finalText,
        int iterations,
        ChatUsage? usage = null,
        double? reward = null,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        Guard.NotNullOrWhiteSpace(id);
        Guard.NotNull(agentName);
        Guard.NotNull(messages);
        Guard.NotNull(finalText);
        Id = id;
        AgentName = agentName;
        Messages = messages;
        FinalText = finalText;
        Iterations = iterations;
        Usage = usage;
        Reward = reward;
        Metadata = metadata;
    }

    /// <summary>Gets the unique id for this trajectory.</summary>
    public string Id { get; }

    /// <summary>Gets the name of the agent that produced the run.</summary>
    public string AgentName { get; }

    /// <summary>Gets the full conversation transcript the run produced.</summary>
    public IReadOnlyList<ChatMessage> Messages { get; }

    /// <summary>Gets the agent's final answer.</summary>
    public string FinalText { get; }

    /// <summary>Gets the number of model calls the run took.</summary>
    public int Iterations { get; }

    /// <summary>Gets aggregate token usage, or <c>null</c> when not reported.</summary>
    public ChatUsage? Usage { get; }

    /// <summary>
    /// Gets or sets the quality score for this trajectory, assigned by an evaluator (higher is better), or
    /// <c>null</c> while ungraded.
    /// </summary>
    public double? Reward { get; set; }

    /// <summary>Gets optional key/value metadata, or <c>null</c> when none was supplied.</summary>
    public IReadOnlyDictionary<string, string>? Metadata { get; }
}
