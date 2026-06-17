using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// The outcome of an agent run: the final text answer, the full conversation transcript it produced,
/// how many model calls it took, whether it finished cleanly, and the aggregate token usage.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Messages"/> is the complete transcript the agent worked with (system prompt, the inbound
/// conversation, every assistant turn, and every tool-result message), so a caller can inspect the
/// agent's reasoning, persist the thread, or feed it into another agent. <see cref="Completed"/> is
/// <c>false</c> only when the agent hit its iteration cap before producing a final (non-tool) answer.
/// </para>
/// <para><b>For Beginners:</b> After an agent runs, this is what you get back. <see cref="FinalText"/> is
/// the answer most callers want; <see cref="Messages"/> is the full play-by-play if you want to see how it
/// got there; <see cref="Completed"/> tells you whether it actually finished or ran out of allowed steps.
/// </para>
/// </remarks>
public sealed class AgentRunResult
{
    private AgentRunResult(
        string finalText,
        IReadOnlyList<ChatMessage> messages,
        int iterations,
        bool completed,
        ChatUsage? usage,
        string? agentName)
    {
        FinalText = finalText;
        // Snapshot the transcript: callers typically pass the live, mutable
        // transcript list, and a historical result must not be rewritten by
        // later mutations of that list (replay/debugging integrity).
        var copy = new ChatMessage[messages.Count];
        for (var i = 0; i < messages.Count; i++)
        {
            copy[i] = messages[i];
        }
        Messages = Array.AsReadOnly(copy);
        Iterations = iterations;
        Completed = completed;
        Usage = usage;
        AgentName = agentName;
    }

    /// <summary>
    /// Gets the agent's final text answer. When <see cref="Completed"/> is <c>false</c> this is the text of
    /// the last message produced before the iteration cap was hit (possibly empty).
    /// </summary>
    public string FinalText { get; }

    /// <summary>
    /// Gets the complete conversation transcript the agent produced, including the system prompt (if any),
    /// the inbound messages, every assistant turn, and every tool-result message.
    /// </summary>
    public IReadOnlyList<ChatMessage> Messages { get; }

    /// <summary>
    /// Gets the number of model calls the run made (each tool round is one call).
    /// </summary>
    public int Iterations { get; }

    /// <summary>
    /// Gets a value indicating whether the agent produced a final answer (<c>true</c>) or stopped because it
    /// reached its iteration cap (<c>false</c>).
    /// </summary>
    public bool Completed { get; }

    /// <summary>
    /// Gets the aggregate token usage across every model call in the run, or <c>null</c> when no call
    /// reported usage.
    /// </summary>
    public ChatUsage? Usage { get; }

    /// <summary>
    /// Gets the name of the agent that produced the final answer, or <c>null</c> when not tracked. For a
    /// single agent this is its own name; for a swarm it is the member that was active when the run ended.
    /// </summary>
    public string? AgentName { get; }

    /// <summary>
    /// Creates a result for a run that produced a final answer.
    /// </summary>
    internal static AgentRunResult Finished(
        string finalText,
        IReadOnlyList<ChatMessage> messages,
        int iterations,
        ChatUsage? usage,
        string? agentName = null) =>
        new(finalText, messages, iterations, completed: true, usage, agentName);

    /// <summary>
    /// Creates a result for a run that hit its iteration cap before producing a final answer.
    /// </summary>
    internal static AgentRunResult Stopped(
        string lastText,
        IReadOnlyList<ChatMessage> messages,
        int iterations,
        ChatUsage? usage,
        string? agentName = null) =>
        new(lastText, messages, iterations, completed: false, usage, agentName);
}
