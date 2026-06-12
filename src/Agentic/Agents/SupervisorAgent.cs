using System.Text;
using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// A coordinator agent that supervises a team of specialized worker <see cref="IAgent{T}"/> instances and
/// routes work to them via native tool-calling. Each worker is surfaced as a <c>transfer_to_&lt;worker&gt;</c>
/// handoff tool, so the supervisor decides — turn by turn — which teammate to delegate to, reads their
/// result, and continues until it produces a final answer.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// Because a <see cref="SupervisorAgent{T}"/> is itself an <see cref="IAgent{T}"/>, supervisors compose:
/// a top-level supervisor can have other supervisors as workers, forming hierarchical teams. The routing
/// itself reuses <see cref="AgentExecutor{T}"/>'s tool-calling loop, so there is no bespoke control flow —
/// delegation is just the coordinator calling tools that happen to be other agents.
/// </para>
/// <para><b>For Beginners:</b> Think of a project lead with a team of specialists. You give the lead a
/// goal; the lead picks the right specialist for each step ("you handle the math", "you write the summary"),
/// collects their work, and reports back the final result. You only talk to the lead.
/// </para>
/// </remarks>
public sealed class SupervisorAgent<T> : IAgent<T>
{
    private readonly AgentExecutor<T> _executor;

    /// <summary>
    /// Initializes a new supervisor over the supplied worker agents.
    /// </summary>
    /// <param name="coordinator">The chat model the supervisor uses to decide routing and compose answers.</param>
    /// <param name="workers">The worker agents the supervisor may delegate to. Must be non-empty.</param>
    /// <param name="options">Supervisor settings. <c>null</c> uses defaults (including an auto-generated routing prompt).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="coordinator"/> or <paramref name="workers"/> (or any worker) is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="workers"/> is empty.</exception>
    public SupervisorAgent(
        IChatClient<T> coordinator,
        IReadOnlyList<IAgent<T>> workers,
        SupervisorOptions? options = null)
    {
        Guard.NotNull(coordinator);
        Guard.NotNull(workers);
        if (workers.Count == 0)
        {
            throw new ArgumentException("A supervisor requires at least one worker agent.", nameof(workers));
        }

        var settings = options ?? new SupervisorOptions();

        var tools = new ToolCollection();
        foreach (var worker in workers)
        {
            Guard.NotNull(worker);
            tools.Add(new AgentAsTool<T>(worker));
        }

        // string.IsNullOrWhiteSpace lacks the [NotNullWhen(false)] annotation
        // on net471, so the negated form doesn't narrow nullability — keep the
        // pattern match (which does narrow) and only swap the body for the
        // CodeRabbit readability improvement.
        Name = settings.Name is { } name && !string.IsNullOrWhiteSpace(name) ? name : "supervisor";
        Description = settings.Description is { } description && !string.IsNullOrWhiteSpace(description)
            ? description
            : "Coordinates a team of specialized agents to accomplish a task.";

        var systemPrompt = settings.SystemPrompt is { } prompt && !string.IsNullOrWhiteSpace(prompt)
            ? prompt
            : BuildDefaultRoutingPrompt(workers);

        _executor = new AgentExecutor<T>(coordinator, tools, new AgentExecutorOptions
        {
            Name = Name,
            Description = Description,
            SystemPrompt = systemPrompt,
            MaxIterations = settings.MaxIterations,
            Temperature = settings.Temperature,
            ToolChoice = ToolChoiceMode.Auto,
        });
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public string Description { get; }

    /// <summary>
    /// Runs the supervisor against a single user request.
    /// </summary>
    /// <param name="request">The user's goal.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>A task producing the supervisor's <see cref="AgentRunResult"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="request"/> is <c>null</c>.</exception>
    public Task<AgentRunResult> RunAsync(string request, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(request);
        return RunAsync(new[] { ChatMessage.User(request) }, cancellationToken);
    }

    /// <inheritdoc/>
    public Task<AgentRunResult> RunAsync(
        IReadOnlyList<ChatMessage> messages,
        CancellationToken cancellationToken = default) =>
        _executor.RunAsync(messages, cancellationToken);

    private static string BuildDefaultRoutingPrompt(IReadOnlyList<IAgent<T>> workers)
    {
        var builder = new StringBuilder();
        builder.AppendLine(
            "You are a supervisor coordinating a team of specialized agents. Break the user's request into " +
            "steps and delegate each step to the most suitable agent by calling its handoff tool. After an " +
            "agent returns its result, decide whether more delegation is needed. When the task is complete, " +
            "reply directly to the user with the final answer and do not call any further tools.");
        builder.AppendLine();
        builder.AppendLine("Available agents:");
        foreach (var worker in workers)
        {
            builder.Append("- ").Append(worker.Name);
            if (worker.Description.Trim().Length > 0)
            {
                builder.Append(": ").Append(worker.Description.Trim());
            }

            builder.AppendLine();
        }

        return builder.ToString();
    }
}
