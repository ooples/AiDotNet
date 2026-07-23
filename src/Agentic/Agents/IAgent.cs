using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage, which is in scope
// project-wide via a global using in AiModelBuilder.cs. The agentic subsystem uses the Models type.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// A named, runnable agent: given a conversation, it produces a final answer (after optionally using
/// tools and/or delegating to other agents along the way).
/// </summary>
/// <typeparam name="T">
/// The numeric type shared with the underlying <see cref="IChatClient{T}"/> (e.g., <see cref="float"/>
/// or <see cref="double"/>), so an agent and the model it drives agree on one tensor element type.
/// </typeparam>
/// <remarks>
/// <para>
/// This is the single abstraction the multi-agent layer composes. A leaf agent (<see cref="AgentExecutor{T}"/>)
/// drives a chat model in a tool-calling loop; a coordinator (supervisor/swarm) is itself an
/// <see cref="IAgent{T}"/> that routes work to other <see cref="IAgent{T}"/> members. Because the contract is
/// uniform, agents nest: a supervisor can supervise supervisors, and any agent can be exposed to another as a
/// callable tool.
/// </para>
/// <para><b>For Beginners:</b> An agent is a worker with a <see cref="Name"/> and a one-line
/// <see cref="Description"/> of what it's good at. You hand it the conversation so far and it hands back a
/// result. Whether it solved the task alone or quietly asked teammates for help is its own business — callers
/// only see the final answer.
/// </para>
/// </remarks>
public interface IAgent<T>
{
    /// <summary>
    /// Gets the unique, stable name other agents and routers use to reference this agent.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets a short natural-language description of the agent's specialty, used by coordinators (and by
    /// models, when the agent is surfaced as a tool) to decide when to route work here.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Runs the agent over the supplied conversation and returns its result.
    /// </summary>
    /// <param name="messages">The ordered conversation so far. Must be non-null and non-empty.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>A task producing the agent's <see cref="AgentRunResult"/>.</returns>
    Task<AgentRunResult> RunAsync(
        IReadOnlyList<ChatMessage> messages,
        CancellationToken cancellationToken = default);
}
