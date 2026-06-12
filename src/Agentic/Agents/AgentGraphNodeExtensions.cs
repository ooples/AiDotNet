using AiDotNet.Agentic.Graph;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// Bridges the agents layer and the graph runtime: adds an <see cref="IAgent{T}"/> as a node in a
/// <see cref="StateGraph{TState}"/>. The node maps the graph state to the agent's input, runs the agent, and
/// folds the result back into the state — so agents (executors, supervisors, swarms) become first-class steps
/// in a typed, checkpointable, resumable graph.
/// </summary>
/// <remarks>
/// <para>
/// This realizes the unified "typed deterministic graph + multi-agent" story: a graph can route between
/// agent nodes with conditional edges, cycles, human-in-the-loop interrupts, and durable checkpointing, while
/// each node is a full agent (which may itself coordinate a team).
/// </para>
/// <para><b>For Beginners:</b> The graph is a flowchart with state flowing through it; these helpers let one
/// box in the flowchart <em>be</em> an agent. You say how to read the agent's question out of the state and
/// how to write its answer back, and the graph handles the rest (routing, retries, saving progress).
/// </para>
/// </remarks>
public static class AgentGraphNodeExtensions
{
    /// <summary>
    /// Adds a graph node that runs an agent against a single user message derived from the state.
    /// </summary>
    /// <typeparam name="T">The numeric type of the agent.</typeparam>
    /// <typeparam name="TState">The graph state type.</typeparam>
    /// <param name="graph">The graph to add the node to.</param>
    /// <param name="name">The node name.</param>
    /// <param name="agent">The agent to run.</param>
    /// <param name="inputSelector">Reads the agent's user message from the state.</param>
    /// <param name="applyResult">Folds the agent's result back into a new state.</param>
    /// <returns>The graph, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when any argument is <c>null</c>.</exception>
    public static StateGraph<TState> AddAgentNode<T, TState>(
        this StateGraph<TState> graph,
        string name,
        IAgent<T> agent,
        Func<TState, string> inputSelector,
        Func<TState, AgentRunResult, TState> applyResult)
    {
        Guard.NotNull(graph);
        Guard.NotNull(agent);
        Guard.NotNull(inputSelector);
        Guard.NotNull(applyResult);

        return graph.AddNode(name, async (state, cancellationToken) =>
        {
            var input = inputSelector(state);
            // Fail at this boundary with the selector named, rather than deep
            // inside ChatMessage.User / the agent with no hint of the cause.
            if (string.IsNullOrWhiteSpace(input))
            {
                throw new InvalidOperationException(
                    $"The input selector for agent node '{name}' returned a null/empty user message.");
            }

            var result = await agent.RunAsync(new[] { ChatMessage.User(input) }, cancellationToken).ConfigureAwait(false);
            return applyResult(state, result);
        });
    }

    /// <summary>
    /// Adds a graph node that runs an agent against a full message list derived from the state (for richer
    /// control than a single user message).
    /// </summary>
    /// <typeparam name="T">The numeric type of the agent.</typeparam>
    /// <typeparam name="TState">The graph state type.</typeparam>
    /// <param name="graph">The graph to add the node to.</param>
    /// <param name="name">The node name.</param>
    /// <param name="agent">The agent to run.</param>
    /// <param name="messagesSelector">Reads the conversation to send from the state.</param>
    /// <param name="applyResult">Folds the agent's result back into a new state.</param>
    /// <returns>The graph, for chaining.</returns>
    /// <exception cref="ArgumentNullException">Thrown when any argument is <c>null</c>.</exception>
    public static StateGraph<TState> AddAgentNode<T, TState>(
        this StateGraph<TState> graph,
        string name,
        IAgent<T> agent,
        Func<TState, IReadOnlyList<ChatMessage>> messagesSelector,
        Func<TState, AgentRunResult, TState> applyResult)
    {
        Guard.NotNull(graph);
        Guard.NotNull(agent);
        Guard.NotNull(messagesSelector);
        Guard.NotNull(applyResult);

        return graph.AddNode(name, async (state, cancellationToken) =>
        {
            var messages = messagesSelector(state);
            // Fail at this boundary with the selector named, rather than deep
            // inside the agent with no hint of the cause.
            if (messages is null || messages.Count == 0)
            {
                throw new InvalidOperationException(
                    $"The messages selector for agent node '{name}' returned a null/empty conversation.");
            }

            var result = await agent.RunAsync(messages, cancellationToken).ConfigureAwait(false);
            return applyResult(state, result);
        });
    }
}
