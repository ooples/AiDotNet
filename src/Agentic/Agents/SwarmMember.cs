using AiDotNet.Agentic.Models;
using AiDotNet.Agentic.Tools;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// One participant in a <see cref="Swarm{T}"/>: a named persona with its own chat model, instructions,
/// tools, and the set of peers it is allowed to hand control to.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// Unlike a worker under a <see cref="SupervisorAgent{T}"/> (which runs as a self-contained subroutine),
/// a swarm member shares one running conversation with its peers. When a member hands off, the next member
/// continues the <em>same</em> conversation, so context flows directly between peers rather than being
/// summarized through a coordinator.
/// </para>
/// <para><b>For Beginners:</b> Picture a support desk where staff pass a customer between them. Each
/// staff member (a member here) has their own expertise, their own tools, and a list of colleagues they
/// can transfer the customer to. The conversation history travels with the customer.
/// </para>
/// </remarks>
public sealed class SwarmMember<T>
{
    /// <summary>
    /// Initializes a new swarm member.
    /// </summary>
    /// <param name="name">The member's unique name (used for handoff routing). Must be non-empty.</param>
    /// <param name="client">The chat model this member uses while it is active.</param>
    /// <param name="systemPrompt">The member's instructions/persona, applied while it is active. <c>null</c> means none.</param>
    /// <param name="description">A short description of the member's specialty (shown to peers in handoff tools).</param>
    /// <param name="tools">The member's own (non-handoff) tools. <c>null</c> means none.</param>
    /// <param name="handoffs">
    /// The names of peers this member may transfer to. <c>null</c> means "any other member"; an empty list
    /// means the member cannot hand off (it is a terminal responder).
    /// </param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> or <paramref name="client"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="name"/> is empty/whitespace.</exception>
    public SwarmMember(
        string name,
        IChatClient<T> client,
        string? systemPrompt = null,
        string? description = null,
        ToolCollection? tools = null,
        IReadOnlyList<string>? handoffs = null)
    {
        Guard.NotNullOrWhiteSpace(name);
        Guard.NotNull(client);

        // Snapshot + validate the caller's handoffs list so a mutable list
        // can't change behind Swarm's back after construction-time validation.
        IReadOnlyList<string>? handoffSnapshot = null;
        if (handoffs is not null)
        {
            var copy = new string[handoffs.Count];
            for (var i = 0; i < handoffs.Count; i++)
            {
                Guard.NotNullOrWhiteSpace(handoffs[i]);
                copy[i] = handoffs[i];
            }
            handoffSnapshot = copy;
        }

        Name = name;
        Client = client;
        SystemPrompt = systemPrompt;
        Description = description ?? string.Empty;
        Tools = tools ?? new ToolCollection();
        Handoffs = handoffSnapshot;
    }

    /// <summary>Gets the member's unique name.</summary>
    public string Name { get; }

    /// <summary>Gets a short description of the member's specialty.</summary>
    public string Description { get; }

    /// <summary>Gets the chat model the member uses while it is the active responder.</summary>
    public IChatClient<T> Client { get; }

    /// <summary>Gets the member's instructions/persona, applied while it is active, or <c>null</c>.</summary>
    public string? SystemPrompt { get; }

    /// <summary>Gets the member's own (non-handoff) tools.</summary>
    public ToolCollection Tools { get; }

    /// <summary>
    /// Gets the peers this member may hand off to: <c>null</c> means "any other member", an empty list
    /// means "no handoffs".
    /// </summary>
    public IReadOnlyList<string>? Handoffs { get; }
}
