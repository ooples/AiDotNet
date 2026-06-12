using AiDotNet.Agentic.Models;
using Newtonsoft.Json.Linq;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// A peer-to-peer multi-agent team where control transfers between members over one shared conversation.
/// Unlike a <see cref="SupervisorAgent{T}"/> (a hub that runs workers as subroutines), a swarm has no
/// central coordinator: whichever member is active answers directly, and may hand the whole conversation
/// to a peer, who then continues from the same history.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// Handoffs are expressed as native tool calls (<c>transfer_to_&lt;peer&gt;</c>). The swarm intercepts these
/// at the loop level: rather than executing them, it switches the active member and re-runs the turn with
/// the new member's instructions and tools against the unchanged conversation. A member's own (non-handoff)
/// tools are executed normally. The whole team shares one <see cref="SwarmOptions.MaxIterations"/> budget,
/// which guarantees termination even if two members would otherwise ping-pong.
/// </para>
/// <para><b>For Beginners:</b> Imagine specialists passing a customer between them. The customer (the
/// conversation) stays the same; whoever is currently helping can either answer or say "let me transfer you
/// to my colleague". This class runs that back-and-forth and gives you the final answer plus which member
/// gave it.
/// </para>
/// </remarks>
public sealed class Swarm<T> : IAgent<T>
{
    private readonly Dictionary<string, SwarmMember<T>> _members;
    private readonly string _entryMemberName;
    private readonly SwarmOptions _options;

    /// <summary>
    /// Initializes a new swarm.
    /// </summary>
    /// <param name="members">The team members. Must be non-empty with unique names.</param>
    /// <param name="entryMemberName">The name of the member that handles the conversation first.</param>
    /// <param name="options">Swarm settings. <c>null</c> uses defaults.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="members"/> (or any member) or <paramref name="entryMemberName"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when members is empty, names are not unique, the entry member is unknown, or a member declares a
    /// handoff to an unknown peer.
    /// </exception>
    public Swarm(IReadOnlyList<SwarmMember<T>> members, string entryMemberName, SwarmOptions? options = null)
    {
        Guard.NotNull(members);
        Guard.NotNullOrWhiteSpace(entryMemberName);
        if (members.Count == 0)
        {
            throw new ArgumentException("A swarm requires at least one member.", nameof(members));
        }

        _members = new Dictionary<string, SwarmMember<T>>(StringComparer.Ordinal);
        foreach (var member in members)
        {
            Guard.NotNull(member);
            if (_members.ContainsKey(member.Name))
            {
                throw new ArgumentException($"Duplicate swarm member name '{member.Name}'.", nameof(members));
            }

            _members.Add(member.Name, member);
        }

        if (!_members.ContainsKey(entryMemberName))
        {
            throw new ArgumentException(
                $"Entry member '{entryMemberName}' is not one of the swarm members.", nameof(entryMemberName));
        }

        foreach (var member in members)
        {
            if (member.Handoffs is null)
            {
                continue;
            }

            foreach (var peer in member.Handoffs)
            {
                if (!_members.ContainsKey(peer))
                {
                    throw new ArgumentException(
                        $"Member '{member.Name}' declares a handoff to unknown peer '{peer}'.", nameof(members));
                }
            }
        }

        _entryMemberName = entryMemberName;
        _options = options ?? new SwarmOptions();

        // Pattern-match form so the compiler narrows nullability on net471
        // (string.IsNullOrWhiteSpace lacks [NotNullWhen(false)] there).
        Name = _options.Name is { } name && !string.IsNullOrWhiteSpace(name)
            ? name
            : "swarm";
        Description = _options.Description is { } description && !string.IsNullOrWhiteSpace(description)
            ? description
            : "A peer-to-peer team of agents that transfer control over a shared conversation.";
    }

    /// <inheritdoc/>
    public string Name { get; }

    /// <inheritdoc/>
    public string Description { get; }

    /// <summary>
    /// Runs the swarm against a single user request, starting with the entry member.
    /// </summary>
    /// <param name="request">The user's request.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>A task producing the swarm's <see cref="AgentRunResult"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="request"/> is <c>null</c>.</exception>
    public Task<AgentRunResult> RunAsync(string request, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(request);
        return RunAsync(new[] { ChatMessage.User(request) }, cancellationToken);
    }

    /// <inheritdoc/>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="messages"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="messages"/> is empty.</exception>
    public async Task<AgentRunResult> RunAsync(
        IReadOnlyList<ChatMessage> messages,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(messages);
        if (messages.Count == 0)
        {
            throw new ArgumentException("The conversation must contain at least one message.", nameof(messages));
        }

        // The single shared conversation, excluding the per-turn system prompt (which reflects whichever
        // member is currently active and is prepended fresh each turn).
        var shared = new List<ChatMessage>(messages.Count + 8);
        foreach (var message in messages)
        {
            if (message.Role != ChatRole.System)
            {
                shared.Add(message);
            }
        }

        var maxIterations = _options.MaxIterations is { } configured && configured > 0
            ? configured
            : AgentExecutorOptions.DefaultMaxIterations;

        var active = _members[_entryMemberName];
        var inputTokens = 0;
        var outputTokens = 0;
        var sawUsage = false;

        for (var iteration = 1; iteration <= maxIterations; iteration++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var handoffTargets = BuildHandoffMap(active);
            var requestOptions = BuildRequestOptions(active, handoffTargets);
            var requestMessages = BuildRequestMessages(active, shared);

            var response = await active.Client.GetResponseAsync(requestMessages, requestOptions, cancellationToken)
                .ConfigureAwait(false);
            shared.Add(response.Message);

            if (response.Usage is { } usage)
            {
                sawUsage = true;
                inputTokens += usage.InputTokens;
                outputTokens += usage.OutputTokens;
            }

            var toolCalls = response.Message.ToolCalls;
            var wantsTools = response.FinishReason == ChatFinishReason.ToolCalls || toolCalls.Count > 0;

            if (wantsTools && toolCalls.Count > 0)
            {
                string? nextMemberName = null;
                foreach (var call in toolCalls)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    if (handoffTargets.TryGetValue(call.ToolName, out var peer))
                    {
                        if (nextMemberName is null)
                        {
                            nextMemberName = peer;
                            shared.Add(ChatMessage.Tool(call.CallId, $"Control transferred to '{peer}'."));
                        }
                        else
                        {
                            shared.Add(ChatMessage.Tool(
                                call.CallId,
                                $"Ignored: already transferring control to '{nextMemberName}' this turn.",
                                isError: true));
                        }
                    }
                    else if (active.Tools.Contains(call.ToolName))
                    {
                        var toolMessage = await active.Tools.InvokeToToolMessageAsync(call, cancellationToken)
                            .ConfigureAwait(false);
                        shared.Add(toolMessage);
                    }
                    else
                    {
                        shared.Add(ChatMessage.Tool(
                            call.CallId, $"Unknown tool '{call.ToolName}'.", isError: true));
                    }
                }

                if (nextMemberName is { } target)
                {
                    active = _members[target];
                }

                continue;
            }

            return AgentRunResult.Finished(
                response.Message.Text,
                shared,
                iteration,
                sawUsage ? new ChatUsage(inputTokens, outputTokens) : null,
                active.Name);
        }

        var lastText = shared.Count > 0 ? shared[shared.Count - 1].Text : string.Empty;
        return AgentRunResult.Stopped(
            lastText,
            shared,
            maxIterations,
            sawUsage ? new ChatUsage(inputTokens, outputTokens) : null,
            active.Name);
    }

    private Dictionary<string, string> BuildHandoffMap(SwarmMember<T> active)
    {
        // Use Ordinal everywhere so the map comparer matches the equality
        // semantics used to skip `active` (line 244). The previous mix
        // (map = OrdinalIgnoreCase, peer-skip = Ordinal) could keep a
        // case-variant duplicate of the active agent under a tool key.
        var map = new Dictionary<string, string>(StringComparer.Ordinal);
        // Materialise the fallback only when Handoffs is null. Iterating
        // _members.Keys directly avoids the per-call LINQ + ToList()
        // allocation for the common "no explicit handoff list" case; the
        // peer-skip inside the loop takes care of dropping the active member.
        var allowed = (IEnumerable<string>?)active.Handoffs ?? _members.Keys;
        foreach (var peer in allowed)
        {
            if (string.Equals(peer, active.Name, StringComparison.Ordinal))
            {
                continue;
            }

            map[ToolNaming.HandoffToolName(peer)] = peer;
        }

        return map;
    }

    private ChatOptions BuildRequestOptions(SwarmMember<T> active, Dictionary<string, string> handoffTargets)
    {
        var definitions = new List<AiToolDefinition>();
        definitions.AddRange(active.Tools.GetDefinitions());
        foreach (var entry in handoffTargets)
        {
            var peer = _members[entry.Value];
            var description = peer.Description.Trim().Length > 0
                ? $"Transfer the conversation to the '{peer.Name}' agent. {peer.Description.Trim()}"
                : $"Transfer the conversation to the '{peer.Name}' agent.";
            definitions.Add(new AiToolDefinition(entry.Key, description, CreateHandoffSchema()));
        }

        return new ChatOptions
        {
            Temperature = _options.Temperature,
            Tools = definitions.Count > 0 ? definitions : null,
            ToolChoice = definitions.Count > 0 ? ToolChoiceMode.Auto : null,
        };
    }

    private static List<ChatMessage> BuildRequestMessages(SwarmMember<T> active, List<ChatMessage> shared)
    {
        var requestMessages = new List<ChatMessage>(shared.Count + 1);
        if (active.SystemPrompt is { } prompt && prompt.Trim().Length > 0)
        {
            requestMessages.Add(ChatMessage.System(prompt));
        }

        requestMessages.AddRange(shared);
        return requestMessages;
    }

    private static JObject CreateHandoffSchema() =>
        new() { ["type"] = "object", ["properties"] = new JObject() };
}
