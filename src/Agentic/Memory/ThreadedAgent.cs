using AiDotNet.Agentic.Agents;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Memory;

/// <summary>
/// Wraps any <see cref="IAgent{T}"/> with conversation memory: each run is tied to a thread id, so prior
/// turns are loaded from an <see cref="IConversationStore"/> and prepended to the new input, and the new
/// user/assistant turn is persisted afterwards. This is how a stateless agent becomes a stateful chat.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// The thread persists a clean user/assistant dialogue: each call appends the user message and the agent's
/// final answer. Intermediate tool calls and the system prompt remain internal to the inner agent's run and
/// are available on the returned <see cref="AgentRunResult"/> for that turn, but are not stored as part of
/// the durable conversation.
/// </para>
/// <para><b>For Beginners:</b> A plain agent forgets everything between questions. Wrap it in a
/// <see cref="ThreadedAgent{T}"/> with a thread id (like a chat-session id) and it remembers: it reads the
/// past conversation, answers with that context, and writes the new exchange back for next time.
/// </para>
/// </remarks>
public sealed class ThreadedAgent<T>
{
    private readonly IAgent<T> _inner;
    private readonly IConversationStore _store;

    /// <summary>
    /// Initializes a new threaded wrapper.
    /// </summary>
    /// <param name="inner">The agent to give conversation memory to.</param>
    /// <param name="store">The store that persists each thread's history.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> or <paramref name="store"/> is <c>null</c>.</exception>
    public ThreadedAgent(IAgent<T> inner, IConversationStore store)
    {
        Guard.NotNull(inner);
        Guard.NotNull(store);
        _inner = inner;
        _store = store;
    }

    /// <summary>Gets the wrapped agent's name.</summary>
    public string Name => _inner.Name;

    /// <summary>Gets the wrapped agent's description.</summary>
    public string Description => _inner.Description;

    /// <summary>
    /// Continues a thread with a single user message: loads the thread's history, runs the inner agent with
    /// that history plus the new message, then appends the user message and the agent's answer to the thread.
    /// </summary>
    /// <param name="threadId">The conversation thread id. Must be non-empty.</param>
    /// <param name="userMessage">The new user message.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>A task producing the inner agent's <see cref="AgentRunResult"/> for this turn.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="userMessage"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="threadId"/> is empty/whitespace.</exception>
    public Task<AgentRunResult> RunAsync(string threadId, string userMessage, CancellationToken cancellationToken = default)
    {
        Guard.NotNull(userMessage);
        return RunAsync(threadId, new[] { ChatMessage.User(userMessage) }, cancellationToken);
    }

    /// <summary>
    /// Continues a thread with one or more new messages.
    /// </summary>
    /// <param name="threadId">The conversation thread id. Must be non-empty.</param>
    /// <param name="newMessages">The new messages to add to the conversation. Must be non-empty.</param>
    /// <param name="cancellationToken">Token used to cancel the run.</param>
    /// <returns>A task producing the inner agent's <see cref="AgentRunResult"/> for this turn.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="newMessages"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="threadId"/> is empty/whitespace, <paramref name="newMessages"/> is empty,
    /// or any new message is not a <see cref="ChatRole.User"/> turn.
    /// </exception>
    public async Task<AgentRunResult> RunAsync(
        string threadId,
        IReadOnlyList<ChatMessage> newMessages,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNullOrWhiteSpace(threadId);
        Guard.NotNull(newMessages);
        if (newMessages.Count == 0)
        {
            throw new ArgumentException("At least one new message is required.", nameof(newMessages));
        }

        // Validate individual elements at the boundary — mirrors the pattern
        // in InMemoryConversationStore.AppendAsync so a null message fails
        // here (with a clear message naming this method) rather than later
        // inside the store with less context. The durable thread is documented
        // as the clean user/assistant dialogue, so inbound system/tool/
        // assistant messages are rejected up front: persisting them verbatim
        // would replay them on every future turn and corrupt the thread.
        foreach (var message in newMessages)
        {
            Guard.NotNull(message);
            if (message.Role != ChatRole.User)
            {
                throw new ArgumentException(
                    $"Thread messages must be user turns; got a {message.Role} message. System prompts and " +
                    "tool exchanges belong to the inner agent's run, not the durable conversation.",
                    nameof(newMessages));
            }
        }

        var history = await _store.GetAsync(threadId, cancellationToken).ConfigureAwait(false);

        var conversation = new List<ChatMessage>(history.Count + newMessages.Count);
        conversation.AddRange(history);
        conversation.AddRange(newMessages);

        var result = await _inner.RunAsync(conversation, cancellationToken).ConfigureAwait(false);

        var turn = new List<ChatMessage>(newMessages.Count + 1);
        turn.AddRange(newMessages);
        turn.Add(ChatMessage.Assistant(result.FinalText));
        await _store.AppendAsync(threadId, turn, cancellationToken).ConfigureAwait(false);

        return result;
    }
}
