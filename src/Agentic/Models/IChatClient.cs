namespace AiDotNet.Agentic.Models;

/// <summary>
/// A message-based chat model client that supports native tool calling, streaming, and structured output.
/// This is the foundation the agentic orchestration subsystem is built on, superseding the legacy
/// text-in/text-out <c>IChatModel&lt;T&gt;</c> / <c>ILanguageModel&lt;T&gt;</c>.
/// </summary>
/// <typeparam name="T">
/// The numeric type used across the AiDotNet ecosystem (e.g., <see cref="float"/> or <see cref="double"/>).
/// Cloud connectors treat it as a marker for ecosystem consistency; the in-process local engine uses it
/// as its tensor element type, so an agent and the model it drives share one numeric type.
/// </typeparam>
/// <remarks>
/// <para>
/// Implementations translate a conversation (a list of <see cref="ChatMessage"/>) plus per-call
/// <see cref="ChatOptions"/> into a provider call, and return either a complete <see cref="ChatResponse"/>
/// or a stream of <see cref="ChatResponseUpdate"/> chunks. The same abstraction covers cloud providers
/// (OpenAI, Anthropic, Azure) and the local engine, so higher layers (tools, graph, agents) are written
/// once and run against any backend.
/// </para>
/// <para><b>For Beginners:</b> This is the "brain" interface every agent talks to. You give it the
/// conversation so far and some settings; it gives you back the next reply — either all at once
/// (<see cref="GetResponseAsync"/>) or piece by piece as it's generated (<see cref="GetStreamingResponseAsync"/>).
/// Because the interface is the same for every provider, you can swap a cloud model for a local one
/// without rewriting your agent.
/// </para>
/// </remarks>
public interface IChatClient<T>
{
    /// <summary>
    /// Gets the identifier of the underlying model (e.g., <c>gpt-4o</c>, <c>claude-3-5-sonnet</c>,
    /// or a local model name).
    /// </summary>
    string ModelId { get; }

    /// <summary>
    /// Sends the conversation and returns the model's complete response.
    /// </summary>
    /// <param name="messages">The ordered conversation so far. Must be non-null and non-empty.</param>
    /// <param name="options">Per-call settings (sampling, tools, output format). <c>null</c> uses defaults.</param>
    /// <param name="cancellationToken">Token used to cancel the request.</param>
    /// <returns>A task producing the complete <see cref="ChatResponse"/>.</returns>
    /// <remarks>
    /// If the response's <see cref="ChatResponse.FinishReason"/> is <see cref="ChatFinishReason.ToolCalls"/>,
    /// the caller is expected to execute the requested tools and continue the conversation with the results.
    /// </remarks>
    Task<ChatResponse> GetResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Sends the conversation and streams the model's response as a sequence of incremental updates.
    /// </summary>
    /// <param name="messages">The ordered conversation so far. Must be non-null and non-empty.</param>
    /// <param name="options">Per-call settings (sampling, tools, output format). <c>null</c> uses defaults.</param>
    /// <param name="cancellationToken">Token used to cancel the stream.</param>
    /// <returns>
    /// An async stream of <see cref="ChatResponseUpdate"/> chunks. Concatenating the text deltas and
    /// accumulating tool-call fragments reconstructs the full reply; the terminal update carries the
    /// finish reason and (where available) usage.
    /// </returns>
    IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default);
}
