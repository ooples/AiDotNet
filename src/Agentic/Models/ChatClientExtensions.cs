namespace AiDotNet.Agentic.Models;

/// <summary>
/// Convenience helpers over <see cref="IChatClient{T}"/> for the common "prompt in, text out" case.
/// </summary>
/// <remarks>
/// <para>
/// The agentic model layer is message-based, but plenty of callers (and internal consumers such as the
/// reasoning strategies) just want to send a prompt and read back text. These extensions wrap a single
/// user message and return the assistant's concatenated text, so simple call sites stay simple while the
/// full message/tool/streaming API remains available when needed.
/// </para>
/// <para><b>For Beginners:</b> Instead of constructing a list of messages and reading a response object,
/// you can write <c>await client.GenerateTextAsync("Summarize this")</c> and get a plain string back.
/// </para>
/// </remarks>
public static class ChatClientExtensions
{
    /// <summary>
    /// Sends a single user prompt and returns the assistant's text reply.
    /// </summary>
    /// <typeparam name="T">The client's numeric type.</typeparam>
    /// <param name="client">The chat client.</param>
    /// <param name="prompt">The user prompt.</param>
    /// <param name="options">Optional per-call settings.</param>
    /// <param name="cancellationToken">Token used to cancel the request.</param>
    /// <returns>The assistant's reply text.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="client"/> or <paramref name="prompt"/> is <c>null</c>.</exception>
    public static async Task<string> GenerateTextAsync<T>(
        this IChatClient<T> client,
        string prompt,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(client);
        Guard.NotNull(prompt);

        var response = await client
            .GetResponseAsync(new[] { ChatMessage.User(prompt) }, options, cancellationToken)
            .ConfigureAwait(false);
        return response.Text;
    }

    /// <summary>
    /// Sends a system instruction plus a user prompt and returns the assistant's text reply.
    /// </summary>
    /// <typeparam name="T">The client's numeric type.</typeparam>
    /// <param name="client">The chat client.</param>
    /// <param name="systemPrompt">The system instructions.</param>
    /// <param name="userPrompt">The user prompt.</param>
    /// <param name="options">Optional per-call settings.</param>
    /// <param name="cancellationToken">Token used to cancel the request.</param>
    /// <returns>The assistant's reply text.</returns>
    /// <exception cref="ArgumentNullException">Thrown when an argument is <c>null</c>.</exception>
    public static async Task<string> GenerateTextAsync<T>(
        this IChatClient<T> client,
        string systemPrompt,
        string userPrompt,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(client);
        Guard.NotNull(systemPrompt);
        Guard.NotNull(userPrompt);

        var messages = new[] { ChatMessage.System(systemPrompt), ChatMessage.User(userPrompt) };
        var response = await client.GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
        return response.Text;
    }
}
