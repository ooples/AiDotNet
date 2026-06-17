using AiDotNet.Agentic.Models;
using Newtonsoft.Json;

// Disambiguate from the legacy-era AiDotNet.PromptEngineering.Templates.ChatMessage that is imported
// project-wide via a global using; the agentic subsystem uses the Models type.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// Helpers that turn a chat model's reply into a strongly-typed .NET object by constraining the model
/// to a JSON schema derived from the target type and deserializing the result.
/// </summary>
/// <remarks>
/// <para>
/// These extensions close the loop between <see cref="JsonSchemaGenerator"/> (C# type → JSON Schema) and
/// the model's structured-output mode: the schema for <typeparamref name="TResult"/> is attached to the
/// request as <see cref="ChatResponseFormatKind.JsonSchema"/>, and the JSON reply is deserialized back
/// into <typeparamref name="TResult"/>. Providers that enforce the schema (and the local engine's
/// constrained decoding) guarantee the reply parses.
/// </para>
/// <para><b>For Beginners:</b> Instead of getting a blob of text and parsing it yourself, you ask the
/// model for a specific shape — e.g. <c>await client.GetStructuredResponseAsync&lt;double, Weather&gt;("Weather in
/// Paris?")</c> — and get back a filled-in <c>Weather</c> object. The library writes the schema, sets the
/// "reply as JSON matching this shape" flag, and deserializes for you.
/// </para>
/// </remarks>
public static class StructuredOutputExtensions
{
    private static readonly JsonSerializerSettings JsonSettings = new()
    {
        NullValueHandling = NullValueHandling.Ignore,
        MissingMemberHandling = MissingMemberHandling.Ignore
    };

    /// <summary>
    /// Sends a single user prompt and deserializes the schema-constrained JSON reply into
    /// <typeparamref name="TResult"/>.
    /// </summary>
    /// <typeparam name="T">The client's numeric type.</typeparam>
    /// <typeparam name="TResult">The type to deserialize the reply into.</typeparam>
    /// <param name="client">The chat client.</param>
    /// <param name="prompt">The user prompt.</param>
    /// <param name="options">Optional per-call settings; the response format is overridden to the schema.</param>
    /// <param name="cancellationToken">Token used to cancel the request.</param>
    /// <returns>The deserialized result.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="client"/> or <paramref name="prompt"/> is <c>null</c>.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the reply could not be deserialized into <typeparamref name="TResult"/>.</exception>
    public static Task<TResult> GetStructuredResponseAsync<T, TResult>(
        this IChatClient<T> client,
        string prompt,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(prompt);
        return client.GetStructuredResponseAsync<T, TResult>(
            new[] { ChatMessage.User(prompt) }, options, cancellationToken);
    }

    /// <summary>
    /// Sends a conversation and deserializes the schema-constrained JSON reply into
    /// <typeparamref name="TResult"/>.
    /// </summary>
    /// <typeparam name="T">The client's numeric type.</typeparam>
    /// <typeparam name="TResult">The type to deserialize the reply into.</typeparam>
    /// <param name="client">The chat client.</param>
    /// <param name="messages">The conversation so far.</param>
    /// <param name="options">Optional per-call settings; the response format is overridden to the schema.</param>
    /// <param name="cancellationToken">Token used to cancel the request.</param>
    /// <returns>The deserialized result.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="client"/> or <paramref name="messages"/> is <c>null</c>.</exception>
    /// <exception cref="InvalidOperationException">Thrown when the reply could not be deserialized into <typeparamref name="TResult"/>.</exception>
    public static async Task<TResult> GetStructuredResponseAsync<T, TResult>(
        this IChatClient<T> client,
        IReadOnlyList<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        Guard.NotNull(client);
        Guard.NotNull(messages);

        var effective = Clone(options);
        effective.ResponseFormat = ChatResponseFormatKind.JsonSchema;
        effective.ResponseJsonSchema = JsonSchemaGenerator.ForType(typeof(TResult));

        var response = await client.GetResponseAsync(messages, effective, cancellationToken).ConfigureAwait(false);

        TResult? result;
        try
        {
            result = JsonConvert.DeserializeObject<TResult>(response.Text, JsonSettings);
        }
        catch (JsonException ex)
        {
            throw new InvalidOperationException(
                $"The model reply could not be deserialized into {typeof(TResult).Name}: {ex.Message}", ex);
        }

        if (result is null)
        {
            throw new InvalidOperationException(
                $"The model reply deserialized to null for {typeof(TResult).Name}. Reply was: {response.Text}");
        }

        return result;
    }

    private static ChatOptions Clone(ChatOptions? options)
    {
        if (options is null)
        {
            return new ChatOptions();
        }

        return new ChatOptions
        {
            Temperature = options.Temperature,
            MaxOutputTokens = options.MaxOutputTokens,
            TopP = options.TopP,
            TopK = options.TopK,
            StopSequences = options.StopSequences,
            Seed = options.Seed,
            Tools = options.Tools,
            ToolChoice = options.ToolChoice,
            RequiredToolName = options.RequiredToolName,
            ResponseFormat = options.ResponseFormat,
            ResponseJsonSchema = options.ResponseJsonSchema
        };
    }
}
