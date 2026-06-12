using System.Security.Cryptography;
using System.Text;
using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// Builds a stable, canonical key for a chat request (its messages plus the request settings that affect the
/// response) so identical requests map to the same recorded interaction.
/// </summary>
/// <remarks>
/// <para>
/// Every variable-length field is length-prefixed, which makes the encoding injective: message text that
/// happens to contain the encoding's separator characters cannot collide with a different request. Tool
/// definitions are fingerprinted in full (name, description, and parameter schema), so two tools that share a
/// name but differ in behavior produce different keys.
/// </para>
/// <para><b>For Beginners:</b> A fingerprint of a request. Two requests with the same conversation and the
/// same settings produce the same fingerprint, which is how record/replay knows it has already seen this
/// exact call.
/// </para>
/// </remarks>
internal static class ChatInteractionKey
{
    /// <summary>
    /// Computes the canonical key for a request.
    /// </summary>
    /// <param name="messages">The conversation.</param>
    /// <param name="options">The request options, or <c>null</c>.</param>
    /// <param name="modelId">
    /// The model identity the recording belongs to, or <c>null</c> for a model-agnostic key. Including it
    /// keeps recordings isolated when one store is shared across clients for different models.
    /// </param>
    /// <returns>A deterministic key string.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="messages"/> is <c>null</c>.</exception>
    public static string For(IReadOnlyList<ChatMessage> messages, ChatOptions? options, string? modelId = null)
    {
        Guard.NotNull(messages);

        var builder = new StringBuilder();
        AppendField(builder, "model", modelId ?? string.Empty);
        builder.Append("msgs:").Append(messages.Count).Append(';');
        foreach (var message in messages)
        {
            AppendField(builder, "r", message.Role.ToString());
            AppendField(builder, "t", message.Text);
        }

        builder.Append("opt;");
        if (options is not null)
        {
            AppendField(builder, "temp", Format(options.Temperature));
            AppendField(builder, "p", Format(options.TopP));
            AppendField(builder, "k", options.TopK?.ToString() ?? "_");
            AppendField(builder, "seed", options.Seed?.ToString() ?? "_");
            AppendField(builder, "max", options.MaxOutputTokens?.ToString() ?? "_");
            AppendField(builder, "fmt", options.ResponseFormat?.ToString() ?? "_");
            if (options.Tools is { Count: > 0 } tools)
            {
                // Fingerprint the FULL definition of every tool, not just its
                // name: two different tools with the same name must not replay
                // each other's recordings. The fingerprint is hashed to keep
                // key size bounded regardless of schema size.
                var toolBuilder = new StringBuilder();
                foreach (var tool in tools)
                {
                    AppendField(toolBuilder, "n", tool.Name);
                    AppendField(toolBuilder, "d", tool.Description);
                    AppendField(toolBuilder, "s", tool.ParametersSchema.ToString(Newtonsoft.Json.Formatting.None));
                }

                AppendField(builder, "tools", HashHex(toolBuilder.ToString()));
            }
        }

        return builder.ToString();
    }

    // Length-prefixed field: "<tag>:<charCount>:<value>;". The explicit length
    // makes the overall encoding unambiguous even when the value contains
    // ':' / ';' or any other character.
    private static void AppendField(StringBuilder builder, string tag, string value) =>
        builder.Append(tag).Append(':').Append(value.Length).Append(':').Append(value).Append(';');

    private static string HashHex(string value)
    {
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(value));
        var hex = new StringBuilder(hash.Length * 2);
        foreach (var b in hash)
        {
            hex.Append(b.ToString("x2"));
        }

        return hex.ToString();
    }

    private static string Format(double? value) =>
        value?.ToString("R", System.Globalization.CultureInfo.InvariantCulture) ?? "_";
}
