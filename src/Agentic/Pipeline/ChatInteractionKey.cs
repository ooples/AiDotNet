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
/// <para><b>For Beginners:</b> A fingerprint of a request. Two requests with the same conversation and the
/// same settings produce the same fingerprint, which is how record/replay knows it has already seen this
/// exact call.
/// </para>
/// </remarks>
public static class ChatInteractionKey
{
    /// <summary>
    /// Computes the canonical key for a request.
    /// </summary>
    /// <param name="messages">The conversation.</param>
    /// <param name="options">The request options, or <c>null</c>.</param>
    /// <returns>A deterministic key string.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="messages"/> is <c>null</c>.</exception>
    public static string For(IReadOnlyList<ChatMessage> messages, ChatOptions? options)
    {
        Guard.NotNull(messages);

        var builder = new StringBuilder();
        foreach (var message in messages)
        {
            builder.Append(message.Role).Append('\x1f').Append(message.Text).Append('\x1e');
        }

        builder.Append("opt\x1f");
        if (options is not null)
        {
            builder.Append("t=").Append(Format(options.Temperature)).Append(';');
            builder.Append("p=").Append(Format(options.TopP)).Append(';');
            builder.Append("k=").Append(options.TopK?.ToString() ?? "_").Append(';');
            builder.Append("seed=").Append(options.Seed?.ToString() ?? "_").Append(';');
            builder.Append("max=").Append(options.MaxOutputTokens?.ToString() ?? "_").Append(';');
            builder.Append("fmt=").Append(options.ResponseFormat?.ToString() ?? "_").Append(';');
            if (options.Tools is { Count: > 0 } tools)
            {
                builder.Append("tools=");
                foreach (var tool in tools)
                {
                    builder.Append(tool.Name).Append(',');
                }
            }
        }

        return builder.ToString();
    }

    private static string Format(double? value) =>
        value?.ToString("R", System.Globalization.CultureInfo.InvariantCulture) ?? "_";
}
