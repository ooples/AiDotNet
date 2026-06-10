using AiDotNet.Agentic.Models;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Pipeline;

/// <summary>
/// A guardrail <see cref="IChatMiddleware"/> that screens the user input before the model is called and/or the
/// model's response after, using an <see cref="IContentModerator"/>. On a violation it short-circuits with a
/// refusal (finish reason <see cref="ChatFinishReason.ContentFilter"/>) or throws, per configuration.
/// </summary>
/// <remarks>
/// <para>
/// As middleware it composes with any <see cref="IChatClient{T}"/> and stacks with logging/telemetry. Input
/// screening blocks before incurring a model call; output screening protects against unsafe completions even
/// from a benign prompt.
/// </para>
/// <para><b>For Beginners:</b> A bouncer for the model. It can refuse risky requests before they reach the
/// model and catch unsafe answers before they reach the user — returning a polite refusal (or raising an
/// error if you prefer to stop hard).
/// </para>
/// </remarks>
public sealed class ContentSafetyMiddleware : IChatMiddleware
{
    private readonly IContentModerator _moderator;
    private readonly ContentSafetyOptions _options;

    /// <summary>
    /// Initializes a new content-safety middleware.
    /// </summary>
    /// <param name="moderator">The moderator that screens content.</param>
    /// <param name="options">Guardrail settings. <c>null</c> uses defaults (screen both sides, return a refusal).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="moderator"/> is <c>null</c>.</exception>
    public ContentSafetyMiddleware(IContentModerator moderator, ContentSafetyOptions? options = null)
    {
        Guard.NotNull(moderator);
        _moderator = moderator;
        _options = options ?? new ContentSafetyOptions();
    }

    /// <inheritdoc/>
    public async Task<ChatResponse> InvokeAsync(ChatRequestContext context, ChatPipelineDelegate next, CancellationToken cancellationToken)
    {
        Guard.NotNull(context);
        Guard.NotNull(next);

        if (_options.CheckInput)
        {
            var input = LatestUserText(context.Messages);
            if (input is { } text && text.Length > 0)
            {
                var verdict = await _moderator.CheckAsync(text, cancellationToken).ConfigureAwait(false);
                if (!verdict.Allowed)
                {
                    return Blocked(verdict.Reason);
                }
            }
        }

        var response = await next(context, cancellationToken).ConfigureAwait(false);

        if (_options.CheckOutput && response.Text.Length > 0)
        {
            var verdict = await _moderator.CheckAsync(response.Text, cancellationToken).ConfigureAwait(false);
            if (!verdict.Allowed)
            {
                return Blocked(verdict.Reason);
            }
        }

        return response;
    }

    private ChatResponse Blocked(string? reason)
    {
        if (_options.ThrowOnViolation)
        {
            throw new ContentSafetyException(reason is { } r && r.Length > 0 ? r : "Content blocked by safety policy.");
        }

        var refusal = _options.RefusalMessage is { } message && message.Trim().Length > 0
            ? message
            : ContentSafetyOptions.DefaultRefusalMessage;
        return new ChatResponse(ChatMessage.Assistant(refusal), ChatFinishReason.ContentFilter);
    }

    private static string? LatestUserText(IReadOnlyList<ChatMessage> messages)
    {
        for (var i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role == ChatRole.User)
            {
                return messages[i].Text;
            }
        }

        return null;
    }
}
