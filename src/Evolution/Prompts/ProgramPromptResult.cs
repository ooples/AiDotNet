using AiDotNet.Agentic.Models;
using AiDotNet.Enums;
using AiDotNet.Validation;

// AiDotNet.PromptEngineering.Templates is imported project-wide and also declares a ChatMessage type.
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Evolution.Prompts;

/// <summary>One rendered prompt: the messages to send, plus the record of how they were assembled.</summary>
/// <remarks>
/// <para>
/// The result carries more than the messages because reproducing a run means reproducing the decisions, not just
/// the text. <see cref="Mode"/> records whether this prompt asked for edits or a rewrite,
/// <see cref="UserTemplateKey"/> records which wording was used, and <see cref="VariationChoices"/> records which
/// alternative phrasing was drawn from the random stream. Persist those alongside the candidate and a later
/// investigation can explain why one proposal differed from another, which is not possible upstream where the
/// variation is drawn from the process-global generator and never recorded.
/// </para>
/// <para>
/// <see cref="WasTruncated"/> is a first-class signal rather than a silent event. A prompt that lost its tail to
/// the character ceiling may have dropped the very example that mattered, and the caller deserves to know.
/// </para>
/// <para><b>For Beginners:</b> This is the finished message ready to send to the model, together with a short
/// receipt of how it was built: which wording was picked, whether the model was asked for a patch or a whole new
/// file, and whether anything had to be cut to fit the size limit. Send <see cref="Messages"/> to your chat
/// client; keep the rest if you want to be able to explain later what the model was shown.</para>
/// </remarks>
public sealed class ProgramPromptResult
{
    /// <summary>Initializes a rendered prompt.</summary>
    /// <param name="systemText">The system message text.</param>
    /// <param name="userText">The user message text.</param>
    /// <param name="mode">Whether the prompt asked for edits or a full rewrite.</param>
    /// <param name="userTemplateKey">The template the user message was rendered from.</param>
    /// <param name="variationChoices">The alternative wording chosen for each variation placeholder.</param>
    /// <param name="wasTruncated">Whether the user message was cut to fit the character ceiling.</param>
    /// <exception cref="ArgumentNullException"><paramref name="systemText"/>, <paramref name="userText"/>, or <paramref name="variationChoices"/> is <c>null</c>.</exception>
    public ProgramPromptResult(
        string systemText,
        string userText,
        ProgramPromptEvolutionMode mode,
        ProgramPromptTemplateKey userTemplateKey,
        IReadOnlyDictionary<string, string> variationChoices,
        bool wasTruncated)
    {
        Guard.NotNull(systemText);
        Guard.NotNull(userText);
        Guard.NotNull(variationChoices);

        SystemText = systemText;
        UserText = userText;
        Mode = mode;
        UserTemplateKey = userTemplateKey;
        VariationChoices = variationChoices;
        WasTruncated = wasTruncated;

        var messages = new List<ChatMessage>(2);
        if (systemText.Length > 0) messages.Add(ChatMessage.System(systemText));
        messages.Add(ChatMessage.User(userText));
        Messages = messages;
    }

    /// <summary>Gets the messages to send, a system message (when non-empty) followed by the user message.</summary>
    public IReadOnlyList<ChatMessage> Messages { get; }

    /// <summary>Gets the system message text.</summary>
    public string SystemText { get; }

    /// <summary>Gets the user message text.</summary>
    public string UserText { get; }

    /// <summary>Gets whether this prompt asked for edit blocks or a complete replacement program.</summary>
    public ProgramPromptEvolutionMode Mode { get; }

    /// <summary>Gets the template the user message was rendered from.</summary>
    public ProgramPromptTemplateKey UserTemplateKey { get; }

    /// <summary>Gets the alternative wording drawn for each variation placeholder; empty when none applied.</summary>
    public IReadOnlyDictionary<string, string> VariationChoices { get; }

    /// <summary>Gets whether the user message was cut to fit the configured character ceiling.</summary>
    public bool WasTruncated { get; }

    /// <summary>Gets the combined character length of both messages.</summary>
    public int TotalLength => SystemText.Length + UserText.Length;

    /// <summary>Returns a description that never echoes prompt text.</summary>
    /// <returns>The mode, template, and total length.</returns>
    public override string ToString() =>
        $"ProgramPromptResult({Mode}, {UserTemplateKey}, {TotalLength} chars, truncated={WasTruncated})";
}
