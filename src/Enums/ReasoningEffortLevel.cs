namespace AiDotNet.Enums;

/// <summary>How much internal deliberation a reasoning model is asked to spend before it answers.</summary>
/// <remarks>
/// <para>
/// Reasoning models (OpenAI's o-series and GPT-5 families, and the GPT-OSS releases) generate hidden reasoning
/// tokens before the visible answer. The provider exposes one knob for how many of those it is willing to spend,
/// sent on the wire as <c>reasoning_effort</c>. More effort usually means a better answer on hard problems and a
/// slower, more expensive call; less effort is right for mechanical edits where the model already knows what to do.
/// </para>
/// <para>
/// <see cref="Unspecified"/> is not a level: it means "send no <c>reasoning_effort</c> field at all" and let the
/// provider apply its own default. That is the reference OpenEvolve behaviour, which only adds the field when the
/// configuration supplies one, so leaving this value alone reproduces upstream request bodies exactly.
/// </para>
/// <para><b>For Beginners:</b> Some newer AI models "think" privately before replying, and you can tell them how
/// hard to think. <see cref="Low"/> is quick and cheap, <see cref="High"/> is slow and thorough, and
/// <see cref="Unspecified"/> simply does not mention the setting so the provider picks for you. If you are unsure,
/// leave it unspecified and only raise it when the model's answers are not good enough.</para>
/// </remarks>
public enum ReasoningEffortLevel
{
    /// <summary>No effort field is sent; the provider's own default applies. This is the reference behaviour.</summary>
    Unspecified = 0,

    /// <summary>The smallest amount of deliberation the model offers, sent as <c>minimal</c>.</summary>
    Minimal = 1,

    /// <summary>Fast and cheap deliberation, sent as <c>low</c>.</summary>
    Low = 2,

    /// <summary>Balanced deliberation, sent as <c>medium</c>.</summary>
    Medium = 3,

    /// <summary>The most deliberation the model offers, sent as <c>high</c>.</summary>
    High = 4
}
