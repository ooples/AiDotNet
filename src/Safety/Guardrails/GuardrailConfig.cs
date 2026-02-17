using AiDotNet.Enums;

namespace AiDotNet.Safety.Guardrails;

/// <summary>
/// Configuration for guardrail modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure input/output guardrails including
/// length limits, topic restrictions, and custom rules.
/// </para>
/// </remarks>
public class GuardrailConfig
{
    /// <summary>Maximum allowed input length in characters. Default: 100000.</summary>
    public int? MaxInputLength { get; set; }

    /// <summary>Maximum allowed output length in characters. Default: 100000.</summary>
    public int? MaxOutputLength { get; set; }

    /// <summary>Restricted topics to block. Null = no topic restrictions.</summary>
    public string[]? TopicRestrictions { get; set; }

    /// <summary>Default action for guardrail violations. Default: Block.</summary>
    public SafetyAction? DefaultAction { get; set; }

    internal int EffectiveMaxInputLength => MaxInputLength ?? 100000;
    internal int EffectiveMaxOutputLength => MaxOutputLength ?? 100000;
    internal string[] EffectiveTopicRestrictions => TopicRestrictions ?? Array.Empty<string>();
    internal SafetyAction EffectiveDefaultAction => DefaultAction ?? SafetyAction.Block;
}
