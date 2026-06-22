namespace AiDotNet.Agentic.Models;

/// <summary>
/// Selects the shape the model's output must take (free text, arbitrary JSON, or schema-constrained JSON).
/// </summary>
/// <remarks>
/// <para>
/// Structured output is what makes model responses safe to parse programmatically. When
/// <see cref="JsonSchema"/> is requested, the accompanying JSON schema is enforced — by the provider
/// for cloud models, or by constrained decoding for the local in-process engine — so the result is
/// guaranteed to deserialize.
/// </para>
/// <para><b>For Beginners:</b> By default a model replies with ordinary prose, which is hard for code
/// to read reliably. These options let you ask for machine-readable output instead:
/// - <b>Text</b>: normal human-readable text (the default).
/// - <b>Json</b>: "reply with valid JSON" (shape not guaranteed).
/// - <b>JsonSchema</b>: "reply with JSON that matches exactly this structure" (shape guaranteed).
/// </para>
/// </remarks>
public enum ChatResponseFormatKind
{
    /// <summary>
    /// Ordinary free-form text. This is the default.
    /// </summary>
    Text,

    /// <summary>
    /// Syntactically valid JSON, but with no guarantee about which fields are present (often called
    /// "JSON mode").
    /// </summary>
    Json,

    /// <summary>
    /// JSON constrained to a supplied JSON schema, so the output is guaranteed to deserialize into the
    /// expected type.
    /// </summary>
    JsonSchema
}
