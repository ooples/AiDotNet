using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Models;

/// <summary>
/// Per-request settings for a chat call: sampling controls, tool availability, and output format.
/// </summary>
/// <remarks>
/// <para>
/// Every property is nullable. <c>null</c> means "use the provider's (or AiDotNet's) sensible default"
/// rather than forcing callers to specify everything. This follows the library-wide options pattern:
/// zero-config by default, fully overridable when needed. Connectors apply documented defaults when a
/// value is <c>null</c> (for example, temperature ≈ 0.7).
/// </para>
/// <para><b>For Beginners:</b> Think of this as the knobs on the request. Leave a knob untouched
/// (<c>null</c>) and a reasonable default is used. Turn it to change behavior:
/// - <see cref="Temperature"/>: higher = more creative/random, lower = more focused.
/// - <see cref="MaxOutputTokens"/>: cap on reply length.
/// - <see cref="Tools"/> / <see cref="ToolChoice"/>: which tools the model may call, and how eagerly.
/// - <see cref="ResponseFormat"/>: ask for plain text or machine-readable JSON.
/// </para>
/// </remarks>
public sealed class ChatOptions
{
    /// <summary>
    /// Gets or sets the sampling temperature (typically 0.0–2.0). Higher is more random.
    /// <c>null</c> uses the connector default (≈ 0.7).
    /// </summary>
    public double? Temperature { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of tokens to generate. <c>null</c> uses the connector default.
    /// </summary>
    public int? MaxOutputTokens { get; set; }

    /// <summary>
    /// Gets or sets nucleus-sampling probability mass (0.0–1.0). <c>null</c> uses the connector default.
    /// </summary>
    public double? TopP { get; set; }

    /// <summary>
    /// Gets or sets top-K sampling. <c>null</c> (or 0) disables it / uses the connector default.
    /// </summary>
    public int? TopK { get; set; }

    /// <summary>
    /// Gets or sets sequences that, when generated, cause the model to stop. <c>null</c> means none.
    /// </summary>
    public IReadOnlyList<string>? StopSequences { get; set; }

    /// <summary>
    /// Gets or sets a deterministic sampling seed where the provider supports it. <c>null</c> means unset.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets the tools the model is allowed to call this turn. <c>null</c> or empty means no tools.
    /// </summary>
    public IReadOnlyList<AiToolDefinition>? Tools { get; set; }

    /// <summary>
    /// Gets or sets how the model may use the supplied <see cref="Tools"/>. <c>null</c> is treated as
    /// <see cref="ToolChoiceMode.Auto"/>.
    /// </summary>
    public ToolChoiceMode? ToolChoice { get; set; }

    /// <summary>
    /// Gets or sets the specific tool the model must call. Only meaningful when <see cref="ToolChoice"/>
    /// is <see cref="ToolChoiceMode.Required"/>. <c>null</c> means "any tool".
    /// </summary>
    public string? RequiredToolName { get; set; }

    /// <summary>
    /// Gets or sets the desired output format. <c>null</c> is treated as <see cref="ChatResponseFormatKind.Text"/>.
    /// </summary>
    public ChatResponseFormatKind? ResponseFormat { get; set; }

    /// <summary>
    /// Gets or sets the JSON schema enforced when <see cref="ResponseFormat"/> is
    /// <see cref="ChatResponseFormatKind.JsonSchema"/>. Ignored for other formats.
    /// </summary>
    public JObject? ResponseJsonSchema { get; set; }
}
