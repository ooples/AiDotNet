using AiDotNet.Agentic.Models;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// Settings for an <see cref="AgentExecutor{T}"/>: identity, system prompt, the tool-loop budget, and the
/// sampling knobs forwarded to each model call.
/// </summary>
/// <remarks>
/// <para>
/// Every value is nullable and defaults to a sensible behavior when left <c>null</c>, following the
/// library-wide options pattern (zero-config by default, fully overridable). The executor applies the
/// documented defaults internally, so the common case is <c>new AgentExecutor&lt;float&gt;(client)</c>.
/// </para>
/// <para><b>For Beginners:</b> These are the dials for one agent. The most useful ones are
/// <see cref="SystemPrompt"/> (the agent's standing instructions / persona) and <see cref="MaxIterations"/>
/// (how many think→use-tool→think rounds it may take before it must answer).
/// </para>
/// </remarks>
public sealed class AgentExecutorOptions
{
    /// <summary>
    /// The default maximum number of model calls in a single run when <see cref="MaxIterations"/> is unset.
    /// </summary>
    public const int DefaultMaxIterations = 10;

    /// <summary>
    /// Gets or sets the agent's name (used by coordinators and when the agent is surfaced as a tool).
    /// <c>null</c> or empty falls back to <c>"agent"</c>.
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Gets or sets the agent's one-line specialty description. <c>null</c> falls back to an empty string.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the system prompt prepended to every run (the agent's standing instructions/persona).
    /// <c>null</c> or whitespace means no system message is added.
    /// </summary>
    public string? SystemPrompt { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of model calls in one run (each tool round costs one call).
    /// <c>null</c> or a non-positive value uses <see cref="DefaultMaxIterations"/>. When the cap is reached
    /// before a final answer, the run returns with <see cref="AgentRunResult.Completed"/> set to <c>false</c>.
    /// </summary>
    public int? MaxIterations { get; set; }

    /// <summary>
    /// Gets or sets the sampling temperature forwarded to each model call. <c>null</c> uses the connector default.
    /// </summary>
    public double? Temperature { get; set; }

    /// <summary>
    /// Gets or sets the per-call output-token cap forwarded to each model call. <c>null</c> uses the connector default.
    /// </summary>
    public int? MaxOutputTokens { get; set; }

    /// <summary>
    /// Gets or sets how eagerly the model may use the agent's tools when tools are present. <c>null</c> is
    /// treated as <see cref="ToolChoiceMode.Auto"/>. Ignored when the agent has no tools.
    /// </summary>
    public ToolChoiceMode? ToolChoice { get; set; }
}
