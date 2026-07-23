namespace AiDotNet.Agentic.Agents;

/// <summary>
/// Settings for a <see cref="SupervisorAgent{T}"/>: its identity, an optional override of the routing
/// system prompt, and the coordinator's loop/sampling budget.
/// </summary>
/// <remarks>
/// <para>
/// Every value is nullable and falls back to a sensible default. In particular, leaving
/// <see cref="SystemPrompt"/> <c>null</c> lets the supervisor auto-generate a routing prompt that lists
/// its workers and their specialties — the zero-config path.
/// </para>
/// <para><b>For Beginners:</b> These are the dials for the "team lead". The one you'll most often touch is
/// <see cref="SystemPrompt"/> if you want to change how the lead decides who does what; otherwise the
/// defaults give you a working team out of the box.
/// </para>
/// </remarks>
public sealed class SupervisorOptions
{
    /// <summary>
    /// Gets or sets the supervisor's name. <c>null</c> or empty falls back to <c>"supervisor"</c>.
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Gets or sets the supervisor's description. <c>null</c> falls back to a generic team-lead description.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets an explicit routing system prompt. <c>null</c> or whitespace auto-generates one that
    /// lists the worker agents and their descriptions.
    /// </summary>
    public string? SystemPrompt { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of coordinator model calls in one run (each handoff round is one
    /// call). <c>null</c> or a non-positive value uses <see cref="AgentExecutorOptions.DefaultMaxIterations"/>.
    /// </summary>
    public int? MaxIterations { get; set; }

    /// <summary>
    /// Gets or sets the sampling temperature for the coordinator. <c>null</c> uses the connector default.
    /// </summary>
    public double? Temperature { get; set; }
}
