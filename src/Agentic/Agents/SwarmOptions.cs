namespace AiDotNet.Agentic.Agents;

/// <summary>
/// Settings for a <see cref="Swarm{T}"/>: its identity and the overall step budget shared across all
/// members for a single run.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These are the dials for the whole team-of-peers. The most important is
/// <see cref="MaxIterations"/>, which caps how many total model calls the swarm may make before it must
/// stop — this is the safety net that prevents two agents from handing a task back and forth forever.
/// </para>
/// </remarks>
public sealed class SwarmOptions
{
    /// <summary>
    /// Gets or sets the swarm's name. <c>null</c> or empty falls back to <c>"swarm"</c>.
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Gets or sets the swarm's description. <c>null</c> falls back to a generic description.
    /// </summary>
    public string? Description { get; set; }

    /// <summary>
    /// Gets or sets the maximum total number of model calls across all members in one run. <c>null</c> or a
    /// non-positive value uses <see cref="AgentExecutorOptions.DefaultMaxIterations"/>. When reached before a
    /// final answer, the run returns with <see cref="AgentRunResult.Completed"/> set to <c>false</c>.
    /// </summary>
    public int? MaxIterations { get; set; }

    /// <summary>
    /// Gets or sets the sampling temperature forwarded to each member's model call. <c>null</c> uses the
    /// connector default.
    /// </summary>
    public double? Temperature { get; set; }
}
