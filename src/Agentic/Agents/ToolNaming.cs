using System.Text;

namespace AiDotNet.Agentic.Agents;

/// <summary>
/// Shared helpers for turning agent names into valid, stable tool names used by handoffs (both the
/// supervisor's <see cref="AgentAsTool{T}"/> and the swarm's control-transfer tools).
/// </summary>
internal static class ToolNaming
{
    /// <summary>The conventional prefix for a handoff/transfer tool.</summary>
    public const string HandoffPrefix = "transfer_to_";

    /// <summary>
    /// Maps an arbitrary agent name to the characters tool names allow (<c>[A-Za-z0-9_-]</c>), replacing
    /// anything else with <c>_</c> and trimming stray underscores.
    /// </summary>
    public static string Sanitize(string name)
    {
        var builder = new StringBuilder(name.Length);
        foreach (var ch in name)
        {
            builder.Append(char.IsLetterOrDigit(ch) || ch is '_' or '-' ? ch : '_');
        }

        var sanitized = builder.ToString().Trim('_');
        return sanitized.Length == 0 ? "agent" : sanitized;
    }

    /// <summary>Builds the handoff tool name for transferring to the named agent.</summary>
    public static string HandoffToolName(string agentName) => HandoffPrefix + Sanitize(agentName);
}
