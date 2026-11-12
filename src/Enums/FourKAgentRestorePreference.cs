namespace AiDotNet.Enums;

/// <summary>
/// 4KAgent's restoration preference (the profile's <c>RestorePerference</c>), which decides the super-resolution experts
/// the agent may choose from.
/// </summary>
public enum FourKAgentRestorePreference
{
    /// <summary>Fidelity-oriented experts, trained on pixel losses (the ExpSR-*-F profiles).</summary>
    Fidelity,

    /// <summary>Perception-oriented experts, the real-world restoration models (the Gen4K-P and *-P profiles).</summary>
    Perception,

    /// <summary>No preference: every expert in the toolbox competes (a profile without <c>RestorePerference</c>).</summary>
    Balanced,
}
