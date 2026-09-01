namespace AiDotNet.Enums;

/// <summary>Identifies whether an evaluation came from the task or a deterministic cache.</summary>
public enum EvolutionCacheStatus
{
    /// <summary>No cache lookup was performed.</summary>
    NotChecked = 0,
    /// <summary>The cache did not contain a reusable evaluation.</summary>
    Miss = 1,
    /// <summary>A prior completed evaluation was reused.</summary>
    Hit = 2
}
