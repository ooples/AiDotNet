namespace AiDotNet.Enums;

/// <summary>Describes how completed worker results are committed to evolution state.</summary>
public enum EvolutionExecutionMode
{
    /// <summary>Commit in evaluation-ID order so worker timing cannot change the result.</summary>
    Deterministic = 0,
    /// <summary>Commit as workers finish; this can improve responsiveness but is schedule-dependent.</summary>
    Opportunistic = 1
}
