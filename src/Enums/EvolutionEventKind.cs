namespace AiDotNet.Enums;

/// <summary>Classifies observer notifications emitted by the engine.</summary>
public enum EvolutionEventKind
{
    /// <summary>A candidate was proposed and assigned an identity.</summary>
    Proposed = 0,
    /// <summary>An evaluation reached a terminal status.</summary>
    Evaluated = 1,
    /// <summary>An archive accepted or replaced an elite.</summary>
    ArchiveChanged = 2,
    /// <summary>Migration copied elites between islands.</summary>
    Migrated = 3,
    /// <summary>A checkpoint was durably published.</summary>
    Checkpointed = 4,
    /// <summary>The run stopped.</summary>
    Stopped = 5
}
