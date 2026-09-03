namespace AiDotNet.Enums;

/// <summary>Says which part of a checkpoint a recovered candidate came from.</summary>
/// <remarks>
/// <para>
/// A checkpoint holds the same candidate in up to three places, and which place it came from changes what its
/// presence means. An <see cref="IslandArchive"/> entry is a current elite: it still owns a cell of its island's
/// map. A <see cref="GlobalElite"/> entry is one of the best across every island, kept in a separate index that
/// survives an island losing the cell. An <see cref="IslandHistory"/> entry is a runner-up that the bounded
/// per-island history retained after it lost its cell.
/// </para>
/// <para><b>For Beginners:</b> When you read back what a finished run saved, this tells you whether a candidate was
/// a current champion of its part of the map, one of the overall best, or a good one that was kept around after
/// being beaten.</para>
/// </remarks>
public enum EvolutionCheckpointEntrySource
{
    /// <summary>A current elite of one island's archive.</summary>
    IslandArchive = 0,

    /// <summary>An entry of the cross-island global elite index.</summary>
    GlobalElite = 1,

    /// <summary>A retained runner-up from one island's bounded history.</summary>
    IslandHistory = 2
}
