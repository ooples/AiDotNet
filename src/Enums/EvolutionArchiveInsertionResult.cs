namespace AiDotNet.Enums;

/// <summary>Reports the result of attempting to add a completed evaluation to an archive.</summary>
/// <remarks>
/// <para>
/// <see cref="AiDotNet.Evolution.MapElitesArchive{TGenome}.TryAdd"/> returns one of these values for every completed
/// evaluation the engine offers it. The archive changes state, and increments its version, only for
/// <see cref="Inserted"/>, <see cref="Replaced"/>, and <see cref="InsertedWithEviction"/>; the other two values leave
/// it untouched. <see cref="Rejected"/> signals a contract failure such as an evaluation that did not complete, a
/// mismatched optimization direction, or a descriptor that is missing or outside its allowed range, whereas
/// <see cref="NotImproved"/> is the normal outcome of a valid candidate that lost to the incumbent in its cell.
/// </para>
/// <para>
/// Eviction only occurs in capacity-limited archives: when every allowed cell is occupied and the candidate maps to
/// a new cell, the archive removes its deterministic worst elite to make room, provided the candidate is better than
/// that worst elite; otherwise the candidate is reported as <see cref="NotImproved"/>.
/// </para>
/// <para><b>For Beginners:</b> Each time the search finishes evaluating a candidate it asks the archive to keep it,
/// and this enum is the archive's answer. Picture a leaderboard with one slot per category: the candidate either
/// fills an empty slot (<see cref="Inserted"/>), beats the current holder of its slot (<see cref="Replaced"/>), loses
/// to that holder (<see cref="NotImproved"/>), is disqualified because its paperwork is wrong (<see cref="Rejected"/>),
/// or, when the board is full, bumps the weakest holder off another slot to make room
/// (<see cref="InsertedWithEviction"/>). Observers receive this value through
/// <see cref="AiDotNet.Evolution.EvolutionEvent{TGenome}.InsertionResult"/>, which makes it easy to count how often
/// the search is still finding improvements.</para>
/// </remarks>
public enum EvolutionArchiveInsertionResult
{
    /// <summary>The candidate filled an empty cell.</summary>
    Inserted = 0,
    /// <summary>The candidate replaced the incumbent in its cell.</summary>
    Replaced = 1,
    /// <summary>The candidate was not better than the incumbent.</summary>
    NotImproved = 2,
    /// <summary>The candidate or one of its descriptors was invalid for this archive.</summary>
    Rejected = 3,
    /// <summary>The candidate was inserted and a deterministic capacity eviction occurred.</summary>
    InsertedWithEviction = 4
}
