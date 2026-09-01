namespace AiDotNet.Enums;

/// <summary>Reports the result of attempting to add a completed evaluation to an archive.</summary>
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
