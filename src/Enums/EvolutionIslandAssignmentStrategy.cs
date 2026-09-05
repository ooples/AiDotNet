namespace AiDotNet.Enums;

/// <summary>Selects how the evolution engine decides which island a new proposal belongs to.</summary>
/// <remarks>
/// <para>
/// The engine allocates a sequential evaluation identifier to every proposal and, by default, assigns the proposal
/// to island <c>evaluationId % IslandCount</c>. That keeps the proposal stream perfectly balanced across islands but
/// can move a child away from the island its parent came from whenever the preferred island is still empty and the
/// engine has to borrow a parent from a neighbouring island.
/// <see cref="InheritParent"/> instead assigns the child to the island the parent was actually drawn from, which is
/// the island-isolation rule used by OpenEvolve's program database. Both strategies are fully deterministic and are
/// folded into the engine's configuration hash, so a checkpoint written under one strategy cannot resume under the
/// other.
/// </para>
/// <para><b>For Beginners:</b> An island model splits the search into several semi-independent populations so they
/// can explore different ideas instead of all converging on the same one. When a new candidate is created, someone
/// has to decide which island it joins. <see cref="RoundRobin"/> deals candidates out like cards, one island after
/// another, which keeps the islands the same size. <see cref="InheritParent"/> keeps families together: the child
/// stays on whichever island its parent lives on, which preserves the islands' separate identities more strongly.
/// Start with <see cref="RoundRobin"/> and switch to <see cref="InheritParent"/> if your islands are drifting into
/// each other.</para>
/// </remarks>
public enum EvolutionIslandAssignmentStrategy
{
    /// <summary>Assign island <c>evaluationId % IslandCount</c>, regardless of where the parent came from.</summary>
    RoundRobin = 0,

    /// <summary>Assign the island the parent was drawn from, falling back to the round-robin island for seeds.</summary>
    InheritParent = 1
}
