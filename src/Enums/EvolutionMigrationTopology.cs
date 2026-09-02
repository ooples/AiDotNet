namespace AiDotNet.Enums;

/// <summary>Selects which islands a migration round copies elites between.</summary>
/// <remarks>
/// <para>
/// An island model keeps several sub-populations apart so that each explores its own region, and periodically copies a
/// few elites between them. The topology is the directed graph those copies follow, and it controls how fast a strong
/// lineage spreads: a one-directional <see cref="Ring"/> needs up to <c>n - 1</c> rounds to reach every island,
/// <see cref="BidirectionalRing"/> halves that, <see cref="Star"/> funnels everything through island 0 in two hops, and
/// <see cref="FullyConnected"/> reaches everyone in a single round at the cost of the fastest loss of diversity.
/// </para>
/// <para>
/// The value is validated when <c>EvolutionEngineOptions</c> is snapshotted and folded into the engine's configuration
/// hash, so a checkpoint written under one topology refuses to resume under another. OpenEvolve hard-codes the
/// bidirectional ring instead (<c>database.py</c> <c>migrate_programs</c> sends every migrant to both
/// <c>(i + 1) % n</c> and <c>(i - 1) % n</c>) and offers no way to choose a different one.
/// </para>
/// <para><b>For Beginners:</b> Imagine several research teams in separate rooms, each working on the same problem.
/// Every so often a team passes copies of its best results to other teams. This setting decides who passes results to
/// whom: around a circle in one direction (<see cref="Ring"/>), around the circle both ways
/// (<see cref="BidirectionalRing"/>), through one central team that everyone reports to and hears from
/// (<see cref="Star"/>), or to everybody at once (<see cref="FullyConnected"/>). Sharing more widely spreads good ideas
/// faster but makes all the rooms look alike sooner, which is exactly what islands are meant to avoid. Start with
/// <see cref="Ring"/>, which is the default, and move to a denser topology only if the islands are converging too
/// slowly.</para>
/// <para>
/// Topology choice in island models is analysed in Whitley, Rana, and Heckendorn, "The Island Model Genetic Algorithm:
/// On Separability, Population Size and Convergence" (Journal of Computing and Information Technology, 1999).
/// </para>
/// </remarks>
public enum EvolutionMigrationTopology
{
    /// <summary>Each island sends to the next island only, so island <c>i</c> feeds island <c>(i + 1) mod n</c>.</summary>
    Ring = 0,

    /// <summary>Each island sends to both neighbours, <c>(i + 1) mod n</c> and <c>(i - 1 + n) mod n</c>.</summary>
    /// <remarks>With exactly two islands both neighbours are the same island, so each source emits one transfer.</remarks>
    BidirectionalRing = 1,

    /// <summary>Island 0 is a hub: it sends to every other island, and every other island sends to it.</summary>
    Star = 2,

    /// <summary>Every island sends to every other island.</summary>
    FullyConnected = 3
}
