namespace AiDotNet.Agentic.Memory;

/// <summary>
/// A memory paired with its relevance score for a particular query, as returned by
/// <see cref="IAgentMemoryStore.SearchAsync"/>.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When the assistant searches its notes, each matching note comes back with a
/// number saying how well it matched (higher = more relevant). This pairs the note with that number.
/// </para>
/// </remarks>
public sealed class ScoredMemory
{
    /// <summary>
    /// Initializes a new scored memory.
    /// </summary>
    /// <param name="memory">The matched memory.</param>
    /// <param name="score">The relevance score (higher is more relevant).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="memory"/> is <c>null</c>.</exception>
    public ScoredMemory(AgentMemory memory, double score)
    {
        Guard.NotNull(memory);
        Memory = memory;
        Score = score;
    }

    /// <summary>Gets the matched memory.</summary>
    public AgentMemory Memory { get; }

    /// <summary>Gets the relevance score (higher is more relevant).</summary>
    public double Score { get; }
}
