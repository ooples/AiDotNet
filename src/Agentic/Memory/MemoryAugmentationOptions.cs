namespace AiDotNet.Agentic.Memory;

/// <summary>
/// Settings for <see cref="MemoryAugmentedAgent{T}"/>: how many memories to recall, the minimum relevance to
/// include, and the heading used when injecting them into the conversation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These control how the assistant uses its long-term notes before answering:
/// how many of the most relevant notes to pull in (<see cref="TopK"/>), how relevant a note must be to bother
/// including it (<see cref="MinScore"/>), and the little title that introduces them.
/// </para>
/// </remarks>
public sealed class MemoryAugmentationOptions
{
    /// <summary>The default number of memories to recall per turn.</summary>
    public const int DefaultTopK = 5;

    /// <summary>The default heading prepended to recalled memories.</summary>
    public const string DefaultHeader = "Relevant information recalled from long-term memory:";

    /// <summary>
    /// Gets or sets the maximum number of memories to recall and inject. <c>null</c> or a non-positive value
    /// uses <see cref="DefaultTopK"/>.
    /// </summary>
    public int? TopK { get; set; }

    /// <summary>
    /// Gets or sets the minimum relevance score a memory must reach to be injected. <c>null</c> means include
    /// every returned match (the store already limits to the top matches).
    /// </summary>
    public double? MinScore { get; set; }

    /// <summary>
    /// Gets or sets the heading prepended to the recalled memories. <c>null</c> or whitespace uses
    /// <see cref="DefaultHeader"/>.
    /// </summary>
    public string? Header { get; set; }
}
