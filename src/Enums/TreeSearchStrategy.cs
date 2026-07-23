namespace AiDotNet.Enums;

/// <summary>
/// Tree search strategies for exploring the reasoning space in Tree-of-Thoughts.
/// </summary>
public enum TreeSearchStrategy
{
    /// <summary>
    /// Explores all nodes at each depth level before going deeper.
    /// Good for comprehensive shallow exploration.
    /// </summary>
    BreadthFirst,

    /// <summary>
    /// Explores one branch fully before backtracking.
    /// Good for deep reasoning along specific paths.
    /// </summary>
    DepthFirst,

    /// <summary>
    /// Always explores the highest-scored node next.
    /// Good for efficient exploration of promising paths.
    /// </summary>
    BestFirst
}
