namespace AiDotNet.Inference.SpeculativeDecoding;

/// <summary>
/// Statistics for a tree speculation step.
/// </summary>
internal class TreeStepStatistics
{
    /// <summary>Number of nodes in tree.</summary>
    public int TreeNodes { get; set; }

    /// <summary>Number of paths explored.</summary>
    public int PathsExplored { get; set; }

    /// <summary>Length of best accepted path.</summary>
    public int BestPathLength { get; set; }
}
