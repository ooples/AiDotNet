namespace AiDotNet.Deployment.Edge;

/// <summary>
/// Represents a model partitioned for cloud+edge deployment.
/// </summary>
public class PartitionedModel
{
    /// <summary>Gets or sets the original model.</summary>
    public object? OriginalModel { get; set; }

    /// <summary>Gets or sets the model part for edge execution.</summary>
    public object? EdgeModel { get; set; }

    /// <summary>Gets or sets the model part for cloud execution.</summary>
    public object? CloudModel { get; set; }

    /// <summary>Gets or sets the partition strategy used.</summary>
    public PartitionStrategy PartitionStrategy { get; set; }

    /// <summary>Gets or sets the intermediate tensor shape between edge and cloud.</summary>
    public int[]? IntermediateShape { get; set; }
}
