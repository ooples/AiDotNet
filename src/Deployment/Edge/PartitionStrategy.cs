namespace AiDotNet.Deployment.Edge;

/// <summary>
/// Strategies for partitioning models between cloud and edge.
/// </summary>
public enum PartitionStrategy
{
    /// <summary>Execute early layers on edge, rest on cloud</summary>
    EarlyLayers,

    /// <summary>Execute most layers on edge, only final on cloud</summary>
    LateLayers,

    /// <summary>Balanced partition</summary>
    Balanced,

    /// <summary>Adaptively determine partition based on runtime conditions</summary>
    Adaptive,

    /// <summary>Manual partition specification</summary>
    Manual
}
