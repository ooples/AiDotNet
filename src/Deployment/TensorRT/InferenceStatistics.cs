namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// Statistics for TensorRT inference engine.
/// </summary>
public class InferenceStatistics
{
    public int NumStreams { get; set; }
    public int AvailableStreams { get; set; }
    public int ActiveStreams { get; set; }
}
