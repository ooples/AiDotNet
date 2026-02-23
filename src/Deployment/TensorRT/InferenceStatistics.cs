namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// Statistics for TensorRT inference engine.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> InferenceStatistics provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public class InferenceStatistics
{
    public int NumStreams { get; set; }
    public int AvailableStreams { get; set; }
    public int ActiveStreams { get; set; }
}
