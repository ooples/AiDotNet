namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// Internal class for building TensorRT engines.
/// </summary>
internal class TensorRTEngineBuilder
{
    public int MaxBatchSize { get; set; }
    public long MaxWorkspaceSize { get; set; }
    public TensorRTPrecision Precision { get; set; } = TensorRTPrecision.FP32;
    public bool StrictTypeConstraints { get; set; }
    public bool EnableDynamicShapes { get; set; }
    public int DeviceId { get; set; }
    public int DLACore { get; set; } = -1;
    public List<OptimizationProfile>? OptimizationProfiles { get; set; }
}
