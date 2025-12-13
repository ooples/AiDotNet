namespace AiDotNet.Deployment.TensorRT;

/// <summary>
/// Internal class for building TensorRT engines.
/// </summary>
internal class TensorRTEngineBuilder
{
    public int MaxBatchSize { get; set; }
    public long MaxWorkspaceSize { get; set; }
    public bool UseFp16 { get; set; }
    public bool UseInt8 { get; set; }
    public bool StrictTypeConstraints { get; set; }
    public bool EnableDynamicShapes { get; set; }
    public int DeviceId { get; set; }
    public int? DlaCore { get; set; }
    public List<OptimizationProfile>? OptimizationProfiles { get; set; }
}
