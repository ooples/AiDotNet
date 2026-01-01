namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Compatibility wrapper that keeps the GpuEngine API while routing work through DirectGpuTensorEngine.
/// </summary>
public sealed class GpuEngine : DirectGpuTensorEngine
{
    private readonly AdaptiveThresholds _thresholds;

    public GpuEngine()
        : this(AdaptiveThresholds.Default, enableTimingDiagnostics: false)
    {
    }

    public GpuEngine(AdaptiveThresholds thresholds, bool enableTimingDiagnostics = false)
        : base()
    {
        _thresholds = thresholds ?? AdaptiveThresholds.Default;
    }

    public AdaptiveThresholds Thresholds => _thresholds;
}
