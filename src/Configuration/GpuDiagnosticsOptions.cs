namespace AiDotNet.Configuration;

/// <summary>
/// Options for controlling GPU backend diagnostic output visibility.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet's GPU backends (OpenCL, HIP, CUDA) emit status messages during
/// device discovery, kernel compilation, and availability checks. Historically
/// these were always written to <see cref="System.Console.WriteLine"/>,
/// producing ~40 lines of output on every <c>AiModelBuilder.BuildAsync()</c>.
/// This clutters the console for applications using rich terminal UI
/// (Spectre.Console), batch processing, or structured logging.
/// </para>
/// <para>
/// Since github.com/ooples/AiDotNet#1122, output is suppressed by default.
/// Applications that want the diagnostic dump opt in either via this options
/// class (passed to <c>AiModelBuilder.ConfigureGpuDiagnostics(...)</c>), via
/// <see cref="GpuDiagnosticsConfig.Verbose"/> directly, or via the
/// <c>AIDOTNET_GPU_VERBOSE</c> environment variable.
/// </para>
/// <para><b>For Beginners:</b> If your AI application was printing lots of
/// <c>[OpenClBackend] Compiling kernels...</c> messages, that's now hidden by
/// default. If you want to see them (for troubleshooting), set
/// <see cref="Verbose"/> to <c>true</c>.</para>
/// </remarks>
public class GpuDiagnosticsOptions
{
    /// <summary>
    /// Whether GPU backends emit verbose diagnostic output. When <c>null</c>,
    /// the existing <see cref="GpuDiagnosticsConfig.Verbose"/> value (set by
    /// env var or prior programmatic assignment) is preserved. Default:
    /// <c>false</c>.
    /// </summary>
    /// <remarks>
    /// Nullable to match the AiDotNet config pattern — <c>null</c> means
    /// "don't change the current setting" rather than "set to false", so
    /// passing an empty options instance is a no-op.
    /// </remarks>
    public bool? Verbose { get; set; }
}
