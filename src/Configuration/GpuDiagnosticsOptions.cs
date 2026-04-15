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
/// Since github.com/ooples/AiDotNet#1122, applications can suppress the output
/// discoverably via this options class or
/// <see cref="GpuDiagnosticsConfig.Verbose"/>. The default behavior depends
/// on the underlying AiDotNet.Tensors package version — as of v0.38.0 the
/// default is verbose (on), and applications must opt out. A follow-up
/// AiDotNet.Tensors release will flip the default to quiet; this options
/// class is forward-compatible with that change.
/// </para>
/// <para>Three ways to suppress:</para>
/// <list type="bullet">
/// <item><c>AiModelBuilder.ConfigureGpuDiagnostics(new() { Verbose = false })</c></item>
/// <item><c>GpuDiagnosticsConfig.Verbose = false</c> (direct)</item>
/// <item><c>AIDOTNET_GPU_VERBOSE=0</c> environment variable</item>
/// </list>
/// <para><b>For Beginners:</b> If your AI application is printing lots of
/// <c>[OpenClBackend] Compiling kernels...</c> messages, pass
/// <c>new GpuDiagnosticsOptions { Verbose = false }</c> to the builder's
/// <c>ConfigureGpuDiagnostics</c> method to hide them.</para>
/// </remarks>
public class GpuDiagnosticsOptions
{
    /// <summary>
    /// Whether GPU backends emit verbose diagnostic output. Default:
    /// <c>null</c> (preserve the existing process-global setting —
    /// <see cref="GpuDiagnosticsConfig.Verbose"/>, set by env var or
    /// prior programmatic assignment). Explicit <c>true</c> / <c>false</c>
    /// values override it.
    /// </summary>
    /// <remarks>
    /// Nullable to match the AiDotNet config pattern — <c>null</c> means
    /// "don't change the current setting" rather than "set to false", so
    /// passing an empty options instance to
    /// <c>AiModelBuilder.ConfigureGpuDiagnostics(...)</c> is a no-op.
    /// </remarks>
    public bool? Verbose { get; set; }
}
