namespace AiDotNet.Configuration;

/// <summary>
/// Process-global control for GPU backend diagnostic output visibility.
/// </summary>
/// <remarks>
/// <para>
/// AiDotNet's GPU backends (OpenCL, HIP, CUDA) emit status messages during device
/// discovery, kernel compilation, and availability checks. Historically these were
/// always written to <see cref="System.Console.WriteLine"/>, producing ~40 lines of
/// output on every <c>AiModelBuilder.BuildAsync()</c>. This clutters the console for
/// applications using rich terminal UI (Spectre.Console), batch processing, or
/// structured logging.
/// </para>
/// <para>
/// Since github.com/ooples/AiDotNet#1122, applications can suppress the output
/// discoverably via this class. Use <see cref="Verbose"/> to read or toggle the
/// setting. The value is forwarded to <see cref="AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput"/>
/// which already existed in the Tensors package but lived on a low-level backend
/// class with no discoverability from AiDotNet application code.
/// </para>
/// <para><b>For Beginners:</b> If your AI application was printing lots of messages like
/// <c>[OpenClBackend] Compiling kernels...</c>, you can suppress them by setting
/// <see cref="Verbose"/> to <c>false</c>. If something isn't working and you want
/// to see what the GPU backend is doing, set it to <c>true</c>.</para>
/// </remarks>
/// <example>
/// <code>
/// // Suppress GPU diagnostics for a quiet terminal.
/// AiDotNet.Configuration.GpuDiagnosticsConfig.Verbose = false;
///
/// // Restore verbose output for troubleshooting.
/// AiDotNet.Configuration.GpuDiagnosticsConfig.Verbose = true;
///
/// // Read the current setting.
/// bool isVerbose = AiDotNet.Configuration.GpuDiagnosticsConfig.Verbose;
/// </code>
/// </example>
public static class GpuDiagnosticsConfig
{
    /// <summary>
    /// Whether GPU backends emit verbose diagnostic output (device discovery,
    /// kernel compilation progress, availability checks).
    /// </summary>
    /// <remarks>
    /// Forwards to <see cref="AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput"/>.
    /// The initial value is read once from the <c>AIDOTNET_GPU_VERBOSE</c>
    /// environment variable at process start: <c>false</c>/<c>0</c>/<c>no</c>/<c>off</c>
    /// → quiet, anything else (including unset) → verbose. Programmatic
    /// assignment overrides the env-var value for the lifetime of the process.
    /// </remarks>
    public static bool Verbose
    {
        get => AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput;
        set => AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput = value;
    }
}
