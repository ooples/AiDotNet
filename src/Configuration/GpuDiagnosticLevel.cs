namespace AiDotNet.Configuration;

/// <summary>
/// Verbosity level for GPU backend diagnostic output.
/// </summary>
/// <remarks>
/// <para>
/// Addresses github.com/ooples/AiDotNet#1122. AiDotNet's GPU backends
/// (OpenCL, HIP, CUDA) emit status messages during device discovery,
/// kernel compilation, and availability checks. Historically these were
/// always written to <see cref="System.Console.WriteLine"/>, producing
/// ~40 lines of output on every <c>AiModelBuilder.BuildAsync()</c>. This
/// enum gives applications fine-grained control over that output.
/// </para>
/// <para><b>For Beginners:</b> Think of this like the log level in any
/// logging framework: <see cref="Silent"/> is OFF, <see cref="Minimal"/>
/// is "just the important stuff", <see cref="Verbose"/> is "tell me
/// everything".
/// </para>
/// </remarks>
public enum GpuDiagnosticLevel
{
    /// <summary>
    /// No GPU backend output written to Console or any sink. Use for
    /// clean-TUI applications (Spectre.Console) or batch jobs where
    /// diagnostic noise obscures results.
    /// </summary>
    Silent = 0,

    /// <summary>
    /// Only critical GPU backend output — device selected, compilation
    /// failures, OpenCL DLL-not-found errors. Suitable for production
    /// applications that want a one-line "GPU initialized" signal but
    /// not the full 40-line init dump.
    /// </summary>
    /// <remarks>
    /// Minimal-level filtering is fully honored only when the underlying
    /// AiDotNet.Tensors package supports per-message level tagging. On
    /// v0.38.0 and earlier, <see cref="Minimal"/> behaves as
    /// <see cref="Silent"/> (any non-Verbose value suppresses all output
    /// at the Tensors layer). The <see cref="GpuDiagnosticSink"/> sink
    /// delegate, when registered, receives level-tagged messages once
    /// the Tensors package honors them.
    /// </remarks>
    Minimal = 1,

    /// <summary>
    /// All GPU backend diagnostic output — device discovery, every kernel
    /// compilation step, OpenCL platform queries, GEMM tuning progress.
    /// Use when troubleshooting GPU driver/OpenCL issues or writing up
    /// a hardware-specific bug report.
    /// </summary>
    Verbose = 2,
}
