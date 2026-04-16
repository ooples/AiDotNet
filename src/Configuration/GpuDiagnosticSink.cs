namespace AiDotNet.Configuration;

/// <summary>
/// Delegate that receives GPU backend diagnostic messages in lieu of
/// <see cref="System.Console.WriteLine"/>. Applications register a sink
/// to route GPU diagnostics through their logging framework of choice
/// (Spectre.Console, Serilog, Microsoft.Extensions.Logging, structured
/// logs, etc.).
/// </summary>
/// <param name="level">
/// The severity of the message. When an application's sink only wants
/// to forward warnings and above, it can filter on this parameter.
/// </param>
/// <param name="message">The diagnostic message text.</param>
/// <remarks>
/// <para>
/// Addresses Option C of github.com/ooples/AiDotNet#1122:
/// "Use ILogger instead of Console.WriteLine — Then applications control
/// output through their logging framework."
/// </para>
/// <para>
/// Forward-compatibility: the sink is captured in
/// <see cref="GpuDiagnosticsConfig.Sink"/> immediately. When the underlying
/// AiDotNet.Tensors package supports sink routing (v0.39+), the diagnostic
/// messages are delivered to the sink WITH level tagging. On current
/// Tensors v0.38.0, the sink is stored but the Console.WriteLine calls
/// in the Tensors layer still go to Console directly; the bool-level
/// gate (<see cref="GpuDiagnosticsConfig.Verbose"/>) still suppresses
/// them when <see cref="GpuDiagnosticLevel.Silent"/> or
/// <see cref="GpuDiagnosticLevel.Minimal"/>.
/// </para>
/// <para>
/// For Microsoft.Extensions.Logging integration, see
/// <see cref="GpuDiagnosticsLoggerExtensions.ToSink(Microsoft.Extensions.Logging.ILogger)"/>
/// which wraps an <c>ILogger</c> instance as a sink.
/// </para>
/// </remarks>
public delegate void GpuDiagnosticSink(GpuDiagnosticLevel level, string message);
