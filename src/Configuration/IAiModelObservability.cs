using AiDotNet.Deployment.Configuration;

namespace AiDotNet.Configuration;

/// <summary>
/// Component that owns the observability configuration for an AI model build: benchmarking,
/// profiling, telemetry, and GPU diagnostics. Audit-2026-05 phase-2a slice 10.
/// </summary>
public interface IAiModelObservability
{
    BenchmarkingOptions? BenchmarkingOptions { get; }
    ProfilingConfig? ProfilingConfig { get; }
    TelemetryConfig? TelemetryConfig { get; }

    void ConfigureBenchmarking(BenchmarkingOptions? options);
    void ConfigureProfiling(ProfilingConfig? config);
    void ConfigureTelemetry(TelemetryConfig? config);

    /// <summary>
    /// Configures GPU diagnostics. Mutates the process-wide static
    /// <see cref="GpuDiagnosticsConfig"/> directly per the pre-refactor semantics; no per-instance
    /// state stored in this component.
    /// </summary>
    void ConfigureGpuDiagnostics(GpuDiagnosticsOptions? options);
}
