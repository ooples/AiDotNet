using AiDotNet.Deployment.Configuration;

namespace AiDotNet.Configuration;

/// <summary>Default implementation of <see cref="IAiModelObservability"/>. Audit-2026-05 phase-2a slice 10.</summary>
internal class AiModelObservability : IAiModelObservability
{
    public BenchmarkingOptions? BenchmarkingOptions { get; private set; }
    public ProfilingConfig? ProfilingConfig { get; private set; }
    public TelemetryConfig? TelemetryConfig { get; private set; }

    public void ConfigureBenchmarking(BenchmarkingOptions? options)
        => BenchmarkingOptions = options ?? new BenchmarkingOptions();

    public void ConfigureProfiling(ProfilingConfig? config)
        => ProfilingConfig = config ?? new ProfilingConfig { Enabled = true };

    public void ConfigureTelemetry(TelemetryConfig? config) => TelemetryConfig = config;

    public void ConfigureGpuDiagnostics(GpuDiagnosticsOptions? options)
    {
        // Mirrors the pre-refactor inline behavior verbatim: mutate the process-wide static
        // GpuDiagnosticsConfig — no per-instance storage. Slices 2-12 generally avoid statics,
        // but ConfigureGpuDiagnostics is a deliberate exception: GPU diagnostic level is a
        // process-global setting, not a per-build configuration.
        if (options is null) return;

        if (options.Level is GpuDiagnosticLevel level)
        {
            GpuDiagnosticsConfig.Level = level;
        }
        else if (options.Verbose is bool verbose)
        {
            GpuDiagnosticsConfig.Verbose = verbose;
        }

        if (options.Sink is GpuDiagnosticSink sink)
        {
            GpuDiagnosticsConfig.Sink = sink;
        }
    }
}
