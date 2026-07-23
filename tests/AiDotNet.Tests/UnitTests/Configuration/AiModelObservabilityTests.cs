using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Deployment.Configuration;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Configuration;

/// <summary>Audit-2026-05 phase-2a slice 10 — observability component isolation tests.</summary>
public class AiModelObservabilityTests
{
    [Fact(Timeout = 30000)]
    public async Task InitialState_AllSlotsAreNull()
    {
        await Task.Yield();
        var o = new AiModelObservability();
        Assert.Null(o.BenchmarkingOptions);
        Assert.Null(o.ProfilingConfig);
        Assert.Null(o.TelemetryConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureBenchmarking_NullAppliesDefault()
    {
        await Task.Yield();
        var o = new AiModelObservability();
        o.ConfigureBenchmarking(null);
        Assert.NotNull(o.BenchmarkingOptions);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureBenchmarking_ExplicitStored()
    {
        await Task.Yield();
        var o = new AiModelObservability();
        var opts = new BenchmarkingOptions();
        o.ConfigureBenchmarking(opts);
        Assert.Same(opts, o.BenchmarkingOptions);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureProfiling_NullAppliesEnabledDefault()
    {
        await Task.Yield();
        var o = new AiModelObservability();
        o.ConfigureProfiling(null);
        Assert.NotNull(o.ProfilingConfig);
        Assert.True(o.ProfilingConfig!.Enabled);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureTelemetry_NullIsValid()
    {
        await Task.Yield();
        var o = new AiModelObservability();
        o.ConfigureTelemetry(null);
        Assert.Null(o.TelemetryConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureTelemetry_ExplicitStored()
    {
        await Task.Yield();
        var o = new AiModelObservability();
        var cfg = new TelemetryConfig();
        o.ConfigureTelemetry(cfg);
        Assert.Same(cfg, o.TelemetryConfig);
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureGpuDiagnostics_Null_NoOps()
    {
        await Task.Yield();
        var o = new AiModelObservability();
        o.ConfigureGpuDiagnostics(null); // should not throw
    }

    [Fact(Timeout = 30000)]
    public async Task ConfigureGpuDiagnostics_LevelMutatesStatic()
    {
        await Task.Yield();
        var o = new AiModelObservability();
        var originalLevel = GpuDiagnosticsConfig.Level;
        try
        {
            o.ConfigureGpuDiagnostics(new GpuDiagnosticsOptions { Level = GpuDiagnosticLevel.Verbose });
            Assert.Equal(GpuDiagnosticLevel.Verbose, GpuDiagnosticsConfig.Level);
        }
        finally
        {
            GpuDiagnosticsConfig.Level = originalLevel; // restore process state
        }
    }
}
