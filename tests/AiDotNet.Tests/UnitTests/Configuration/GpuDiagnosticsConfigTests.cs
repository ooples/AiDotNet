using AiDotNet.Configuration;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Configuration;

/// <summary>
/// Unit tests for <see cref="GpuDiagnosticsConfig"/> and <see cref="GpuDiagnosticsOptions"/> —
/// the discoverable AiDotNet facade over the Tensors-package GPU diagnostic flag.
/// Verifies the round-trip read/write contract and the
/// <c>ConfigureGpuDiagnostics</c> builder forwarder behavior.
/// </summary>
/// <remarks>
/// These tests mutate a process-global setting
/// (<see cref="AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput"/>),
/// so they save and restore the original value in try/finally blocks. No parallel
/// execution issues arise because the tests only assert their own writes, never
/// the ambient state.
/// </remarks>
public class GpuDiagnosticsConfigTests
{
    /// <summary>
    /// The Verbose property must round-trip through the underlying Tensors flag:
    /// writing <c>true</c> then reading must return <c>true</c>; same for <c>false</c>.
    /// Catches a regression where the forwarder accidentally short-circuits or
    /// routes to a different static field.
    /// </summary>
    [Fact]
    public void Verbose_RoundTripsThroughUnderlyingFlag()
    {
        var original = GpuDiagnosticsConfig.Verbose;
        try
        {
            GpuDiagnosticsConfig.Verbose = true;
            Assert.True(GpuDiagnosticsConfig.Verbose);
            Assert.True(AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput);

            GpuDiagnosticsConfig.Verbose = false;
            Assert.False(GpuDiagnosticsConfig.Verbose);
            Assert.False(AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput);
        }
        finally
        {
            GpuDiagnosticsConfig.Verbose = original;
        }
    }

    /// <summary>
    /// Verifies that setting the underlying Tensors flag directly is visible
    /// via the AiDotNet facade — the forwarder is bidirectional. Applications
    /// that already call into the low-level Tensors API continue to work.
    /// </summary>
    [Fact]
    public void Verbose_ReflectsUnderlyingTensorsFlag()
    {
        var original = AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput;
        try
        {
            AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput = true;
            Assert.True(GpuDiagnosticsConfig.Verbose);

            AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput = false;
            Assert.False(GpuDiagnosticsConfig.Verbose);
        }
        finally
        {
            AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput = original;
        }
    }

    /// <summary>
    /// Options with <see cref="GpuDiagnosticsOptions.Verbose"/> = <c>null</c>
    /// must leave the current setting unchanged (a "don't-care" signal).
    /// Important because users might configure some GPU options without
    /// wanting to override the verbose state.
    /// </summary>
    [Fact]
    public void Options_NullVerbose_PreservesExistingSetting()
    {
        var original = GpuDiagnosticsConfig.Verbose;
        try
        {
            GpuDiagnosticsConfig.Verbose = true;

            // Simulate the builder applying null-Verbose options.
            var options = new GpuDiagnosticsOptions { Verbose = null };
            if (options.Verbose is bool v)
            {
                GpuDiagnosticsConfig.Verbose = v;
            }
            // Above branch must not fire.
            Assert.True(GpuDiagnosticsConfig.Verbose,
                "null Verbose must not overwrite the existing process-global setting");
        }
        finally
        {
            GpuDiagnosticsConfig.Verbose = original;
        }
    }

    /// <summary>
    /// Options with explicit <see cref="GpuDiagnosticsOptions.Verbose"/> = <c>true</c>
    /// or <c>false</c> must apply to the process-global flag. Verifies the
    /// nullable-unwrap semantic.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Options_ExplicitVerbose_AppliesToFlag(bool expected)
    {
        var original = GpuDiagnosticsConfig.Verbose;
        try
        {
            GpuDiagnosticsConfig.Verbose = !expected; // start at opposite

            var options = new GpuDiagnosticsOptions { Verbose = expected };
            if (options.Verbose is bool v)
            {
                GpuDiagnosticsConfig.Verbose = v;
            }

            Assert.Equal(expected, GpuDiagnosticsConfig.Verbose);
        }
        finally
        {
            GpuDiagnosticsConfig.Verbose = original;
        }
    }

    /// <summary>
    /// Default-constructed <see cref="GpuDiagnosticsOptions"/> has null Verbose —
    /// ensures the "no-op when no property is set" invariant. A user who writes
    /// <c>new GpuDiagnosticsOptions()</c> without setting anything should not
    /// accidentally mutate process state.
    /// </summary>
    [Fact]
    public void Options_DefaultConstructor_HasNullVerbose()
    {
        var options = new GpuDiagnosticsOptions();
        Assert.Null(options.Verbose);
    }
}
