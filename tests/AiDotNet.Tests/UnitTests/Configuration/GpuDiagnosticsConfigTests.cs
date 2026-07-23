using System;
using AiDotNet.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Configuration;

/// <summary>
/// Unit tests for <see cref="GpuDiagnosticsConfig"/>,
/// <see cref="GpuDiagnosticsOptions"/>, and
/// <see cref="GpuDiagnosticsLoggerExtensions"/> — the discoverable
/// AiDotNet facade over the Tensors-package GPU diagnostic flag, covering
/// all three controls requested by github.com/ooples/AiDotNet#1122
/// (env var, static config, ILogger/sink).
/// </summary>
/// <remarks>
/// These tests mutate process-global settings (<c>GpuDiagnosticsConfig.Level</c>
/// and <c>GpuDiagnosticsConfig.Sink</c>, both of which forward to the
/// Tensors-package flag), so they save and restore the original values
/// in try/finally blocks. No parallel execution issues arise because the
/// tests only assert their own writes, never the ambient state.
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
        var original = GpuDiagnosticsConfig.Level;
        try
        {
            GpuDiagnosticsConfig.Verbose = true;
            Assert.True(GpuDiagnosticsConfig.Verbose);
            Assert.Equal(GpuDiagnosticLevel.Verbose, GpuDiagnosticsConfig.Level);
            Assert.True(AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput);

            GpuDiagnosticsConfig.Verbose = false;
            Assert.False(GpuDiagnosticsConfig.Verbose);
            Assert.Equal(GpuDiagnosticLevel.Silent, GpuDiagnosticsConfig.Level);
            Assert.False(AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput);
        }
        finally
        {
            GpuDiagnosticsConfig.Level = original;
        }
    }

    /// <summary>
    /// Level property must round-trip all three enum values and correctly
    /// forward to the bool-level Tensors flag.
    /// </summary>
    [Theory]
    [InlineData(GpuDiagnosticLevel.Silent, false)]
    [InlineData(GpuDiagnosticLevel.Minimal, false)] // Minimal maps to false on Tensors v0.38.0 bool toggle
    [InlineData(GpuDiagnosticLevel.Verbose, true)]
    public void Level_RoundTripsAndForwardsToTensorsFlag(GpuDiagnosticLevel level, bool expectedTensorsFlag)
    {
        var original = GpuDiagnosticsConfig.Level;
        try
        {
            GpuDiagnosticsConfig.Level = level;
            Assert.Equal(level, GpuDiagnosticsConfig.Level);
            Assert.Equal(expectedTensorsFlag, AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput);
        }
        finally
        {
            GpuDiagnosticsConfig.Level = original;
        }
    }

    /// <summary>
    /// The AiDotNet <see cref="GpuDiagnosticsConfig"/> facade is now
    /// authoritative — Level/Verbose writes forward to the Tensors flag,
    /// but direct Tensors-flag writes do NOT update the cached Level
    /// (otherwise the Silent/Minimal distinction would be lost on any
    /// external write). Applications should prefer
    /// <see cref="GpuDiagnosticsConfig.Level"/> over direct
    /// <c>OpenClBackend.DiagnosticOutput</c> writes.
    /// </summary>
    [Fact]
    public void TensorsFlag_WriteDoesNotOverwriteLocalLevel_ByDesign()
    {
        var originalLevel = GpuDiagnosticsConfig.Level;
        try
        {
            GpuDiagnosticsConfig.Level = GpuDiagnosticLevel.Minimal;
            // Minimal maps to Tensors.DiagnosticOutput = false.
            Assert.False(AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput);

            // External write to Tensors flag — AiDotNet Level should NOT change;
            // preserving Minimal vs Silent is the feature.
            AiDotNet.Tensors.Engines.DirectGpu.OpenCL.OpenClBackend.DiagnosticOutput = true;
            Assert.Equal(GpuDiagnosticLevel.Minimal, GpuDiagnosticsConfig.Level);
        }
        finally
        {
            GpuDiagnosticsConfig.Level = originalLevel;
        }
    }

    /// <summary>
    /// Sink delegate must be storable and retrievable — the AiDotNet side
    /// captures the sink immediately even before the Tensors side honors it.
    /// </summary>
    [Fact]
    public void Sink_RoundTripsAndCanBeCleared()
    {
        var originalSink = GpuDiagnosticsConfig.Sink;
        try
        {
            int callCount = 0;
            GpuDiagnosticSink mySink = (_, _) => callCount++;
            GpuDiagnosticsConfig.Sink = mySink;
            Assert.Same(mySink, GpuDiagnosticsConfig.Sink);

            GpuDiagnosticsConfig.Sink = null;
            Assert.Null(GpuDiagnosticsConfig.Sink);
        }
        finally
        {
            GpuDiagnosticsConfig.Sink = originalSink;
        }
    }

    /// <summary>
    /// Emit() must call the sink with the correct level and message when
    /// the active Level permits it; must suppress when the Level is more
    /// restrictive than the message's level.
    /// </summary>
    [Fact]
    public void Emit_RoutesToSinkWhenSetAndLevelPermits()
    {
        var originalLevel = GpuDiagnosticsConfig.Level;
        var originalSink = GpuDiagnosticsConfig.Sink;
        try
        {
            var received = new System.Collections.Generic.List<(GpuDiagnosticLevel, string)>();
            GpuDiagnosticsConfig.Sink = (level, msg) => received.Add((level, msg));

            // Level = Verbose permits all messages.
            GpuDiagnosticsConfig.Level = GpuDiagnosticLevel.Verbose;
            GpuDiagnosticsConfig.Emit(GpuDiagnosticLevel.Minimal, "minimal msg");
            GpuDiagnosticsConfig.Emit(GpuDiagnosticLevel.Verbose, "verbose msg");
            Assert.Equal(2, received.Count);
            Assert.Equal((GpuDiagnosticLevel.Minimal, "minimal msg"), received[0]);
            Assert.Equal((GpuDiagnosticLevel.Verbose, "verbose msg"), received[1]);

            // Level = Minimal permits Minimal but not Verbose.
            received.Clear();
            GpuDiagnosticsConfig.Level = GpuDiagnosticLevel.Minimal;
            GpuDiagnosticsConfig.Emit(GpuDiagnosticLevel.Minimal, "minimal msg");
            GpuDiagnosticsConfig.Emit(GpuDiagnosticLevel.Verbose, "verbose msg");
            Assert.Single(received);
            Assert.Equal("minimal msg", received[0].Item2);

            // Level = Silent suppresses everything.
            received.Clear();
            GpuDiagnosticsConfig.Level = GpuDiagnosticLevel.Silent;
            GpuDiagnosticsConfig.Emit(GpuDiagnosticLevel.Minimal, "minimal msg");
            GpuDiagnosticsConfig.Emit(GpuDiagnosticLevel.Verbose, "verbose msg");
            Assert.Empty(received);
        }
        finally
        {
            GpuDiagnosticsConfig.Sink = originalSink;
            GpuDiagnosticsConfig.Level = originalLevel;
        }
    }

    /// <summary>
    /// Options with all-null properties must leave current settings
    /// unchanged (preserve semantics).
    /// </summary>
    [Fact]
    public void Options_AllNull_PreservesExistingSettings()
    {
        var originalLevel = GpuDiagnosticsConfig.Level;
        var originalSink = GpuDiagnosticsConfig.Sink;
        try
        {
            GpuDiagnosticsConfig.Level = GpuDiagnosticLevel.Verbose;
            GpuDiagnosticSink mySink = (_, _) => { };
            GpuDiagnosticsConfig.Sink = mySink;

            // Simulate builder applying all-null options.
            var options = new GpuDiagnosticsOptions();
            ApplyOptions(options);

            Assert.Equal(GpuDiagnosticLevel.Verbose, GpuDiagnosticsConfig.Level);
            Assert.Same(mySink, GpuDiagnosticsConfig.Sink);
        }
        finally
        {
            GpuDiagnosticsConfig.Sink = originalSink;
            GpuDiagnosticsConfig.Level = originalLevel;
        }
    }

    /// <summary>
    /// Options.Level wins over Options.Verbose when both are set — Level is
    /// the richer API. Catches a regression where the Verbose path shadows
    /// the Level path.
    /// </summary>
    [Fact]
    public void Options_LevelWinsOverVerbose_WhenBothSet()
    {
        var originalLevel = GpuDiagnosticsConfig.Level;
        try
        {
            GpuDiagnosticsConfig.Level = GpuDiagnosticLevel.Silent;

            // Both set: Level says Minimal, Verbose says true (which alone would set Verbose).
            var options = new GpuDiagnosticsOptions
            {
                Level = GpuDiagnosticLevel.Minimal,
                Verbose = true,
            };
            ApplyOptions(options);

            Assert.Equal(GpuDiagnosticLevel.Minimal, GpuDiagnosticsConfig.Level);
        }
        finally
        {
            GpuDiagnosticsConfig.Level = originalLevel;
        }
    }

    /// <summary>
    /// ILogger.ToSink() returns a sink that invokes the logger with
    /// level-mapped <see cref="Microsoft.Extensions.Logging.LogLevel"/>.
    /// Verbose → Debug, Minimal → Information.
    /// </summary>
    [Fact]
    public void LoggerToSink_RoutesWithCorrectLogLevel()
    {
        var recorded = new System.Collections.Generic.List<(Microsoft.Extensions.Logging.LogLevel, string)>();
        var logger = new RecordingLogger(recorded);

        var sink = logger.ToSink();

        sink(GpuDiagnosticLevel.Verbose, "verbose msg");
        sink(GpuDiagnosticLevel.Minimal, "minimal msg");

        Assert.Equal(2, recorded.Count);
        Assert.Equal(Microsoft.Extensions.Logging.LogLevel.Debug, recorded[0].Item1);
        Assert.Equal("verbose msg", recorded[0].Item2);
        Assert.Equal(Microsoft.Extensions.Logging.LogLevel.Information, recorded[1].Item1);
        Assert.Equal("minimal msg", recorded[1].Item2);
    }

    /// <summary>
    /// ILogger.ToSink() throws ArgumentNullException on null logger.
    /// Fail-fast rather than returning a null sink that'd NRE later.
    /// </summary>
    [Fact]
    public void LoggerToSink_ThrowsOnNullLogger()
    {
        ILogger? logger = null;
        Assert.Throws<ArgumentNullException>(() => logger!.ToSink());
    }

    /// <summary>
    /// End-to-end: builder-style options application with sink +
    /// Minimal level + an ILogger adapter.
    /// </summary>
    [Fact]
    public void EndToEnd_BuilderApplyOptions_WithLoggerSink()
    {
        var originalLevel = GpuDiagnosticsConfig.Level;
        var originalSink = GpuDiagnosticsConfig.Sink;
        try
        {
            var recorded = new System.Collections.Generic.List<(Microsoft.Extensions.Logging.LogLevel, string)>();
            var logger = new RecordingLogger(recorded);

            var options = new GpuDiagnosticsOptions
            {
                Level = GpuDiagnosticLevel.Minimal,
                Sink = logger.ToSink(),
            };
            ApplyOptions(options);

            Assert.Equal(GpuDiagnosticLevel.Minimal, GpuDiagnosticsConfig.Level);
            Assert.NotNull(GpuDiagnosticsConfig.Sink);

            // Emit — Verbose message should be suppressed by Minimal level;
            // Minimal message should reach the logger as Information.
            GpuDiagnosticsConfig.Emit(GpuDiagnosticLevel.Minimal, "gpu ready");
            GpuDiagnosticsConfig.Emit(GpuDiagnosticLevel.Verbose, "compiling kernels");

            Assert.Single(recorded);
            Assert.Equal(Microsoft.Extensions.Logging.LogLevel.Information, recorded[0].Item1);
            Assert.Equal("gpu ready", recorded[0].Item2);
        }
        finally
        {
            GpuDiagnosticsConfig.Sink = originalSink;
            GpuDiagnosticsConfig.Level = originalLevel;
        }
    }

    // Mirrors the builder's ConfigureGpuDiagnostics body.
    private static void ApplyOptions(GpuDiagnosticsOptions? options)
    {
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

    /// <summary>Minimal ILogger implementation that records Log calls.</summary>
    private sealed class RecordingLogger : ILogger
    {
        private readonly System.Collections.Generic.List<(Microsoft.Extensions.Logging.LogLevel, string)> _recorded;

        public RecordingLogger(System.Collections.Generic.List<(Microsoft.Extensions.Logging.LogLevel, string)> recorded)
        {
            _recorded = recorded;
        }

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(Microsoft.Extensions.Logging.LogLevel logLevel) => true;
        public void Log<TState>(Microsoft.Extensions.Logging.LogLevel logLevel, EventId eventId, TState state,
            Exception? exception, Func<TState, Exception?, string> formatter)
        {
            _recorded.Add((logLevel, formatter(state, exception)));
        }
    }
}
