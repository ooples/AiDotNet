using System;
using AiDotNet.Configuration;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Covers the always-on fused-path counters (#1930, ask 1: "surface the miss reason").
/// </summary>
/// <remarks>
/// <para>
/// <c>FusedOptimizerPathEvent</c> already carried the reason, but <c>Emit</c> drops anything below the
/// active <see cref="TrainingDiagnosticsConfig.Level"/>, which is <c>Silent</c> by default. So the reason
/// was only observable to someone who suspected a problem and turned diagnostics on BEFORE the run — which
/// is precisely not the situation the issue describes.
/// </para>
/// <para>
/// The issue's worked example: RWKVForecaster produced NaN across its parameter buffer after one training
/// step, intermittently, and the control arm used an optimizer with no fused spec — so compiled training
/// never engaged for that arm and nothing recorded it. The correlation with "stateful optimizers" was read
/// as an allocator fault. These tests pin that a run like that now leaves evidence behind by default.
/// </para>
/// </remarks>
/// <remarks>
/// In the existing <c>TrainingDiagnosticsSequential</c> collection because the counters and the level are
/// process-global: any training test running concurrently increments the same counters, and asserting exact
/// values against a shared counter is how a test becomes flaky rather than wrong.
/// </remarks>
[Collection("TrainingDiagnosticsSequential")]
public class FusedOptimizerPathCountersTests : IDisposable
{
    private readonly TrainingDiagnosticLevel _originalLevel;

    public FusedOptimizerPathCountersTests()
    {
        _originalLevel = TrainingDiagnosticsConfig.Level;
        TrainingDiagnosticsConfig.ResetFusedOptimizerCounters();
    }

    public void Dispose()
    {
        TrainingDiagnosticsConfig.Level = _originalLevel;
        TrainingDiagnosticsConfig.ResetFusedOptimizerCounters();
    }

    /// <summary>
    /// A miss is counted and its reason retained with diagnostics at their default Silent level.
    /// </summary>
    /// <remarks>
    /// This is the assertion that would have failed before: at Silent, <c>Emit</c> returns immediately and
    /// nothing anywhere recorded that the fused path had been skipped.
    /// </remarks>
    [Fact]
    public void MissIsRecordedAtTheDefaultSilentLevel()
    {
        TrainingDiagnosticsConfig.Level = TrainingDiagnosticLevel.Silent;

        TrainingDiagnosticsConfig.RecordFusedOptimizerPath(
            hit: false, reason: "optimizer GradientDescentOptimizer not compatible with fused kernel");

        Assert.Equal(0, TrainingDiagnosticsConfig.FusedOptimizerHits);
        Assert.Equal(1, TrainingDiagnosticsConfig.FusedOptimizerMisses);
        Assert.Contains("not compatible with fused kernel",
            TrainingDiagnosticsConfig.LastFusedOptimizerMissReason);
    }

    /// <summary>
    /// Hits are counted too, so "the fused path never engaged" is distinguishable from "nothing trained".
    /// </summary>
    /// <remarks>
    /// A miss counter alone cannot tell those apart — both leave it at zero — and that ambiguity is what
    /// made the original investigation read the fused path as a non-variable.
    /// </remarks>
    [Fact]
    public void HitsAndMissesAreCountedSeparately()
    {
        TrainingDiagnosticsConfig.Level = TrainingDiagnosticLevel.Silent;

        TrainingDiagnosticsConfig.RecordFusedOptimizerPath(hit: true, reason: null);
        TrainingDiagnosticsConfig.RecordFusedOptimizerPath(hit: true, reason: null);
        TrainingDiagnosticsConfig.RecordFusedOptimizerPath(hit: false, reason: "no trainable layers");

        Assert.Equal(2, TrainingDiagnosticsConfig.FusedOptimizerHits);
        Assert.Equal(1, TrainingDiagnosticsConfig.FusedOptimizerMisses);
        Assert.Equal("no trainable layers", TrainingDiagnosticsConfig.LastFusedOptimizerMissReason);
    }

    /// <summary>
    /// A hit does not erase the previous miss reason.
    /// </summary>
    /// <remarks>
    /// The failure this guards is a run that misses early — during warmup, say — then fuses for the rest.
    /// Clearing the reason on every hit would leave the miss count non-zero with nothing to explain it.
    /// </remarks>
    [Fact]
    public void AHitDoesNotClearTheLastMissReason()
    {
        TrainingDiagnosticsConfig.RecordFusedOptimizerPath(hit: false, reason: "sticky-disabled from prior fallback");
        TrainingDiagnosticsConfig.RecordFusedOptimizerPath(hit: true, reason: null);

        Assert.Equal("sticky-disabled from prior fallback",
            TrainingDiagnosticsConfig.LastFusedOptimizerMissReason);
    }

    /// <summary>
    /// Reset clears both counters and the reason, so one test's run cannot be read as another's.
    /// </summary>
    [Fact]
    public void ResetClearsCountersAndReason()
    {
        TrainingDiagnosticsConfig.RecordFusedOptimizerPath(hit: true, reason: null);
        TrainingDiagnosticsConfig.RecordFusedOptimizerPath(hit: false, reason: "some reason");

        TrainingDiagnosticsConfig.ResetFusedOptimizerCounters();

        Assert.Equal(0, TrainingDiagnosticsConfig.FusedOptimizerHits);
        Assert.Equal(0, TrainingDiagnosticsConfig.FusedOptimizerMisses);
        Assert.Null(TrainingDiagnosticsConfig.LastFusedOptimizerMissReason);
    }
}
