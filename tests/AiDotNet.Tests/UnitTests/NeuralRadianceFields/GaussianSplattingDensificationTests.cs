using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralRadianceFields;

/// <summary>
/// Regression tests for #1835 — <see cref="GaussianSplattingOptions"/>'s new densification
/// schedule knobs (DensificationStartIteration, DensificationEndIteration,
/// GradientNormThreshold, OpacityPruneThreshold, MaxGaussianCount,
/// GradientAccumulationWindow) surface through GS's internal <c>Effective*</c> getters, and
/// the schedule window gates <c>DensifyAndPrune</c> so it doesn't fire during the warm-up
/// or post-freeze phases.
/// </summary>
public class GaussianSplattingDensificationTests
{
    private static GaussianSplatting<double> BuildGS(GaussianSplattingOptions options)
    {
        var pts = new Matrix<double>(4, 3);
        var cols = new Matrix<double>(4, 3);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                pts[i, j] = 0.1 * (i + j);
                cols[i, j] = 0.5;
            }
        }
        return new GaussianSplatting<double>(options, pts, cols);
    }

    [Fact]
    public void GaussianSplattingOptions_NewDensificationKnobs_AreNullable()
    {
        // Compilation check + null-default check.
        var o = new GaussianSplattingOptions();
        Assert.Null(o.DensificationStartIteration);
        Assert.Null(o.DensificationEndIteration);
        Assert.Null(o.GradientNormThreshold);
        Assert.Null(o.OpacityPruneThreshold);
        Assert.Null(o.MaxGaussianCount);
        Assert.Null(o.GradientAccumulationWindow);
    }

    [Fact]
    public void Options_ExplicitDensificationKnobs_RoundTripThroughOptions()
    {
        var o = new GaussianSplattingOptions
        {
            DensificationStartIteration = 200,
            DensificationEndIteration = 20000,
            GradientNormThreshold = 5e-4,
            OpacityPruneThreshold = 0.008,
            MaxGaussianCount = 1_500_000,
            GradientAccumulationWindow = 250,
            ShDegree = 0,
            EnableDensification = false,
            EnableSpatialIndex = false,
            MaxGaussians = 8,
        };
        var gs = BuildGS(o);

        // Effective* getters visible via InternalsVisibleTo("AiDotNetTests").
        Assert.Equal(200, gs.EffectiveDensificationStartIteration);
        Assert.Equal(20000, gs.EffectiveDensificationEndIteration);
        Assert.Equal(5e-4, gs.EffectiveGradientNormThreshold);
        Assert.Equal(0.008, gs.EffectiveOpacityPruneThreshold);
        Assert.Equal(1_500_000, gs.EffectiveMaxGaussianCount);
        Assert.Equal(250, gs.EffectiveGradientAccumulationWindow);
    }

    [Fact]
    public void Options_UnsetKnobs_FallBackToPaperDefaults()
    {
        var o = new GaussianSplattingOptions
        {
            ShDegree = 0,
            EnableDensification = false,
            EnableSpatialIndex = false,
            MaxGaussians = 8,
            // Legacy fields set (pre-#1835 callers).
            SplitGradientThreshold = 0.0002,
            PruneOpacityThreshold = 0.005,
        };
        var gs = BuildGS(o);

        // Paper defaults for the new-only knobs.
        Assert.Equal(500, gs.EffectiveDensificationStartIteration);
        Assert.Equal(15000, gs.EffectiveDensificationEndIteration);
        Assert.Equal(100, gs.EffectiveGradientAccumulationWindow);

        // Legacy fields fall through when the new knobs are null.
        Assert.Equal(0.0002, gs.EffectiveGradientNormThreshold);
        Assert.Equal(0.005, gs.EffectiveOpacityPruneThreshold);
        Assert.Equal(8, gs.EffectiveMaxGaussianCount);
    }

    [Fact]
    public void DensifyAndPrune_BeforeStartWindow_NoOps()
    {
        // Small run: 100 iters, start = 40, end = 90. Before iter 40, densification MUST NOT
        // fire — DensifyAndPrune's [Start, End] gate is what makes the schedule work.
        var o = new GaussianSplattingOptions
        {
            ShDegree = 0,
            EnableDensification = true,
            EnableSpatialIndex = false,
            MaxGaussians = 32,
            DensificationStartIteration = 40,
            DensificationEndIteration = 90,
            OpacityPruneThreshold = 0.99,   // Would prune almost everything if the gate misfires.
            GradientNormThreshold = 1e-9,   // Would split almost everything if the gate misfires.
        };
        var gs = BuildGS(o);
        int startCount = gs.GaussianCount;

        // Simulate iteration 10 — inside DensifyAndPrune's window check via internal helper.
        gs.RunDensifyAndPruneForTest(iteration: 10, gradientsMagnitude: 1.0);

        // Warm-up: nothing should have been split OR pruned.
        Assert.Equal(startCount, gs.GaussianCount);
    }

    [Fact]
    public void DensifyAndPrune_AfterEndWindow_NoOps()
    {
        var o = new GaussianSplattingOptions
        {
            ShDegree = 0,
            EnableDensification = true,
            EnableSpatialIndex = false,
            MaxGaussians = 32,
            DensificationStartIteration = 10,
            DensificationEndIteration = 20,
            OpacityPruneThreshold = 0.99,
            GradientNormThreshold = 1e-9,
        };
        var gs = BuildGS(o);
        int startCount = gs.GaussianCount;

        // Iteration 25 is past the freeze window — no changes should happen.
        gs.RunDensifyAndPruneForTest(iteration: 25, gradientsMagnitude: 1.0);

        Assert.Equal(startCount, gs.GaussianCount);
    }

    [Fact]
    public void DensifyAndPrune_InsideActiveWindow_ActuallyMutates()
    {
        // Positive case — inside [Start, End], with a gradient magnitude above the split
        // threshold AND opacity below the prune threshold, DensifyAndPrune MUST mutate.
        // Without this test the schedule window gates could be no-ops and the pass-only
        // suite (before/after) wouldn't catch it.
        var o = new GaussianSplattingOptions
        {
            ShDegree = 0,
            EnableDensification = true,
            EnableSpatialIndex = false,
            MaxGaussians = 32,
            DensificationStartIteration = 5,
            DensificationEndIteration = 100,
            // Very high prune threshold — every default-opacity (0.5) Gaussian gets culled.
            OpacityPruneThreshold = 0.99,
            // Very low split threshold — anything non-zero triggers splits.
            GradientNormThreshold = 1e-9,
        };
        var gs = BuildGS(o);
        int startCount = gs.GaussianCount;

        // Iteration 50 is comfortably inside the window.
        gs.RunDensifyAndPruneForTest(iteration: 50, gradientsMagnitude: 1.0);

        // Both prune (opacity 0.5 < 0.99 threshold) and split (gradient 1.0 > 1e-9 threshold)
        // should fire. Exact post-count depends on ordering; the invariant is the count
        // CHANGED. Either splits raced under the MaxGaussians cap, or prune emptied the cloud.
        Assert.NotEqual(startCount, gs.GaussianCount);
    }
}
