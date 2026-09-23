using System;
using AiDotNet.LearningRateSchedulers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LearningRateSchedulers;

/// <summary>
/// Pins the three phases of the tri-stage schedule (#1928).
/// </summary>
/// <remarks>
/// The reason this schedule exists rather than reusing a near neighbour is that the hold phase
/// changes where decay starts. These assert the shape, not just the endpoints, because a scheduler
/// that ramps and decays correctly but skips the hold would pass any test that only looked at the
/// first and last step — and would begin decaying four times earlier than the paper.
/// </remarks>
public class TriStageSchedulerTests
{
    private static TriStageScheduler Wav2Vec2Style(int total = 1000)
        => TriStageScheduler.FromFractions(
            baseLearningRate: 1e-4, totalSteps: total, warmupFraction: 0.1, holdFraction: 0.4);

    [Fact]
    public void TheRateRampsToThePeakAcrossWarmupAndNotBefore()
    {
        var scheduler = Wav2Vec2Style();

        Assert.True(scheduler.GetLearningRateAtStep(0) < 1e-4);
        Assert.True(scheduler.GetLearningRateAtStep(50) < scheduler.GetLearningRateAtStep(99));
        Assert.Equal(1e-4, scheduler.GetLearningRateAtStep(99), precision: 12);
    }

    [Fact]
    public void TheRateHoldsAtThePeakForTheWholeHoldPhase()
    {
        // The phase that distinguishes this schedule from warmup-then-decay.
        var scheduler = Wav2Vec2Style();

        foreach (int step in new[] { 100, 250, 400, 499 })
            Assert.Equal(1e-4, scheduler.GetLearningRateAtStep(step), precision: 12);

        // And decay has genuinely begun immediately after it, rather than the hold running long.
        Assert.True(scheduler.GetLearningRateAtStep(501) < 1e-4);
    }

    [Fact]
    public void TheRateDecaysLinearlyToTheFloorOverTheRemainder()
    {
        var scheduler = Wav2Vec2Style();

        // Halfway through the 500-step decay phase the rate is halfway down.
        Assert.Equal(0.5e-4, scheduler.GetLearningRateAtStep(750), precision: 6);
        Assert.Equal(0.0, scheduler.GetLearningRateAtStep(1000), precision: 12);

        // Linear, not merely decreasing: equal step intervals give equal drops.
        double first = scheduler.GetLearningRateAtStep(600) - scheduler.GetLearningRateAtStep(700);
        double second = scheduler.GetLearningRateAtStep(800) - scheduler.GetLearningRateAtStep(900);
        Assert.Equal(first, second, precision: 12);
    }

    [Fact]
    public void ADeclaredFloorIsHonouredAtBothEnds()
    {
        var scheduler = new TriStageScheduler(
            baseLearningRate: 1e-3, warmupSteps: 10, holdSteps: 10, totalSteps: 100,
            minLearningRate: 1e-5);

        Assert.True(scheduler.GetLearningRateAtStep(0) >= 1e-5);
        Assert.Equal(1e-5, scheduler.GetLearningRateAtStep(100), precision: 12);
    }

    [Fact]
    public void APhaseLayoutWithNoRoomToDecayIsRejected()
    {
        // Clamping this would turn a mis-transcribed paper value into a silently different
        // schedule, which is the failure this whole area is meant to prevent.
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new TriStageScheduler(1e-3, warmupSteps: 60, holdSteps: 60, totalSteps: 100));
    }
}
