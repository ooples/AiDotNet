using AiDotNet.LearningRateSchedulers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LearningRateSchedulers;

/// <summary>
/// Pins the warmup-hold-decay shape of the extended Noam schedule (Squeezeformer).
/// </summary>
public class NoamHoldAnnealingSchedulerTests
{
    [Fact]
    public void TheHoldPhase_EndingPastIntMax_StillHoldsThePeak()
    {
        // WarmupSteps + HoldSteps = int.MaxValue + 9 wraps negative in int arithmetic, which made
        // every post-warmup step skip the hold and decay instead.
        var scheduler = new NoamHoldAnnealingScheduler(1e-3, warmupSteps: 10, holdSteps: int.MaxValue, decayRate: 0.5);

        Assert.Equal(1e-3, scheduler.GetLearningRateAtStep(9), precision: 15);
        Assert.Equal(1e-3, scheduler.GetLearningRateAtStep(1_000_000), precision: 15);
        Assert.Equal(1e-3, scheduler.GetLearningRateAtStep(int.MaxValue - 1), precision: 15);
    }

    [Fact]
    public void TheDecayStartsWhenTheHoldEnds()
    {
        // step + 1 = warmup + hold starts the decay: (warmup / (step + 1 - hold))^rate = (10 / 10)^0.5 = 1,
        // then (10 / 11)^0.5 on the next step.
        var scheduler = new NoamHoldAnnealingScheduler(1e-3, warmupSteps: 10, holdSteps: 5, decayRate: 0.5);

        Assert.Equal(1e-3, scheduler.GetLearningRateAtStep(13), precision: 15);
        Assert.Equal(1e-3, scheduler.GetLearningRateAtStep(14), precision: 15);
        Assert.Equal(1e-3 * System.Math.Sqrt(10.0 / 11.0), scheduler.GetLearningRateAtStep(15), precision: 15);
    }
}
