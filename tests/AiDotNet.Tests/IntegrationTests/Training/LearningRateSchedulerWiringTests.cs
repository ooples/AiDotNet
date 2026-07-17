using System;
using System.Threading.Tasks;
using AiDotNet.LearningRateSchedulers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Covers the learning-rate schedule contract: the metric reaches metric-driven schedules, and the
/// adaptive rule behaves identically now that it is expressed as a scheduler.
/// </summary>
/// <remarks>
/// Two mechanisms used to write the optimizer's learning rate: an attached ILearningRateScheduler,
/// and an inline branch in OptimizerBase that fired when UseAdaptiveLearningRate was set. The inline
/// branch overwrote whatever the scheduler had just set, so a configured schedule silently did
/// nothing. The adaptive rule is now AdaptiveFitnessScheduler, leaving exactly one writer.
/// </remarks>
public class LearningRateSchedulerWiringTests
{
    [Fact(Timeout = 60000)]
    public async Task StepWithMetric_IsOnTheContract_AndStepDrivenSchedulesIgnoreIt()
    {
        await Task.Yield();
        // A step-driven schedule must behave identically whether or not a metric is supplied, so a
        // caller can drive every schedule uniformly through Step(metric).
        ILearningRateScheduler withMetric = new ExponentialLRScheduler(0.1, gamma: 0.5);
        ILearningRateScheduler withoutMetric = new ExponentialLRScheduler(0.1, gamma: 0.5);

        for (int i = 0; i < 3; i++)
        {
            withMetric.Step(metric: 42.0);
            withoutMetric.Step();
        }

        Assert.Equal(withoutMetric.CurrentLearningRate, withMetric.CurrentLearningRate, precision: 12);
    }

    [Fact(Timeout = 60000)]
    public async Task ReduceOnPlateau_ReducesWhenSteppedThroughTheInterface()
    {
        await Task.Yield();
        // The regression that motivated putting the metric on the contract: driving this scheduler
        // through ILearningRateScheduler used to hit the metric-less overload, which never reduces.
        ILearningRateScheduler scheduler = new ReduceOnPlateauScheduler(
            baseLearningRate: 0.1, factor: 0.5, patience: 1);

        scheduler.Step(1.0);   // baseline
        scheduler.Step(1.0);   // no improvement (1 bad epoch)
        scheduler.Step(1.0);   // no improvement -> exceeds patience -> reduce
        scheduler.Step(1.0);

        Assert.True(
            scheduler.CurrentLearningRate < 0.1,
            $"plateau scheduler never reduced through the interface (lr={scheduler.CurrentLearningRate})");
    }

    [Fact(Timeout = 60000)]
    public async Task ReduceOnPlateau_HoldsWhileImproving()
    {
        await Task.Yield();
        var scheduler = new ReduceOnPlateauScheduler(baseLearningRate: 0.1, factor: 0.5, patience: 1);

        for (double metric = 1.0; metric > 0.2; metric -= 0.2)
        {
            scheduler.Step(metric);
        }

        Assert.Equal(0.1, scheduler.CurrentLearningRate, precision: 12);
    }

    [Fact(Timeout = 60000)]
    public async Task AdaptiveScheduler_ShrinksWhileImproving_GrowsWhileStalled()
    {
        await Task.Yield();
        // The exact semantics of the inline rule it replaces: improving -> *decay, stalled -> /decay.
        var scheduler = new AdaptiveFitnessScheduler(baseLearningRate: 0.1, decay: 0.5);

        scheduler.Step(1.0);          // first observation is an improvement over +inf
        Assert.Equal(0.05, scheduler.CurrentLearningRate, precision: 12);

        scheduler.Step(0.5);          // improving again
        Assert.Equal(0.025, scheduler.CurrentLearningRate, precision: 12);

        scheduler.Step(0.9);          // worse -> grow
        Assert.Equal(0.05, scheduler.CurrentLearningRate, precision: 12);
    }

    [Fact(Timeout = 60000)]
    public async Task AdaptiveScheduler_HonorsMetricDirection()
    {
        await Task.Yield();
        // Fitness may be a score (R², accuracy) where HIGHER is better. Ignoring direction would
        // invert the rule — growing the rate exactly when the model is improving.
        var higherIsBetter = new AdaptiveFitnessScheduler(
            baseLearningRate: 0.1, decay: 0.5, higherIsBetter: true);

        higherIsBetter.Step(0.5);
        higherIsBetter.Step(0.9);     // improving for a score

        Assert.True(
            higherIsBetter.CurrentLearningRate < 0.1,
            "a rising score must count as improvement and shrink the rate");
    }

    [Fact(Timeout = 60000)]
    public async Task AdaptiveScheduler_ClampsToBounds()
    {
        await Task.Yield();
        var scheduler = new AdaptiveFitnessScheduler(
            baseLearningRate: 0.1, decay: 0.5, minLearningRate: 0.05, maxLearningRate: 0.2);

        for (int i = 0; i < 10; i++)
        {
            scheduler.Step(1.0 - (i * 0.01)); // improving forever
        }

        Assert.Equal(0.05, scheduler.CurrentLearningRate, precision: 12);

        for (int i = 0; i < 20; i++)
        {
            scheduler.Step(99.0); // stalled forever
        }

        Assert.Equal(0.2, scheduler.CurrentLearningRate, precision: 12);
    }

    [Fact(Timeout = 60000)]
    public async Task AdaptiveScheduler_RejectsDecayThatWouldInvertTheRule()
    {
        await Task.Yield();
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new AdaptiveFitnessScheduler(baseLearningRate: 0.1, decay: 1.5));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new AdaptiveFitnessScheduler(baseLearningRate: 0.1, decay: 0.0));
    }

    [Fact(Timeout = 60000)]
    public async Task AdaptiveScheduler_NonFiniteMetric_CountsAsStalled()
    {
        await Task.Yield();
        var scheduler = new AdaptiveFitnessScheduler(baseLearningRate: 0.1, decay: 0.5);
        scheduler.Step(1.0);   // -> 0.05

        scheduler.Step(double.NaN);

        // A diverged run must not read as an improvement.
        Assert.Equal(0.1, scheduler.CurrentLearningRate, precision: 12);
    }
}
