using System.Threading.Tasks;
using AiDotNet.Diffusion.Schedulers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.Schedulers;

/// <summary>
/// Regression tests for <see cref="FlowMatchingScheduler{T}.SetTimesteps"/>: the linearly spaced
/// flow-matching schedule it builds must actually be installed. It used to be computed and then
/// discarded in favour of the base integer-stride schedule, which stops far short of t = 0 whenever
/// the step count does not divide the training timesteps.
/// </summary>
public class FlowMatchingScheduleCollectionTests
{
    [Fact(Timeout = 60000)]
    public async Task SetTimesteps_NonDivisorStepCount_SpansTheWholeRangeDownToNearZero()
    {
        var scheduler = new FlowMatchingScheduler<double>(SchedulerConfig<double>.CreateRectifiedFlow());

        scheduler.SetTimesteps(600);
        var timesteps = scheduler.Timesteps;

        Assert.Equal(600, timesteps.Length);
        Assert.Equal(999, timesteps[0]);
        // Linear spacing of 999/600 ~= 1.665 ends at round(1.665) = 2. The base stride
        // (1000 / 600 = 1) stopped at 400, leaving a single 0.4-wide final Euler jump to t = 0.
        Assert.Equal(2, timesteps[timesteps.Length - 1]);
        for (int i = 1; i < timesteps.Length; i++)
            Assert.True(timesteps[i] < timesteps[i - 1], $"Schedule not strictly decreasing at index {i}.");

        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task SetTimesteps_28Steps_UsesLinearSpacingNotIntegerStride()
    {
        var scheduler = new FlowMatchingScheduler<double>(SchedulerConfig<double>.CreateRectifiedFlow());

        scheduler.SetTimesteps(28);
        var timesteps = scheduler.Timesteps;

        Assert.Equal(28, timesteps.Length);
        Assert.Equal(999, timesteps[0]);
        // 999 - 27 * (999 / 28) = 35.68 -> 36. The base stride of 35 ended at 999 - 27 * 35 = 54.
        Assert.Equal(36, timesteps[timesteps.Length - 1]);

        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task SetTimesteps_AllTrainingSteps_HasNoRepeatedTimestep()
    {
        var scheduler = new FlowMatchingScheduler<double>(SchedulerConfig<double>.CreateRectifiedFlow());

        scheduler.SetTimesteps(1000);
        var timesteps = scheduler.Timesteps;

        Assert.Equal(1000, timesteps.Length);
        Assert.Equal(999, timesteps[0]);
        Assert.Equal(0, timesteps[timesteps.Length - 1]);
        for (int i = 1; i < timesteps.Length; i++)
            Assert.True(timesteps[i] < timesteps[i - 1], $"Repeated or rising timestep at index {i}.");

        await Task.CompletedTask;
    }
}
