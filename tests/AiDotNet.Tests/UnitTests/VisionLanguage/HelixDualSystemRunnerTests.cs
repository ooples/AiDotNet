using System.Threading.Tasks;
using AiDotNet.VisionLanguage.Robotics;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Unit tests for Helix's dual-system fast/slow rate splitter (Figure AI 2025, arXiv:2502.07092).
/// Covers System-2 caching, the S1:S2 ratio contract, and reset behaviour. Uses callback fakes —
/// no actual model weights required.
/// </summary>
public class HelixDualSystemRunnerTests
{
    [Fact(Timeout = 30000)]
    public async Task FirstStep_InvokesSystem2()
    {
        await Task.Yield();
        int s2InvocationCount = 0;
        int s1InvocationCount = 0;

        var runner = new HelixDualSystemRunner<double>(
            system2Forward: (_, _) => { s2InvocationCount++; return new Tensor<double>([4]); },
            system1Forward: (_, _) => { s1InvocationCount++; return new Tensor<double>([2]); },
            system2TicksValid: 5);

        runner.Step(new Tensor<double>([3]), "go");

        Assert.Equal(1, s2InvocationCount);
        Assert.Equal(1, s1InvocationCount);
    }

    [Fact(Timeout = 30000)]
    public async Task NextFourSteps_ReuseCachedSystem2Latent()
    {
        await Task.Yield();
        int s2InvocationCount = 0;
        int s1InvocationCount = 0;

        var runner = new HelixDualSystemRunner<double>(
            system2Forward: (_, _) => { s2InvocationCount++; return new Tensor<double>([4]); },
            system1Forward: (_, _) => { s1InvocationCount++; return new Tensor<double>([2]); },
            system2TicksValid: 5);

        for (int i = 0; i < 5; i++)
            runner.Step(new Tensor<double>([3]), "go");

        // S2 fires once at tick 0; S1 fires every tick.
        Assert.Equal(1, s2InvocationCount);
        Assert.Equal(5, s1InvocationCount);
    }

    [Fact(Timeout = 30000)]
    public async Task SixthStep_ReinvokesSystem2()
    {
        await Task.Yield();
        int s2InvocationCount = 0;

        var runner = new HelixDualSystemRunner<double>(
            system2Forward: (_, _) => { s2InvocationCount++; return new Tensor<double>([4]); },
            system1Forward: (_, _) => new Tensor<double>([2]),
            system2TicksValid: 5);

        for (int i = 0; i < 6; i++)
            runner.Step(new Tensor<double>([3]), "go");

        Assert.Equal(2, s2InvocationCount);
    }

    [Fact(Timeout = 30000)]
    public async Task Reset_ClearsCacheAndTickCounter()
    {
        await Task.Yield();
        int s2InvocationCount = 0;

        var runner = new HelixDualSystemRunner<double>(
            system2Forward: (_, _) => { s2InvocationCount++; return new Tensor<double>([4]); },
            system1Forward: (_, _) => new Tensor<double>([2]),
            system2TicksValid: 100);

        runner.Step(new Tensor<double>([3]), "go");
        Assert.Equal(1, s2InvocationCount);
        Assert.Equal(1, runner.CurrentTick);

        runner.Reset();

        Assert.Equal(0, runner.CurrentTick);
        Assert.Null(runner.CachedLatent);

        runner.Step(new Tensor<double>([3]), "go");
        Assert.Equal(2, s2InvocationCount);
    }

    [Fact(Timeout = 30000)]
    public async Task Rollout_RunsExactlyNStepsAndRespectsRatio()
    {
        await Task.Yield();
        int s2InvocationCount = 0;
        int s1InvocationCount = 0;

        var runner = new HelixDualSystemRunner<double>(
            system2Forward: (_, _) => { s2InvocationCount++; return new Tensor<double>([4]); },
            system1Forward: (_, _) => { s1InvocationCount++; return new Tensor<double>([2]); },
            system2TicksValid: 3);

        var actions = runner.Rollout(new Tensor<double>([3]), "go", numSteps: 9);

        Assert.Equal(9, actions.Length);
        Assert.Equal(9, s1InvocationCount);
        // S2 fires at tick 0, 3, 6 → 3 invocations.
        Assert.Equal(3, s2InvocationCount);
    }
}
