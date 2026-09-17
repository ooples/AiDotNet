using AiDotNet.Evolution;
using AiDotNet.Evolve.Cli;
using Xunit;

namespace AiDotNet.Evolve.Cli.Tests;

public sealed class RunInterruptPolicyTests
{
    [Fact]
    public void RunFirstDrainsThenCancelsThenAllowsTermination()
    {
        var control = new EvolutionRunControl();
        using var cancellation = new CancellationTokenSource();
        using var error = new StringWriter();
        var policy = new RunInterruptPolicy(control, cancellation, error);
        Assert.True(policy.Interrupt());
        Assert.True(control.IsStopRequested);
        Assert.False(cancellation.IsCancellationRequested);
        Assert.Contains("durability is not yet confirmed", error.ToString());
        Assert.True(policy.Interrupt());
        Assert.True(cancellation.IsCancellationRequested);
        Assert.False(policy.Interrupt());
    }

    [Fact]
    public void NonRunCommandCancelsOnItsFirstInterrupt()
    {
        using var cancellation = new CancellationTokenSource();
        var policy = new RunInterruptPolicy(null, cancellation, TextWriter.Null);
        Assert.True(policy.Interrupt());
        Assert.True(cancellation.IsCancellationRequested);
        Assert.False(policy.Interrupt());
    }
}
