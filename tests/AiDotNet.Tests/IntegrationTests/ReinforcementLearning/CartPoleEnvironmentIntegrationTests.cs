using System;
using AiDotNet.ReinforcementLearning.Environments;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class CartPoleEnvironmentIntegrationTests
{
    [Fact]
    public void Reset_ReturnsStateWithExpectedDimensionsAndRange()
    {
        var environment = new CartPoleEnvironment<double>(maxSteps: 5, seed: 123);

        var state = environment.Reset();

        Assert.Equal(environment.ObservationSpaceDimension, state.Length);
        for (int i = 0; i < state.Length; i++)
        {
            Assert.InRange(state[i], -0.1, 0.1);
        }
    }

    [Fact]
    public void Step_MaxStepsReached_EndsEpisodeAndReturnsZeroReward()
    {
        var environment = new CartPoleEnvironment<double>(maxSteps: 1, seed: 7);
        var action = new Vector<double>(environment.ActionSpaceSize);
        action[1] = 1.0;

        var (nextState, reward, done, info) = environment.Step(action);

        Assert.Equal(environment.ObservationSpaceDimension, nextState.Length);
        Assert.True(done);
        Assert.Equal(0.0, reward);
        Assert.True(info.ContainsKey("steps"));
        Assert.True(info.ContainsKey("x"));
        Assert.True(info.ContainsKey("theta"));
    }

    [Fact]
    public void Step_InvalidActionIndex_Throws()
    {
        var environment = new CartPoleEnvironment<double>(maxSteps: 5, seed: 5);
        var invalidAction = new Vector<double>(1);
        invalidAction[0] = 3.0;

        Assert.Throws<ArgumentException>(() => environment.Step(invalidAction));
    }
}
