using System;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.DDPG;
using AiDotNet.ReinforcementLearning.Agents.TD3;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Regression tests for issue #1727: the deterministic actor-critic agents (DDPG, TD3) shipped a
/// stubbed <c>Train()</c> that hardcoded its losses to zero and never updated the actor or critic
/// networks — the agents did not learn. These tests fill a small replay buffer, run a few training
/// steps, and assert the agent's parameters actually change and stay finite.
/// </summary>
public class ContinuousControlTrainingTests
{
    private const int StateDim = 4;
    private const int ActionDim = 2;
    private const int BatchSize = 16;

    private static void FillBuffer(Action<Vector<double>, Vector<double>, double, Vector<double>, bool> store, int count)
    {
        var rng = RandomHelper.CreateSeededRandom(11);
        for (int i = 0; i < count; i++)
        {
            var s = new Vector<double>(StateDim);
            var ns = new Vector<double>(StateDim);
            for (int j = 0; j < StateDim; j++)
            {
                s[j] = rng.NextDouble() * 2.0 - 1.0;
                ns[j] = rng.NextDouble() * 2.0 - 1.0;
            }
            var a = new Vector<double>(ActionDim);
            for (int k = 0; k < ActionDim; k++) a[k] = rng.NextDouble() * 2.0 - 1.0;
            double reward = a[0] > 0.0 ? 1.0 : -1.0;
            store(s, a, reward, ns, false);
        }
    }

    private static bool ParametersChangedAndFinite(Vector<double> before, Vector<double> after)
    {
        Assert.Equal(before.Length, after.Length);
        bool anyChanged = false;
        for (int i = 0; i < after.Length; i++)
        {
            Assert.False(double.IsNaN(after[i]), $"Parameter {i} became NaN after training.");
            Assert.False(double.IsInfinity(after[i]), $"Parameter {i} became infinite after training.");
            if (Math.Abs(before[i] - after[i]) > 1e-9) anyChanged = true;
        }
        return anyChanged;
    }

    [Fact(Timeout = 60000)]
    public async Task DDPG_Train_UpdatesParameters()
    {
        await Task.Yield();
        var agent = new DDPGAgent<double>(new DDPGOptions<double>
        {
            StateSize = StateDim,
            ActionSize = ActionDim,
            BatchSize = BatchSize,
            WarmupSteps = 0,
        });
        FillBuffer(agent.StoreExperience, BatchSize * 2);

        var before = agent.GetParameters();
        for (int step = 0; step < 10; step++) agent.Train();
        var after = agent.GetParameters();

        Assert.True(ParametersChangedAndFinite(before, after),
            "DDPGAgent parameters did not change after training — Train() is not applying a gradient step.");
    }

    [Fact(Timeout = 60000)]
    public async Task TD3_Train_UpdatesParameters()
    {
        await Task.Yield();
        var agent = new TD3Agent<double>(new TD3Options<double>
        {
            StateSize = StateDim,
            ActionSize = ActionDim,
            BatchSize = BatchSize,
            WarmupSteps = 0,
        });
        FillBuffer(agent.StoreExperience, BatchSize * 2);

        var before = agent.GetParameters();
        for (int step = 0; step < 10; step++) agent.Train();
        var after = agent.GetParameters();

        Assert.True(ParametersChangedAndFinite(before, after),
            "TD3Agent parameters did not change after training — Train() is not applying a gradient step.");
    }
}
