using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.CQL;
using AiDotNet.ReinforcementLearning.Agents.IQL;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Regression tests for issue #1724: the offline RL agents (CQL, IQL) shipped a stubbed
/// <c>Train()</c> that hardcoded its losses to zero and never updated any network — the agents
/// did not learn at all. These tests load a small offline dataset, run a few training steps, and
/// assert the agent's parameters actually change and stay finite.
/// </summary>
public class OfflineRLTrainingTests
{
    private const int StateDim = 4;
    private const int ActionDim = 2;
    private const int BatchSize = 32;

    private static List<(Vector<double> state, Vector<double> action, double reward, Vector<double> nextState, bool done)>
        BuildDataset(int count)
    {
        var rng = RandomHelper.CreateSeededRandom(7);
        var data = new List<(Vector<double>, Vector<double>, double, Vector<double>, bool)>(count);
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
            // Reward correlated with the first action component so there is something to learn.
            double reward = a[0] > 0.0 ? 1.0 : -1.0;
            data.Add((s, a, reward, ns, false));
        }
        return data;
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
    public async Task CQL_Train_UpdatesParameters()
    {
        await Task.Yield();
        var agent = new CQLAgent<double>(new CQLOptions<double>
        {
            StateSize = StateDim,
            ActionSize = ActionDim,
            BatchSize = BatchSize,
        });
        agent.LoadOfflineData(BuildDataset(BatchSize * 2));

        var before = agent.GetParameters();
        for (int step = 0; step < 10; step++) agent.Train();
        var after = agent.GetParameters();

        Assert.True(ParametersChangedAndFinite(before, after),
            "CQLAgent parameters did not change after training — Train() is not applying a gradient step.");
    }

    [Fact(Timeout = 60000)]
    public async Task IQL_Train_UpdatesParameters()
    {
        await Task.Yield();
        var agent = new IQLAgent<double>(new IQLOptions<double>
        {
            StateSize = StateDim,
            ActionSize = ActionDim,
            BatchSize = BatchSize,
        });
        agent.LoadOfflineData(BuildDataset(BatchSize * 2));

        var before = agent.GetParameters();
        for (int step = 0; step < 10; step++) agent.Train();
        var after = agent.GetParameters();

        Assert.True(ParametersChangedAndFinite(before, after),
            "IQLAgent parameters did not change after training — Train() is not applying a gradient step.");
    }
}
