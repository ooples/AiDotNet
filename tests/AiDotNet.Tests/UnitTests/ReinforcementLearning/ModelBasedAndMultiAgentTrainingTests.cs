using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.Dreamer;
using AiDotNet.ReinforcementLearning.Agents.MADDPG;
using AiDotNet.ReinforcementLearning.Agents.MuZero;
using AiDotNet.ReinforcementLearning.Agents.WorldModels;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Regression tests for issue #1727: MADDPG, Dreamer, World Models, and MuZero shipped a stubbed
/// <c>Train()</c> whose losses were hardcoded to zero so their networks were never updated.
/// These tests fill each agent's buffer with the kind of transition it expects, run a few
/// training steps, and assert the agent's parameters actually change and stay finite.
/// </summary>
public class ModelBasedAndMultiAgentTrainingTests
{
    private static Vector<double> Rand(Random rng, int dim)
    {
        var v = new Vector<double>(dim);
        for (int i = 0; i < dim; i++) v[i] = rng.NextDouble() * 2.0 - 1.0;
        return v;
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
    public async Task MADDPG_Train_UpdatesParameters()
    {
        await Task.Yield();
        const int numAgents = 2, stateDim = 4, actionDim = 2, batch = 8;
        var agent = new MADDPGAgent<double>(new MADDPGOptions<double>
        {
            NumAgents = numAgents,
            StateSize = stateDim,
            ActionSize = actionDim,
            BatchSize = batch,
            WarmupSteps = 0,
        });
        var rng = RandomHelper.CreateSeededRandom(3);
        for (int t = 0; t < batch * 2; t++)
        {
            var states = new List<Vector<double>>();
            var actions = new List<Vector<double>>();
            var rewards = new List<double>();
            var nextStates = new List<Vector<double>>();
            for (int a = 0; a < numAgents; a++)
            {
                var act = Rand(rng, actionDim);
                states.Add(Rand(rng, stateDim));
                actions.Add(act);
                rewards.Add(act[0] > 0 ? 1.0 : -1.0);
                nextStates.Add(Rand(rng, stateDim));
            }
            agent.StoreMultiAgentExperience(states, actions, rewards, nextStates, false);
        }

        var before = agent.GetParameters();
        for (int step = 0; step < 10; step++) agent.Train();
        var after = agent.GetParameters();

        Assert.True(ParametersChangedAndFinite(before, after),
            "MADDPGAgent parameters did not change after training — Train() is not applying a gradient step.");
    }

    [Fact(Timeout = 60000)]
    public async Task Dreamer_Train_UpdatesParameters()
    {
        await Task.Yield();
        const int obs = 4, actionDim = 2, batch = 8;
        var agent = new DreamerAgent<double>(new DreamerOptions<double>
        {
            ObservationSize = obs,
            ActionSize = actionDim,
            LatentSize = 16,
            ImaginationHorizon = 3,
            BatchSize = batch,
        });
        var rng = RandomHelper.CreateSeededRandom(5);
        for (int t = 0; t < batch * 2; t++)
        {
            var a = Rand(rng, actionDim);
            agent.StoreExperience(Rand(rng, obs), a, a[0] > 0 ? 1.0 : -1.0, Rand(rng, obs), false);
        }

        var before = agent.GetParameters();
        for (int step = 0; step < 5; step++) agent.Train();
        var after = agent.GetParameters();

        Assert.True(ParametersChangedAndFinite(before, after),
            "DreamerAgent parameters did not change after training — Train() is not applying a gradient step.");
    }

    [Fact(Timeout = 60000)]
    public async Task WorldModels_Train_UpdatesParameters()
    {
        await Task.Yield();
        const int obs = 4, actionDim = 2, batch = 8;
        var agent = new WorldModelsAgent<double>(new WorldModelsOptions<double>
        {
            ObservationWidth = obs,
            ObservationHeight = 1,
            ObservationChannels = 1,
            ActionSize = actionDim,
            LatentSize = 8,
            RNNHiddenSize = 8,
            BatchSize = batch,
        });
        var rng = RandomHelper.CreateSeededRandom(9);
        for (int t = 0; t < batch * 2; t++)
        {
            var a = Rand(rng, actionDim);
            agent.StoreExperience(Rand(rng, obs), a, a[0] > 0 ? 1.0 : -1.0, Rand(rng, obs), false);
        }

        var before = agent.GetParameters();
        for (int step = 0; step < 5; step++) agent.Train();
        var after = agent.GetParameters();

        Assert.True(ParametersChangedAndFinite(before, after),
            "WorldModelsAgent parameters did not change after training — Train() is not applying a gradient step.");
    }

    [Fact(Timeout = 60000)]
    public async Task MuZero_Train_UpdatesParameters()
    {
        await Task.Yield();
        const int obs = 4, actionDim = 2, batch = 8;
        var agent = new MuZeroAgent<double>(new MuZeroOptions<double>
        {
            ObservationSize = obs,
            ActionSize = actionDim,
            LatentStateSize = 16,
            BatchSize = batch,
        });
        var rng = RandomHelper.CreateSeededRandom(13);
        for (int t = 0; t < batch * 2; t++)
        {
            var a = new Vector<double>(actionDim);
            a[rng.Next(actionDim)] = 1.0; // one-hot action
            agent.StoreExperience(Rand(rng, obs), a, rng.NextDouble(), Rand(rng, obs), false);
        }

        var before = agent.GetParameters();
        for (int step = 0; step < 5; step++) agent.Train();
        var after = agent.GetParameters();

        Assert.True(ParametersChangedAndFinite(before, after),
            "MuZeroAgent parameters did not change after training — Train() builds gradients but never applies them.");
    }

    [Fact(Timeout = 60000)]
    public async Task MuZero_MultiStepUnroll_UpdatesParametersAcrossEpisodeBoundaries()
    {
        await Task.Yield();
        // Paper-faithful K-step unroll (#1756): Train() must unroll the learned model K>1 steps and
        // backprop one joint loss through the recurrence — updating all three networks — without
        // throwing, including when the sampled trajectory window CONTAINS an episode boundary. Note
        // SampleSequence truncates at the first Done, so the window ends at the terminal transition
        // rather than spanning across it into the next episode.
        const int obs = 4, actionDim = 2, batch = 8;
        var agent = new MuZeroAgent<double>(new MuZeroOptions<double>
        {
            ObservationSize = obs,
            ActionSize = actionDim,
            LatentStateSize = 12,
            BatchSize = batch,
            UnrollSteps = 4,   // K = 4 recurrent unroll
            TDSteps = 3,
        });
        var rng = RandomHelper.CreateSeededRandom(7);
        for (int t = 0; t < batch * 4; t++)
        {
            var a = new Vector<double>(actionDim);
            a[rng.Next(actionDim)] = 1.0; // one-hot action
            agent.StoreExperience(Rand(rng, obs), a, rng.NextDouble(), Rand(rng, obs), done: t % 11 == 10);
        }

        var before = agent.GetParameters();
        for (int step = 0; step < 5; step++) agent.Train(); // K=4 unroll must not throw
        var after = agent.GetParameters();

        Assert.True(ParametersChangedAndFinite(before, after),
            "MuZero K=4 unrolled training did not update parameters — the joint recurrent unroll backprop is not flowing.");
    }
}
