using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.QMIX;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// QMIX trains its agent network and its mixing network end to end on the shared fused step (#1804; Rashid et al.
/// 2018): the TD error of Q_tot reaches the agents through the mixer. Batches of episodes of different lengths unroll
/// to different horizons, so the fused plan must follow the batch's shape and train exactly as the eager tape does.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class QmixFusedTrainingTests
{
    private const double Tolerance = 1e-9;
    private const int Observation = 3, Actions = 2, State = 2;

    private static QMIXAgent<double> Make(bool doubleQ) => new(new QMIXOptions<double>
    {
        NumAgents = 2,
        StateSize = Observation,
        ActionSize = Actions,
        GlobalStateSize = State,
        RnnHiddenDim = 4,
        MixingEmbedDim = 4,
        HypernetLayers = doubleQ ? 2 : 1,
        HypernetEmbed = 3,
        DoubleQ = doubleQ,
        MeanLoss = doubleQ,
        BatchSize = 2,
        ReplayBufferSize = 3,
        TargetUpdateFrequency = 2,
        LearningRate = 0.01,
        Seed = 7,
    });

    private static Vector<double> Vec(int size, double phase)
        => new(Enumerable.Range(0, size).Select(i => Math.Sin(0.7 * i + phase)).ToArray());

    private static void Episode(QMIXAgent<double> agent, int episode, int length)
    {
        for (int t = 0; t < length; t++)
        {
            List<Vector<double>> Observations(int step) => new() { Vec(Observation, episode + step), Vec(Observation, episode + step + 1.3) };
            var actions = Enumerable.Range(0, 2).Select(i =>
            {
                var v = new Vector<double>(Actions);
                v[(episode + t + i) % Actions] = 1.0;
                return v;
            }).ToList();
            bool last = t == length - 1;
            agent.StoreMultiAgentExperience(Observations(t), actions, Math.Cos(episode + t), Observations(t + 1),
                Vec(State, 0.1 * (episode + t)), Vec(State, 0.1 * (episode + t + 1)), done: last && episode % 2 == 0, truncated: last && episode % 2 == 1);
        }
    }

    private static (Vector<double> Agent, Vector<double> Mixer) Train(QMIXAgent<double> agent, bool compile)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            int[] lengths = { 2, 3, 2, 4, 3, 3 };
            for (int episode = 0; episode < lengths.Length; episode++)
            {
                Episode(agent, episode, lengths[episode]);
                double loss = agent.Train();
                Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), $"episode {episode}: loss {loss}");
            }
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
        return agent.LiveNetworkParameters();
    }

    private static double Worst(Vector<double> a, Vector<double> b)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    [Theory(Timeout = 120000)]
    [InlineData(false)]
    [InlineData(true)]
    public async Task EndToEndTraining_MovesBothNetworks_AndRunsFusedLikeEager(bool doubleQ)
    {
        await Task.Yield();
        using var eagerAgent = Make(doubleQ);
        using var fusedAgent = Make(doubleQ);
        var before = eagerAgent.LiveNetworkParameters();
        Assert.Equal(0.0, Worst(before.Agent, fusedAgent.LiveNetworkParameters().Agent));

        var eager = Train(eagerAgent, compile: false);
        var fused = Train(fusedAgent, compile: true);

        Assert.True(Worst(eager.Agent, before.Agent) > 0, "the agent network did not move: the TD error never reached it.");
        Assert.True(Worst(eager.Mixer, before.Mixer) > 0, "the mixing network did not move.");
        double worst = Math.Max(Worst(eager.Agent, fused.Agent), Worst(eager.Mixer, fused.Mixer));
        Assert.True(worst <= Tolerance, $"fused QMIX training diverged from eager by {worst:G6}.");
    }
}
