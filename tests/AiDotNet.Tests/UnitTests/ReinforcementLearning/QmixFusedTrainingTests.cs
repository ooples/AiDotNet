using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents.QMIX;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// QMIX trains its agent networks and its mixing network end to end on the shared fused step (#1804; Rashid et al.
/// 2018). The agents' chosen Q-values used to be copied into the mixing input by host indexing, which cut the agents
/// off: only the mixing network trained. Each step's states, actions, global state and TD target travel in its data.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class QmixFusedTrainingTests
{
    private const double Tolerance = 1e-9;
    private const int StateSize = 3;
    private const int ActionSize = 2;

    private static QMIXAgent<double> Make() => new(new QMIXOptions<double>
    {
        NumAgents = 2,
        StateSize = StateSize,
        ActionSize = ActionSize,
        GlobalStateSize = 1,
        BatchSize = 1,
        // One slot: every Train call learns from the newest experience, so two agents fed alike train alike.
        ReplayBufferSize = 1,
        TargetUpdateFrequency = 2,
        LearningRate = 0.05,
        AgentHiddenLayers = new List<int> { 4 },
        MixingHiddenLayers = new List<int> { 4 },
    });

    private static Vector<double> State(int step, int agent)
    {
        var v = new Vector<double>(StateSize);
        for (int i = 0; i < StateSize; i++) v[i] = Math.Sin(0.7 * i + step + 1.3 * agent);
        return v;
    }

    private static Vector<double> Action(int step, int agent)
    {
        var v = new Vector<double>(ActionSize);
        v[(step + agent) % ActionSize] = 1.0;
        return v;
    }

    private static void Feed(QMIXAgent<double> agent, int step) => agent.StoreMultiAgentExperience(
        new List<Vector<double>> { State(step, 0), State(step, 1) },
        new List<Vector<double>> { Action(step, 0), Action(step, 1) },
        teamReward: Math.Cos(step),
        nextAgentStates: new List<Vector<double>> { State(step + 1, 0), State(step + 1, 1) },
        globalState: new Vector<double>(new[] { 0.1 * step }),
        nextGlobalState: new Vector<double>(new[] { 0.1 * (step + 1) }),
        done: false);

    private static Vector<double> AgentNetworkParameters(QMIXAgent<double> agent)
    {
        var networks = (List<INeuralNetwork<double>>)typeof(QMIXAgent<double>)
            .GetField("_agentNetworks", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(agent)!;
        return new Vector<double>(networks.SelectMany(n => n.GetParameters().ToArray()).ToArray());
    }

    private static (Vector<double> All, Vector<double> Agents) Train(QMIXAgent<double> agent, bool compile)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            for (int step = 0; step < 4; step++)
            {
                Feed(agent, step);
                agent.Train();
            }
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
        return (agent.GetParameters(), AgentNetworkParameters(agent));
    }

    private static double Worst(Vector<double> a, Vector<double> b)
    {
        Assert.Equal(a.Length, b.Length);
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    [Fact(Timeout = 120000)]
    public async Task EndToEndTraining_MovesTheAgentNetworks_AndRunsFusedLikeEager()
    {
        await Task.Yield();
        var eagerAgent = Make();
        var fusedAgent = Make();
        fusedAgent.SetParameters(eagerAgent.GetParameters());
        var agentsBefore = AgentNetworkParameters(eagerAgent);

        var eager = Train(eagerAgent, compile: false);
        var fused = Train(fusedAgent, compile: true);

        Assert.True(Worst(eager.Agents, agentsBefore) > 0,
            "the agent networks did not move: the TD error never reached them.");
        double worst = Worst(eager.All, fused.All);
        Assert.True(worst <= Tolerance, $"fused QMIX training diverged from eager by {worst:G6}.");
    }
}
