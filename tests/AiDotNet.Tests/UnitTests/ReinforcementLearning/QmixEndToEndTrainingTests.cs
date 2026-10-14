using System.Collections.Generic;
using System.Linq;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.QMIX;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// QMIX trains the agent networks and the mixing network end to end (Rashid et al. 2018): the TD loss on the mixed
/// team value must reach every agent's Q-network.
/// </summary>
/// <remarks>
/// The step used to copy each agent's chosen-action Q-value into the mixer input through the tensor indexer. That copy
/// records nothing on the gradient tape, so the agents received no gradient and only the mixing network trained; the
/// agent-agnostic loss still fell, which is why nothing noticed.
/// </remarks>
public class QmixEndToEndTrainingTests
{
    [Fact]
    public void Train_UpdatesEveryAgentNetwork_NotOnlyTheMixer()
    {
        var agent = new QMIXAgent<double>(new QMIXOptions<double>
        {
            NumAgents = 2,
            StateSize = 2,
            ActionSize = 3,
            GlobalStateSize = 1,
            BatchSize = 1,
            ReplayBufferSize = 4,
            TargetUpdateFrequency = 100,
            EpsilonStart = 0.0,
            EpsilonEnd = 0.0,
            EpsilonDecay = 1.0,
            LearningRate = 0.05,
            DiscountFactor = 0.9,
            LossFunction = new MeanSquaredErrorLoss<double>(),
            AgentHiddenLayers = new List<int> { 4 },
            MixingHiddenLayers = new List<int> { 4 },
            Seed = 7,
        });

        var states = new List<Vector<double>> { Row(0.1, 0.2), Row(0.3, 0.4) };
        var actions = new List<Vector<double>>
        {
            agent.SelectActionForAgent(0, states[0], training: true),
            agent.SelectActionForAgent(1, states[1], training: true),
        };
        agent.StoreMultiAgentExperience(
            states,
            actions,
            teamReward: 5.0,
            nextAgentStates: new List<Vector<double>> { Row(0.2, 0.1), Row(0.4, 0.3) },
            globalState: Row(0.5),
            nextGlobalState: Row(0.6),
            done: true);

        var before = AgentNetworkValues(agent);
        Assert.Equal(2, before.Count);

        agent.Train();

        var after = AgentNetworkValues(agent);
        foreach (var (id, values) in before)
        {
            double moved = values.Zip(after[id], (a, b) => System.Math.Abs(a - b)).Max();
            Assert.True(moved > 0.0, $"agent network {id} received no update: the TD loss never reached it");
        }
    }

    private static Vector<double> Row(params double[] values)
    {
        var row = new Vector<double>(values.Length);
        for (int i = 0; i < values.Length; i++) row[i] = values[i];
        return row;
    }

    // One entry per online agent network, keyed by its index segment; target networks are excluded.
    private static Dictionary<string, double[]> AgentNetworkValues(QMIXAgent<double> agent)
    {
        var byNetwork = new Dictionary<string, List<double>>();
        foreach (var chunk in agent.GetParameterStateChunks())
        {
            const string marker = "QMIXAgent<T>::_agentNetworks";
            int at = chunk.StableId.IndexOf(marker, System.StringComparison.Ordinal);
            if (at < 0) continue;
            string rest = chunk.StableId.Substring(at + marker.Length);
            string network = rest.Split('/', '.', ':').FirstOrDefault(part => part.Length > 0) ?? rest;
            if (!byNetwork.TryGetValue(network, out var values)) byNetwork[network] = values = new List<double>();
            values.AddRange(chunk.Tensor.ToArray());
        }
        return byNetwork.ToDictionary(pair => pair.Key, pair => pair.Value.ToArray());
    }
}
