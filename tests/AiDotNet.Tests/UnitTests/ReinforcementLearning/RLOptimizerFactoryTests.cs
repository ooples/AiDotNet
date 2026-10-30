using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;
using AiDotNet.ReinforcementLearning.Agents.DQN;
using AiDotNet.ReinforcementLearning.Agents.MADDPG;
using AiDotNet.ReinforcementLearning.Agents.QMIX;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// <c>OptimizerFactory</c> gives every network an agent trains its own optimizer (issue #2090). One shared instance
/// would share its step counter, so Adam's bias correction would advance on every network's step instead of only on
/// that network's own.
/// </summary>
/// <remarks>The QMIX test trains through the fused step, which shares global optimizer state with the other fused tests.</remarks>
[Collection("FusedOptimizerGlobalState")]
public class RLOptimizerFactoryTests
{
    /// <summary>A factory that records every optimizer it hands out.</summary>
    private sealed class CountingFactory
    {
        public List<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>> Created { get; } = new();

        public IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> Create()
        {
            var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 1e-3 });
            Created.Add(optimizer);
            return optimizer;
        }
    }

    /// <summary>Every network the agent registered, read from the base class's protected list.</summary>
    private static IReadOnlyList<INeuralNetwork<double>> NetworksOf(DeepReinforcementLearningAgentBase<double> agent) =>
        (typeof(DeepReinforcementLearningAgentBase<double>)
                .GetField("Networks", BindingFlags.NonPublic | BindingFlags.Instance)
                ?? throw new InvalidOperationException("DeepReinforcementLearningAgentBase has no Networks field."))
            .GetValue(agent) as List<INeuralNetwork<double>>
        ?? throw new InvalidOperationException("Networks is not a List<INeuralNetwork<double>>.");

    /// <summary>
    /// The factory ran once per network, and each network trains with the instance its own call returned: its
    /// <c>TrainingOptimizer</c>, which <c>Train</c> steps with.
    /// </summary>
    private static void AssertOneOptimizerPerNetwork(string agent, CountingFactory factory, IReadOnlyList<INeuralNetwork<double>> networks)
    {
        Assert.NotEmpty(networks);
        Assert.Equal(networks.Count, factory.Created.Count);
        var used = networks.Select(n => ((NeuralNetworkBase<double>)n).TrainingOptimizer).ToList();
        Assert.All(used, optimizer => Assert.Contains(factory.Created, created => ReferenceEquals(created, optimizer)));
        // ReferenceEqualityComparer is not available on net471, so compare pairwise.
        for (int i = 0; i < used.Count; i++)
            for (int j = i + 1; j < used.Count; j++)
                Assert.False(ReferenceEquals(used[i], used[j]), $"{agent}: networks {i} and {j} share one optimizer instance.");
    }

    [Fact]
    public void DQN_GivesTheQNetworkAndItsTargetTheirOwnOptimizers()
    {
        var factory = new CountingFactory();
        var agent = new DQNAgent<double>(new DQNOptions<double>
        {
            StateSize = 4,
            ActionSize = 2,
            HiddenLayers = new List<int> { 8 },
            OptimizerFactory = factory.Create,
        });
        AssertOneOptimizerPerNetwork("DQN", factory, NetworksOf(agent));
    }

    [Fact]
    public void MADDPG_GivesEveryActorAndCriticItsOwnOptimizer()
    {
        var factory = new CountingFactory();
        var agent = new MADDPGAgent<double>(new MADDPGOptions<double>
        {
            NumAgents = 2,
            StateSize = 4,
            ActionSize = 2,
            ActorHiddenLayers = new List<int> { 8 },
            CriticHiddenLayers = new List<int> { 8 },
            OptimizerFactory = factory.Create,
        });
        var networks = NetworksOf(agent);
        Assert.Equal(8, networks.Count); // actor, target actor, critic and target critic for each of two agents
        AssertOneOptimizerPerNetwork("MADDPG", factory, networks);
    }

    [Fact]
    public void DecisionTransformer_TrainsItsNetworkWithTheFactorysOptimizer()
    {
        var factory = new CountingFactory();
        var agent = new DecisionTransformerAgent<double>(new DecisionTransformerOptions<double>
        {
            StateSize = 4,
            ActionSize = 2,
            EmbeddingDim = 8,
            NumLayers = 1,
            OptimizerFactory = factory.Create,
        });
        AssertOneOptimizerPerNetwork("DecisionTransformer", factory, NetworksOf(agent));
    }

    [Fact]
    public void QMIX_BuildsOneJointOptimizerPerAgent_EvenFromSharedOptions()
    {
        // QMIX trains its agent networks and mixing network on one TD loss in one step, so each agent asks for
        // exactly one optimizer however many steps it takes; two agents built from the same options get one each.
        var factory = new CountingFactory();
        var options = new QMIXOptions<double>
        {
            NumAgents = 2,
            StateSize = 3,
            ActionSize = 2,
            GlobalStateSize = 1,
            BatchSize = 1,
            ReplayBufferSize = 1,
            AgentHiddenLayers = new List<int> { 4 },
            MixingHiddenLayers = new List<int> { 4 },
            OptimizerFactory = factory.Create,
        };

        var agents = new[] { new QMIXAgent<double>(options), new QMIXAgent<double>(options) };
        Assert.Empty(factory.Created);
        foreach (var agent in agents)
        {
            for (int step = 0; step < 2; step++)
            {
                agent.StoreMultiAgentExperience(
                    new List<Vector<double>> { State(step, 0), State(step, 1) },
                    new List<Vector<double>> { OneHot(step), OneHot(step + 1) },
                    teamReward: Math.Cos(step),
                    nextAgentStates: new List<Vector<double>> { State(step + 1, 0), State(step + 1, 1) },
                    globalState: new Vector<double>(new[] { 0.1 * step }),
                    nextGlobalState: new Vector<double>(new[] { 0.1 * (step + 1) }),
                    done: false);
                agent.Train();
            }
        }

        Assert.Equal(2, factory.Created.Count);
        Assert.NotSame(factory.Created[0], factory.Created[1]);
        // And each agent's joint step holds the optimizer the factory gave it, not a fallback SGD.
        var joint = typeof(QMIXAgent<double>).GetField("_jointOptimizer", BindingFlags.NonPublic | BindingFlags.Instance);
        Assert.NotNull(joint);
        Assert.Same(factory.Created[0], joint.GetValue(agents[0]));
        Assert.Same(factory.Created[1], joint.GetValue(agents[1]));
    }

    private static Vector<double> State(int step, int agent)
    {
        var v = new Vector<double>(3);
        for (int i = 0; i < 3; i++) v[i] = Math.Sin(0.7 * i + step + 1.3 * agent);
        return v;
    }

    private static Vector<double> OneHot(int index)
    {
        var v = new Vector<double>(2);
        v[index % 2] = 1.0;
        return v;
    }
}
