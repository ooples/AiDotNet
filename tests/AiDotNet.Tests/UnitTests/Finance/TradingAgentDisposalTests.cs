using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Disposing a trading agent must release every network it built.
///
/// <para>The trading agents derive from <see cref="ReinforcementLearningAgentBase{T}"/> directly rather than
/// from <see cref="DeepReinforcementLearningAgentBase{T}"/>, and the plain base's <c>Dispose</c> only suppressed
/// finalization. So a disposed PPO/DQN/A2C/SAC agent left its policy, value, Q and target networks — their
/// pooled weight buffers, GPU allocations and compiled training plans — alive until the garbage collector
/// ran, while the caller believed it had released them.</para>
///
/// <para>The networks are found by reflecting over the agent's own fields rather than through any list the
/// agent maintains, so the test also fails if a future network field is added and never registered for
/// disposal. A network's release is observed on its layers: <see cref="NeuralNetworkBase{T}"/> cascades
/// disposal into each layer through <see cref="DisposeOnceGuard"/>, so the guard refusing a layer means
/// that layer was already released by its network.</para>
/// </summary>
public class TradingAgentDisposalTests
{
    private const int StateSize = 4;
    private const int ActionSize = 3;

    private static NeuralNetworkArchitecture<double> Arch(int inputs, int outputs)
        => new(inputFeatures: inputs, outputSize: outputs);

    public static IEnumerable<object[]> Agents()
    {
        yield return new object[] { "PPO" };
        yield return new object[] { "DQN" };
        yield return new object[] { "A2C" };
        yield return new object[] { "SAC" };
        yield return new object[] { "MarketMaking" };
        yield return new object[] { "FinRL-DQN" };
        yield return new object[] { "FinRL-PPO" };
    }

    private static TradingAgentBase<double> Create(string kind) => kind switch
    {
        "PPO" => new FinancialPPOAgent<double>(
            Arch(StateSize, ActionSize), Arch(StateSize, 1),
            new FinancialPPOAgentOptions<double> { StateSize = StateSize, ActionSize = ActionSize, ContinuousActions = false }),
        "DQN" => new FinancialDQNAgent<double>(
            Arch(StateSize, ActionSize),
            new FinancialDQNAgentOptions<double> { StateSize = StateSize, ActionSize = ActionSize }),
        "A2C" => new FinancialA2CAgent<double>(
            Arch(StateSize, ActionSize), Arch(StateSize, 1),
            new FinancialA2CAgentOptions<double> { StateSize = StateSize, ActionSize = ActionSize }),
        // SAC's critics score a (state, action) pair.
        "SAC" => new FinancialSACAgent<double>(
            Arch(StateSize, ActionSize), Arch(StateSize + ActionSize, 1),
            new FinancialSACAgentOptions<double> { StateSize = StateSize, ActionSize = ActionSize }),
        "MarketMaking" => new MarketMakingAgent<double>(
            new MarketMakingOptions<double> { StateSize = StateSize, ActionSize = ActionSize }),
        "FinRL-DQN" => new FinRLAgent<double>(
            Arch(StateSize, ActionSize),
            new TradingAgentOptions<double> { StateSize = StateSize, ActionSize = ActionSize },
            FinRLAlgorithm.DQN),
        "FinRL-PPO" => new FinRLAgent<double>(
            Arch(StateSize, ActionSize),
            new TradingAgentOptions<double> { StateSize = StateSize, ActionSize = ActionSize },
            FinRLAlgorithm.PPO,
            Arch(StateSize, 1)),
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null),
    };

    /// <summary>
    /// Every network reachable from the agent's instance fields, including the networks of an agent it wraps
    /// (FinRL delegates to an inner DQN/PPO/A2C/SAC agent). Distinct by reference.
    /// </summary>
    private static List<INeuralNetwork<double>> OwnedNetworks(object agent)
    {
        var found = new List<INeuralNetwork<double>>();
        Collect(agent, found, new List<object>());
        return found;
    }

    private static void Collect(object owner, List<INeuralNetwork<double>> found, List<object> visited)
    {
        if (visited.Any(v => ReferenceEquals(v, owner))) return;
        visited.Add(owner);

        for (var type = owner.GetType(); type is not null && type != typeof(object); type = type.BaseType)
        {
            foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                var value = field.GetValue(owner);
                switch (value)
                {
                    case INeuralNetwork<double> network:
                        if (!found.Any(n => ReferenceEquals(n, network))) found.Add(network);
                        break;
                    case IEnumerable<INeuralNetwork<double>> networks:
                        foreach (var n in networks)
                        {
                            if (n is not null && !found.Any(f => ReferenceEquals(f, n))) found.Add(n);
                        }
                        break;
                    case ReinforcementLearningAgentBase<double> inner:
                        Collect(inner, found, visited);
                        break;
                }
            }
        }
    }

    private static List<ILayer<double>> LayersOf(INeuralNetwork<double> network)
    {
        var networkBase = Assert.IsAssignableFrom<NeuralNetworkBase<double>>(network);
        return networkBase.Layers.ToList();
    }

    [Theory]
    [MemberData(nameof(Agents))]
    public void Dispose_releases_every_network_the_agent_owns(string kind)
    {
        var agent = Create(kind);
        var networks = OwnedNetworks(agent);

        // Guard against a vacuous pass: each agent really does own networks with layers to release.
        Assert.NotEmpty(networks);
        foreach (var network in networks)
        {
            Assert.NotEmpty(LayersOf(network));
        }

        agent.Dispose();

        foreach (var network in networks)
        {
            var layers = LayersOf(network);
            for (var i = 0; i < layers.Count; i++)
            {
                // TryDispose returns false only when the layer was already released through the guard,
                // which is exactly what a network's own Dispose cascade does. Returning true means this
                // probe was the first to dispose the layer: the agent never released its network.
                var layer = Assert.IsAssignableFrom<IDisposable>(layers[i]);
                Assert.False(
                    DisposeOnceGuard.TryDispose(layer),
                    $"{kind}: layer {i} ({layers[i].GetType().Name}) of a {network.GetType().Name} owned by " +
                    $"{agent.GetType().Name} was still live after the agent was disposed.");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Agents))]
    public void Disposing_an_agent_twice_is_harmless(string kind)
    {
        var agent = Create(kind);

        agent.Dispose();
        var second = Record.Exception(() => agent.Dispose());

        Assert.Null(second);
    }
}
