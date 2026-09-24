using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.ReinforcementLearning.Agents;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// Disposing any deep RL agent must release every network it owns, each exactly once.
///
/// <para><see cref="DeepReinforcementLearningAgentBase{T}.Dispose"/> releases what the agent registered in
/// <c>Networks</c>, and nothing else. CQL, IQL and TD3 never registered theirs, and DuelingDQN's networks are
/// not <see cref="INeuralNetwork{T}"/> at all, so disposing those agents released nothing. QMIX built its
/// mixing networks twice and dropped the first pair unregistered.</para>
///
/// <para>The agents are discovered by reflection over the library, so a new agent is covered without editing
/// this file, and each agent's networks are found by reflecting over its own fields, so a network field that
/// is never registered fails here.</para>
/// </summary>
public class DeepAgentDisposalTests
{
    public static IEnumerable<object[]> DeepAgentTypes()
        => typeof(DeepReinforcementLearningAgentBase<>).Assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition && t.GetGenericArguments().Length == 1)
            .Where(t =>
            {
                try { return typeof(DeepReinforcementLearningAgentBase<double>).IsAssignableFrom(t.MakeGenericType(typeof(double))); }
                catch (ArgumentException) { return false; } // generic constraint not satisfied by double
            })
            .OrderBy(t => t.FullName, StringComparer.Ordinal)
            .Select(t => new object[] { t.Name });

    private static DeepReinforcementLearningAgentBase<double> Create(string typeName)
    {
        var definition = typeof(DeepReinforcementLearningAgentBase<>).Assembly.GetTypes()
            .Single(t => t.IsGenericTypeDefinition && t.Name == typeName
                         && typeof(DeepReinforcementLearningAgentBase<double>).IsAssignableFrom(t.MakeGenericType(typeof(double))));
        return Assert.IsAssignableFrom<DeepReinforcementLearningAgentBase<double>>(
            Activator.CreateInstance(definition.MakeGenericType(typeof(double))));
    }

    [Fact]
    public void Discovery_finds_every_deep_agent()
    {
        // Guard against the theory silently running over nothing.
        var names = DeepAgentTypes().Select(row => (string)row[0]).ToList();
        Assert.True(names.Count >= 20, $"Only {names.Count} deep agents discovered: {string.Join(", ", names)}");
        foreach (var expected in new[] { "CQLAgent`1", "IQLAgent`1", "TD3Agent`1", "MADDPGAgent`1", "QMIXAgent`1", "DuelingDQNAgent`1" })
        {
            Assert.Contains(expected, names);
        }
    }

    [Theory]
    [MemberData(nameof(DeepAgentTypes))]
    public void Every_network_field_is_registered_exactly_once(string typeName)
    {
        var agent = Create(typeName);
        var registered = RegisteredNetworks(agent);

        var duplicates = registered.GroupBy(n => n, ReferenceComparer.Instance).Where(g => g.Count() > 1).ToList();
        Assert.True(duplicates.Count == 0, $"{typeName} registers {duplicates.Count} network(s) more than once.");

        foreach (var (fieldName, network) in NetworkFields(agent))
        {
            Assert.True(
                registered.Any(r => ReferenceEquals(r, network)),
                $"{typeName}.{fieldName} holds a {network.GetType().Name} that is not registered in Networks, " +
                "so disposing the agent never releases it.");
        }
    }

    [Theory]
    [MemberData(nameof(DeepAgentTypes))]
    public void Dispose_releases_every_layer_the_agent_owns(string typeName)
    {
        var agent = Create(typeName);
        var layers = OwnedLayers(agent);
        Assert.NotEmpty(layers);

        agent.Dispose();

        for (var i = 0; i < layers.Count; i++)
        {
            var (owner, layer) = layers[i];
            // False means the layer was already released through the once-only guard -- what a network's
            // (or a DuelingNetwork's) Dispose cascade does. True means this probe released it first.
            Assert.False(
                DisposeOnceGuard.TryDispose(layer),
                $"{typeName}: a {layer.GetType().Name} owned via {owner} was still live after the agent was disposed.");
        }
    }

    [Theory]
    [MemberData(nameof(DeepAgentTypes))]
    public void Disposing_an_agent_twice_is_harmless(string typeName)
    {
        var agent = Create(typeName);
        agent.Dispose();
        Assert.Null(Record.Exception(() => agent.Dispose()));
    }

    private static List<INeuralNetwork<double>> RegisteredNetworks(DeepReinforcementLearningAgentBase<double> agent)
    {
        var value = typeof(DeepReinforcementLearningAgentBase<double>)
            .GetField("Networks", BindingFlags.Instance | BindingFlags.NonPublic)
            ?.GetValue(agent);
        return Assert.IsAssignableFrom<IEnumerable<INeuralNetwork<double>>>(value).ToList();
    }

    /// <summary>Every network held by a field declared on the concrete agent, directly or in a collection.</summary>
    private static IEnumerable<(string Field, INeuralNetwork<double> Network)> NetworkFields(object agent)
    {
        for (var type = agent.GetType(); type is not null && type != typeof(DeepReinforcementLearningAgentBase<double>); type = type.BaseType)
        {
            foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                switch (field.GetValue(agent))
                {
                    case INeuralNetwork<double> network:
                        yield return (field.Name, network);
                        break;
                    case IEnumerable<INeuralNetwork<double>> networks:
                        foreach (var n in networks.Where(n => n is not null)) yield return (field.Name + "[]", n);
                        break;
                }
            }
        }
    }

    /// <summary>
    /// Every layer the agent owns: the layers of each network it holds, and the layers of any other
    /// parameter-owning component (DuelingDQN's DuelingNetwork holds its DenseLayers directly).
    /// </summary>
    internal static List<(string Owner, IDisposable Layer)> OwnedLayers(object agent)
    {
        var found = new List<(string, IDisposable)>();
        var visited = new List<object>();
        Walk(agent, agent.GetType().Name, found, visited, depth: 0);
        return found;
    }

    private static void Walk(object owner, string path, List<(string, IDisposable)> found, List<object> visited, int depth)
    {
        if (depth > 4 || visited.Any(v => ReferenceEquals(v, owner))) return;
        visited.Add(owner);

        for (var type = owner.GetType(); type is not null && type != typeof(object); type = type.BaseType)
        {
            foreach (var field in type.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
            {
                Visit(field.GetValue(owner), path + "." + field.Name, found, visited, depth);
            }
        }
    }

    private static void Visit(object? value, string path, List<(string, IDisposable)> found, List<object> visited, int depth)
    {
        switch (value)
        {
            case null:
            case string:
                return;
            case NeuralNetworkBase<double> network:
                foreach (var layer in network.Layers) Add(layer, path, found);
                return;
            case ILayer<double> layer:
                Add(layer, path, found);
                return;
            case IEnumerable sequence when value is not ReinforcementLearningAgentBase<double>:
                var index = 0;
                foreach (var item in sequence) Visit(item, $"{path}[{index++}]", found, visited, depth + 1);
                return;
            case IParameterSource<double> component when value is not ReinforcementLearningAgentBase<double>:
                Walk(component, path, found, visited, depth + 1);
                return;
        }
    }

    private static void Add(ILayer<double> layer, string path, List<(string, IDisposable)> found)
    {
        if (layer is IDisposable disposable && !found.Any(f => ReferenceEquals(f.Item2, disposable)))
        {
            found.Add((path, disposable));
        }
    }

    private sealed class ReferenceComparer : IEqualityComparer<object>
    {
        public static readonly ReferenceComparer Instance = new();
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }
}
