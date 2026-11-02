using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

/// <summary>
/// A deep RL agent's options Seed must fix its initial network weights: two agents built from options with the same
/// seed start identical, and a different seed starts them apart. Seed used to reach only the exploration and
/// sampling generator, so every seeded agent without its own per-layer seeding (PPO among them) started from fresh
/// random weights, and PPO's entropy-bonus comparison compared two different initialisations.
/// </summary>
public class DeepAgentSeedDeterminismTests
{
    public static IEnumerable<object[]> DeepAgentTypes() => DeepAgentDisposalTests.DeepAgentTypes();

    [Theory]
    [MemberData(nameof(DeepAgentTypes))]
    public void Same_seed_gives_the_same_initial_weights_and_another_seed_does_not(string typeName)
    {
        var first = Weights(typeName, seed: 7);
        var second = Weights(typeName, seed: 7);
        var other = Weights(typeName, seed: 8);

        Assert.True(first.Length > 0, $"{typeName} materialised no network weights.");
        Assert.True(first.SequenceEqual(second), $"{typeName}: two agents seeded 7 start from different weights.");
        Assert.False(first.SequenceEqual(other), $"{typeName}: seeds 7 and 8 start from the same weights.");
    }

    private static double[] Weights(string typeName, int seed)
    {
        var definition = typeof(DeepReinforcementLearningAgentBase<>).Assembly.GetTypes()
            .Single(t => t.IsGenericTypeDefinition && t.Name == typeName
                         && typeof(DeepReinforcementLearningAgentBase<double>).IsAssignableFrom(t.MakeGenericType(typeof(double))));
        var closed = definition.MakeGenericType(typeof(double));

        // The default agent's own options, re-seeded, rebuild the agent through its options constructor.
        using var template = Assert.IsAssignableFrom<DeepReinforcementLearningAgentBase<double>>(Activator.CreateInstance(closed));
        var options = template.GetOptions();
        options.Seed = seed;
        var constructor = closed.GetConstructors()
            .First(c => c.GetParameters().Length >= 1
                        && c.GetParameters()[0].ParameterType.IsInstanceOfType(options)
                        && c.GetParameters().Skip(1).All(p => p.IsOptional));
        var arguments = new object?[constructor.GetParameters().Length];
        arguments[0] = options;
        for (int i = 1; i < arguments.Length; i++) arguments[i] = Type.Missing;

        using var agent = Assert.IsAssignableFrom<DeepReinforcementLearningAgentBase<double>>(constructor.Invoke(arguments));
        // Materialise lazily-initialised layers with one forward pass before reading the weights.
        agent.Predict(new Vector<double>(agent.FeatureCount));
        return agent.GetParameters().ToArray();
    }
}
