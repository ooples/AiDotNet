using System;
using System.Collections.Generic;
using AiDotNet.Finance.Trading.Agents;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Two financial agents built from the same options Seed and trained on the same environment must produce
/// bit-identical training: the same action sequence and the same final parameters. Every stochastic choice
/// (exploration, sampling, target-network scheduling, default weight initialization) has to come from the
/// seed — an unseeded draw anywhere turns an agent-vs-agent bake-off into a comparison of RNG luck.
/// </summary>
public sealed class FinancialAgentDeterminismTests
{
    private const int Seed = 1234;

    [Theory]
    [InlineData(Ppo)]
    [InlineData(Dqn)]
    [InlineData(A2C)]
    [InlineData(Sac)]
    [InlineData(MarketMaking)]
    [Trait("category", "unit")]
    public void Same_seed_produces_bit_identical_training(string kind)
    {
        var first = Run(kind);
        var second = Run(kind);

        Assert.Equal(first.Actions.Count, second.Actions.Count);
        for (int i = 0; i < first.Actions.Count; i++)
        {
            Assert.True(first.Actions[i] == second.Actions[i],
                $"{kind}: step {i} action differs between identically seeded runs ({first.Actions[i]} vs {second.Actions[i]}).");
        }

        Assert.Equal(first.Parameters.Length, second.Parameters.Length);
        for (int i = 0; i < first.Parameters.Length; i++)
        {
            Assert.True(BitConverter.DoubleToInt64Bits(first.Parameters[i]) == BitConverter.DoubleToInt64Bits(second.Parameters[i]),
                $"{kind}: parameter {i} differs between identically seeded runs ({first.Parameters[i]:R} vs {second.Parameters[i]:R}).");
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Different_seeds_produce_different_exploration()
    {
        // Guard against "deterministic" meaning "not random at all".
        var a = Run(Dqn, seed: 1);
        var b = Run(Dqn, seed: 2);
        Assert.NotEqual(string.Join("|", a.Actions), string.Join("|", b.Actions));
    }

    private static (List<string> Actions, double[] Parameters) Run(string kind, int seed = Seed)
    {
        var env = ZigZagEnvironment();
        var options = Options(kind, env.ObservationSpaceDimension, actionSize: 3, seed);
        options.BatchSize = 8;
        options.WarmupSteps = 0;
        options.ReplayBufferSize = 1000;
        options.TargetUpdateFrequency = 5;
        options.EpsilonStart = 0.5;
        options.EpsilonEnd = 0.05;
        options.EpsilonDecay = 0.9;
        options.LearningRate = 0.001;

        using var agent = Create(kind, options);
        var actions = new List<string>();
        for (int episode = 0; episode < 2; episode++)
        {
            var state = env.Reset();
            for (int step = 0; step < 40; step++)
            {
                var action = agent.SelectAction(state, training: true);
                actions.Add(Format(action));
                var (next, reward, done, _) = env.Step(action);
                agent.StoreExperience(state, action, reward, next, done);
                agent.Train();
                state = next;
                if (done)
                {
                    break;
                }
            }
        }

        var parameters = agent.GetParameters();
        var copy = new double[parameters.Length];
        for (int i = 0; i < copy.Length; i++)
        {
            copy[i] = parameters[i];
        }

        return (actions, copy);
    }

    private static string Format(Vector<double> action)
    {
        var parts = new string[action.Length];
        for (int i = 0; i < action.Length; i++)
        {
            parts[i] = BitConverter.DoubleToInt64Bits(action[i]).ToString("X16");
        }

        return string.Join(",", parts);
    }
}
