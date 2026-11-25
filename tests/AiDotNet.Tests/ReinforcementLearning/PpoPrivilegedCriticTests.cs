using System;
using System.Linq;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ReinforcementLearning;

/// <summary>
/// FinancialPPOAgent with an asymmetric (privileged) critic: the critic sees the observation plus values the actor never
/// sees (Pinto et al. 2018). With no privileged inputs the agent is unchanged (PpoReferenceTests pin that path to
/// Baselines' ppo1).
/// </summary>
public class PpoPrivilegedCriticTests
{
    private static FinancialPPOAgentOptions<double> Options(int privileged, int seed = 7) => new()
    {
        StateSize = 2,
        ActionSize = 2,
        ContinuousActions = false,
        HiddenLayers = [16],
        RolloutLength = 64,
        MiniBatchSize = 16,
        NumEpochs = 4,
        LearningRate = 3e-3,
        EntropyCoefficient = 0.0,
        ValueCoefficient = 1.0,
        DiscountFactor = 0.0,
        GAELambda = 0.95,
        PrivilegedCriticInputSize = privileged,
        Seed = seed,
    };

    private static FinancialPPOAgent<double> Agent(FinancialPPOAgentOptions<double> o, int criticInputs) =>
        new(new NeuralNetworkArchitecture<double>(inputFeatures: o.StateSize, outputSize: o.ActionSize),
            new NeuralNetworkArchitecture<double>(inputFeatures: criticInputs, outputSize: 1), o);

    [Fact]
    [Trait("category", "unit")]
    public void Privileged_and_plain_storage_paths_refuse_each_other_and_shapes_are_checked()
    {
        var state = new Vector<double>([1.0, 0.0]);
        using (var plain = Agent(Options(0), 2))
        {
            var action = plain.SelectAction(state, training: true);
            Assert.Throws<InvalidOperationException>(() => plain.StoreExperienceWithPrivilegedCritic(
                state, action, 0.0, state, true, new Vector<double>([1.0]), new Vector<double>([1.0])));
        }

        using var privileged = Agent(Options(1), 3);
        var a = privileged.SelectAction(state, training: true);
        Assert.Throws<InvalidOperationException>(() => privileged.StoreExperience(state, a, 0.0, state, true));
        Assert.Throws<ArgumentException>(() => privileged.StoreExperienceWithPrivilegedCritic(
            state, a, 0.0, state, true, new Vector<double>([1.0, 2.0]), new Vector<double>([1.0])));
        privileged.StoreExperienceWithPrivilegedCritic(state, a, 0.0, state, true, new Vector<double>([1.0]), new Vector<double>([1.0]));
        Assert.Equal(1, privileged.RolloutCount);

        // The critic architecture must take the observation plus the privileged inputs.
        Assert.Throws<ArgumentException>(() => Agent(Options(1), 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => Agent(Options(-1), 2));
    }

    [Fact]
    [Trait("category", "unit")]
    public void A_critic_that_sees_the_hidden_reward_fits_returns_the_plain_critic_cannot()
    {
        // One-step episodes with a constant observation: the reward z ~ N(0, 1) is invisible to the actor and to a plain
        // critic (which can only learn E[z] = 0), but is the privileged critic's extra input. The loss with no entropy
        // term and advantages standardized to mean zero is dominated by the value loss, so the privileged agent's loss
        // must fall far below the plain one's (~Var z = 1).
        double LateLoss(int privileged)
        {
            using var agent = Agent(Options(privileged), 2 + privileged);
            var rng = RandomHelper.CreateSeededRandom(11);
            var state = new Vector<double>([1.0, 0.0]);
            var losses = new System.Collections.Generic.List<double>();
            for (int step = 0; step < 64 * 60; step++)
            {
                double z = Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
                var action = agent.SelectAction(state, training: true);
                if (privileged > 0)
                    agent.StoreExperienceWithPrivilegedCritic(state, action, z, state, true, new Vector<double>([z]), new Vector<double>([0.0]));
                else
                    agent.StoreExperience(state, action, z, state, true);
                if (agent.RolloutCount >= 64) losses.Add(agent.Train());
            }
            return losses.Skip(losses.Count - 10).Average();
        }

        var plain = LateLoss(0);
        var privileged = LateLoss(1);
        Assert.True(plain > 0.6, $"plain critic loss {plain:F3} (expected about Var z = 1)");
        Assert.True(privileged < 0.25 * plain, $"privileged {privileged:F3} vs plain {plain:F3}");
    }
}
