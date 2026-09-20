using System;
using System.Reflection;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Evidence that the remaining trading-agent options are read by something.
/// </summary>
/// <remarks>
/// <c>RewardScale</c>, <c>SACAlpha</c>, <c>AutoTuneAlpha</c> and <c>Tau</c> were all settable and all
/// documented, and none of them was referenced by any agent. An option that is silently ignored is worse
/// than a missing one: it reads as a knob that has been tried.
/// </remarks>
public sealed class TradingAgentOptionWiringTests
{
    private const int StateSize = 4;

    private static bool SameParameters(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }

        return true;
    }

    private static double MaxAbsoluteDifference(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length) return double.PositiveInfinity;
        double worst = 0.0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    private static INeuralNetwork<double> Network(object agent, Type type, string fieldName)
    {
        var field = type.GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
        if (field is null)
        {
            throw new InvalidOperationException($"{type.Name} has no field '{fieldName}'.");
        }

        if (field.GetValue(agent) is not INeuralNetwork<double> network)
        {
            throw new InvalidOperationException($"Field '{fieldName}' does not hold an INeuralNetwork<double>.");
        }

        return network;
    }

    private static Vector<double> TrainDqnWithRewardScale(double rewardScale, int seed)
    {
        var options = (FinancialDQNAgentOptions<double>)Options(Dqn, StateSize, 3, seed);
        options.WarmupSteps = 0;
        options.BatchSize = 4;
        options.LearningRate = 0.01;
        options.EpsilonStart = 0.0;
        options.EpsilonEnd = 0.0;
        options.RewardScale = rewardScale;

        using var agent = (FinancialDQNAgent<double>)Create(Dqn, options);
        for (int i = 0; i < 16; i++)
        {
            agent.StoreExperience(
                State(StateSize, salt: i), OneHot(3, i % 3), 0.01, State(StateSize, salt: i + 1), done: true);
        }

        for (int i = 0; i < 20; i++)
        {
            agent.Train();
        }

        return agent.GetParameters().Clone();
    }

    [Fact]
    [Trait("category", "unit")]
    public void RewardScale_changes_what_the_agent_learns()
    {
        var unscaled = TrainDqnWithRewardScale(1.0, seed: 71);
        var scaled = TrainDqnWithRewardScale(50.0, seed: 71);

        Assert.False(SameParameters(unscaled, scaled),
            "RewardScale changed nothing: the reward reaches the update unscaled.");
    }

    private static FinancialSACAgentOptions<double> SacOptions(int seed)
    {
        var options = (FinancialSACAgentOptions<double>)Options(Sac, StateSize, 1, seed);
        options.WarmupSteps = 0;
        options.BatchSize = 8;
        options.LearningRate = 0.01;
        options.AutoTuneAlpha = false;
        return options;
    }

    private static void StoreBandit(FinancialSACAgent<double> agent, Vector<double> state, int pairs)
    {
        var good = new Vector<double>(1);
        good[0] = 1.0;
        var bad = new Vector<double>(1);
        bad[0] = -1.0;
        for (int i = 0; i < pairs; i++)
        {
            agent.StoreExperience(state, good, 1.0, state, done: false);
            agent.StoreExperience(state, bad, -1.0, state, done: false);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Tau_controls_how_fast_the_target_critic_tracks()
    {
        var state = State(StateSize, salt: 2);

        double Drift(double tau)
        {
            var options = SacOptions(seed: 72);
            options.Tau = tau;
            using var agent = (FinancialSACAgent<double>)Create(Sac, options);
            var target = Network(agent, typeof(FinancialSACAgent<double>), "_targetCritic1");
            var before = target.GetParameters().Clone();

            StoreBandit(agent, state, pairs: 16);
            for (int i = 0; i < 20; i++)
            {
                agent.Train();
            }

            return MaxAbsoluteDifference(before, target.GetParameters());
        }

        double slow = Drift(0.005);
        double fast = Drift(0.5);

        Assert.True(slow > 0.0, "the target critic never moved at all: the soft update is a no-op.");
        Assert.True(fast > slow,
            $"a larger Tau did not track faster (tau=0.005 moved {slow:E3}, tau=0.5 moved {fast:E3}).");
    }

    [Fact]
    [Trait("category", "unit")]
    public void SACAlpha_changes_the_soft_value_target()
    {
        var state = State(StateSize, salt: 3);

        Vector<double> TrainWithAlpha(double alpha)
        {
            var options = SacOptions(seed: 73);
            options.SACAlpha = alpha;
            using var agent = (FinancialSACAgent<double>)Create(Sac, options);
            StoreBandit(agent, state, pairs: 16);
            for (int i = 0; i < 20; i++)
            {
                agent.Train();
            }

            var critic = Network(agent, typeof(FinancialSACAgent<double>), "_critic1");
            return critic.GetParameters().Clone();
        }

        var cold = TrainWithAlpha(0.0);
        var hot = TrainWithAlpha(1.0);

        Assert.False(SameParameters(cold, hot),
            "SACAlpha changed nothing: the entropy term is missing from the critic target.");
    }

    [Fact]
    [Trait("category", "unit")]
    public void AutoTuneAlpha_moves_the_temperature()
    {
        var options = SacOptions(seed: 74);
        options.AutoTuneAlpha = true;
        using var agent = (FinancialSACAgent<double>)Create(Sac, options);

        double before = agent.CurrentAlpha;
        StoreBandit(agent, State(StateSize, salt: 4), pairs: 16);
        for (int i = 0; i < 50; i++)
        {
            agent.Train();
        }

        Assert.NotEqual(before, agent.CurrentAlpha, 12);
    }

    [Theory]
    [Trait("category", "unit")]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    public void Validate_rejects_a_non_positive_RewardScale(double rewardScale)
    {
        var options = new TradingAgentOptions<double> { RewardScale = rewardScale };
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Theory]
    [Trait("category", "unit")]
    [InlineData(0.0)]
    [InlineData(1.5)]
    public void Validate_rejects_a_Tau_outside_the_unit_interval(double tau)
    {
        var options = new TradingAgentOptions<double> { Tau = tau };
        Assert.Throws<ArgumentException>(() => options.Validate());
    }

    [Fact]
    [Trait("category", "unit")]
    public void Validate_rejects_a_non_positive_rollout_configuration()
    {
        Assert.Throws<ArgumentException>(() => new FinancialA2CAgentOptions<double> { NSteps = 0 }.Validate());
        Assert.Throws<ArgumentException>(() => new FinancialA2CAgentOptions<double> { NumEnvironments = 0 }.Validate());
    }
}
