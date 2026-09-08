using System;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// FinancialDQNAgent had an exploration schedule that never ran and a target sync that was a coin flip.
///
/// <para><c>SelectAction</c> compared against <c>TradingOptions.EpsilonStart</c> — the INITIAL value — on every
/// call, so the behaviour policy never annealed. With the default 1.0 the agent acts uniformly at random for an
/// entire run, which makes its learning curve noise. Separately, the target network synced on
/// <c>Next(TargetUpdateFrequency) == 0</c>; at the default 1000 over a few hundred updates that is roughly
/// 0.6 syncs per run, at random moments.</para>
///
/// <para>Measured downstream: across 235 production bake-off runs this agent completed a maximum of ONE round
/// trip, ever. <c>EpsilonEnd</c> and <c>EpsilonDecay</c> existed on the options the whole time and nothing
/// read them.</para>
/// </summary>
public class FinancialDqnExplorationScheduleTests
{
    [Fact]
    public void Epsilon_decays_toward_the_floor_instead_of_staying_at_its_initial_value()
    {
        var agent = Agent(epsilonStart: 1.0, epsilonEnd: 0.05, epsilonDecay: 0.5);
        var before = Epsilon(agent);

        // Each update applies one decay step: 1.0 -> 0.5 -> 0.25 -> ... floored at EpsilonEnd.
        for (var i = 0; i < 6; i++)
        {
            Train(agent);
        }

        var after = Epsilon(agent);
        Assert.Equal(1.0, before, precision: 6);
        Assert.True(after < before, $"epsilon did not decay: {before} -> {after}");
        Assert.Equal(0.05, after, precision: 6);
    }

    [Fact]
    public void Epsilon_never_falls_below_the_configured_floor()
    {
        // The floor is what keeps a converged policy still exploring a little. Decaying past it would make the
        // agent greedy forever, which is the opposite failure to the one being fixed.
        var agent = Agent(epsilonStart: 0.5, epsilonEnd: 0.1, epsilonDecay: 0.1);
        for (var i = 0; i < 20; i++)
        {
            Train(agent);
        }

        Assert.Equal(0.1, Epsilon(agent), precision: 6);
    }

    [Fact]
    public void A_schedule_that_asks_for_no_decay_is_honoured()
    {
        // EpsilonDecay = 1.0 means "hold the rate". It must not be mistaken for "no schedule configured" and
        // silently annealed anyway.
        var agent = Agent(epsilonStart: 0.3, epsilonEnd: 0.01, epsilonDecay: 1.0);
        for (var i = 0; i < 5; i++)
        {
            Train(agent);
        }

        Assert.Equal(0.3, Epsilon(agent), precision: 6);
    }

    [Fact]
    public void The_exploration_rate_is_reported_so_the_curve_can_be_seen()
    {
        // Without a published metric an agent that annealed and one that never explored report identical
        // rewards, and the difference lives only in a private field.
        var agent = Agent(epsilonStart: 0.8, epsilonEnd: 0.02, epsilonDecay: 0.5);
        Train(agent);

        var metrics = agent.GetTradingMetrics();
        Assert.True(metrics.ContainsKey("Epsilon"), "the agent does not report its exploration rate");
        Assert.Equal(0.4, Convert.ToDouble(metrics["Epsilon"]), precision: 6);
    }

    private static FinancialDQNAgent<double> Agent(double epsilonStart, double epsilonEnd, double epsilonDecay)
    {
        var options = new FinancialDQNAgentOptions<double>
        {
            StateSize = 4,
            ActionSize = 3,
            EpsilonStart = epsilonStart,
            EpsilonEnd = epsilonEnd,
            EpsilonDecay = epsilonDecay,
            BatchSize = 2,
            ReplayBufferSize = 32,
        };

        // Built through the same helper the existing trading-agent tests use, so this exercises the shape the
        // suite already trusts rather than a second, divergent construction path.
        return new FinancialDQNAgent<double>(FinanceTestHelpers.CreateArchitecture<double>(4, 3), options);
    }

    /// <summary>Drives one training update through the public surface, which is what advances the schedule.</summary>
    private static void Train(FinancialDQNAgent<double> agent)
    {
        var numOps = MathHelper.GetNumericOperations<double>();
        var state = FinanceTestHelpers.CreateRandomVector<double>(4, seed: 123);
        var next = FinanceTestHelpers.CreateRandomVector<double>(4, seed: 321);

        for (var i = 0; i < 4; i++)
        {
            agent.StoreTradingExperience(
                state, agent.SelectTradingAction(state, training: true),
                numOps.FromDouble(0.01), next, done: false, pnl: numOps.FromDouble(0.02));
        }

        _ = agent.TrainOnExperiences();
    }

    private static double Epsilon(FinancialDQNAgent<double> agent) =>
        Convert.ToDouble(agent.GetTradingMetrics()["Epsilon"]);
}
