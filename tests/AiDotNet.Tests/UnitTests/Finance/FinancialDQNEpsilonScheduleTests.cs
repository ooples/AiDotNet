using AiDotNet.Finance.Trading.Agents;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Regression tests for <see cref="FinancialDQNAgent{T}"/>'s epsilon-greedy schedule. Exploration used to
/// compare an unseeded draw against <c>EpsilonStart</c> forever: <c>EpsilonEnd</c> and <c>EpsilonDecay</c> were
/// never read, so with the default <c>EpsilonStart = 1.0</c> a trained DQN kept acting uniformly at random for
/// its whole training run.
/// </summary>
public sealed class FinancialDQNEpsilonScheduleTests
{
    private const int StateSize = 4;
    private const int ActionSize = 3;

    private static FinancialDQNAgent<double> TrainWithSchedule(
        double start, double end, double decay, int updates, out Vector<double> state)
    {
        var options = Options(Dqn, StateSize, ActionSize, seed: 21);
        options.BatchSize = 1;
        options.WarmupSteps = 0;
        options.EpsilonStart = start;
        options.EpsilonEnd = end;
        options.EpsilonDecay = decay;
        var agent = (FinancialDQNAgent<double>)Create(Dqn, options);

        state = State(StateSize, salt: 3);
        agent.StoreExperience(state, OneHot(ActionSize, 0), 0.5, State(StateSize, salt: 4), done: false);
        for (int i = 0; i < updates; i++)
        {
            agent.Train();
        }

        return agent;
    }

    private static double NonGreedyFraction(FinancialDQNAgent<double> agent, Vector<double> state, int draws)
    {
        int greedy = ArgMax(agent.SelectAction(state, training: false));
        int nonGreedy = 0;
        for (int i = 0; i < draws; i++)
        {
            if (ArgMax(agent.SelectAction(state, training: true)) != greedy)
            {
                nonGreedy++;
            }
        }

        return nonGreedy / (double)draws;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Epsilon_decays_towards_EpsilonEnd_so_a_trained_agent_acts_greedily()
    {
        // 1.0 * 0.5^40 ~ 1e-12: after 40 updates exploration must be effectively off.
        using var agent = TrainWithSchedule(start: 1.0, end: 0.0, decay: 0.5, updates: 40, out var state);

        Assert.Equal(0.0, NonGreedyFraction(agent, state, draws: 500));
    }

    [Fact]
    [Trait("category", "unit")]
    public void Epsilon_is_floored_at_EpsilonEnd()
    {
        // Decays 1.0 -> 0.2 floor. A random action is non-greedy 2/3 of the time, so the expected
        // non-greedy fraction is 0.2 * 2/3 = 0.133 (it was 1.0 * 2/3 = 0.667 with no decay).
        using var agent = TrainWithSchedule(start: 1.0, end: 0.2, decay: 0.5, updates: 40, out var state);

        double fraction = NonGreedyFraction(agent, state, draws: 3000);
        Assert.True(fraction > 0.10 && fraction < 0.17,
            $"non-greedy fraction {fraction:F3}; expected ~0.133 for epsilon floored at 0.2");
        Assert.Equal(0.2, agent.GetTradingMetrics()["Epsilon"], 12);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Epsilon_does_not_decay_before_any_update()
    {
        using var agent = TrainWithSchedule(start: 1.0, end: 0.0, decay: 0.5, updates: 0, out var state);

        Assert.Equal(1.0, agent.GetTradingMetrics()["Epsilon"], 12);
        double fraction = NonGreedyFraction(agent, state, draws: 3000);
        Assert.True(fraction > 0.6 && fraction < 0.73, $"non-greedy fraction {fraction:F3}; expected ~2/3 at epsilon 1");
    }
}
