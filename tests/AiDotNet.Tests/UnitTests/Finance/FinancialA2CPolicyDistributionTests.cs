using AiDotNet.Finance.Trading.Agents;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Regression tests for <see cref="FinancialA2CAgent{T}"/>'s policy distribution. The actor is a regression
/// network with a linear (identity) output, so its outputs are LOGITS — they can be negative and do not sum to
/// one. Sampling from them as if they were probabilities collapsed exploration onto an edge tier, and training
/// them by MSE towards the sampled one-hot action ignored the advantage entirely (a punished action was
/// reinforced exactly like a rewarded one).
/// </summary>
public sealed class FinancialA2CPolicyDistributionTests
{
    private const int StateSize = 4;
    private const int ActionSize = 3;

    private static FinancialA2CAgent<double> CreateZeroedAgent(int seed, out Vector<double> state)
    {
        var options = Options(A2C, StateSize, ActionSize, seed);
        options.BatchSize = 1;
        options.WarmupSteps = 0;
        options.EntropyCoefficient = 0.0;
        var agent = (FinancialA2CAgent<double>)Create(A2C, options);

        state = State(StateSize, salt: 1);
        agent.SelectAction(state, training: false); // materialize lazily-shaped layers
        // All-zero weights make every actor output (logit) exactly 0 for every state, so the policy
        // must be the uniform distribution and the critic must predict V(s) = 0.
        agent.SetParameters(new Vector<double>((int)agent.ParameterCount));
        return agent;
    }

    [Fact]
    [Trait("category", "unit")]
    public void Training_actions_are_sampled_from_the_softmax_of_the_actor_logits()
    {
        using var agent = CreateZeroedAgent(seed: 11, out var state);

        const int draws = 3000;
        var counts = new int[ActionSize];
        for (int i = 0; i < draws; i++)
        {
            var action = agent.SelectAction(state, training: true);
            Assert.Equal(ActionSize, action.Length);
            counts[ArgMax(action)]++;
        }

        // softmax([0, 0, 0]) = [1/3, 1/3, 1/3]. Treating the raw zeros as probabilities never sampled the
        // first or the interior tier at all (the cumulative sum never exceeds the uniform draw).
        for (int a = 0; a < ActionSize; a++)
        {
            double frequency = counts[a] / (double)draws;
            Assert.True(frequency > 0.28 && frequency < 0.39,
                $"action {a} sampled with frequency {frequency:F3}; counts = [{string.Join(", ", counts)}]");
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Actor_update_moves_probability_away_from_an_action_with_negative_advantage()
    {
        using var agent = CreateZeroedAgent(seed: 3, out var state);
        const int punished = 1;

        // Terminal transition, reward -1, V(s) = 0  =>  advantage = -1 for the interior action.
        agent.StoreExperience(state, SampleActionForIndex(agent, state, punished), -1.0, state, done: true);
        agent.Train();

        // grad of -A * log softmax(z)[a] with A < 0 lowers z[a] and raises every other logit, so the
        // greedy action must leave the punished tier. MSE-to-one-hot raised z[a] instead.
        int greedy = ArgMax(agent.SelectAction(state, training: false));
        Assert.NotEqual(punished, greedy);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Actor_update_moves_probability_towards_an_action_with_positive_advantage()
    {
        using var agent = CreateZeroedAgent(seed: 4, out var state);
        const int rewarded = 2;

        agent.StoreExperience(state, SampleActionForIndex(agent, state, rewarded), 1.0, state, done: true);
        agent.Train();

        Assert.Equal(rewarded, ArgMax(agent.SelectAction(state, training: false)));
    }
}
