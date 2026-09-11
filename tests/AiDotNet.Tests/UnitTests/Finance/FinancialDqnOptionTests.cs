using System.Linq;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// Evidence that <see cref="FinancialDQNAgentOptions{T}.UseDoubleDQN"/> and
/// <see cref="FinancialDQNAgentOptions{T}.UseDuelingNetwork"/> change what the agent does.
/// </summary>
/// <remarks>
/// Both options were settable — and <c>UseDoubleDQN</c> even defaulted to <c>true</c> — but neither was read
/// anywhere in the agent. Every DQN built the same plain Q-network and every TD target took the plain
/// <c>max_a' Q'(s', a')</c>, so switching either option changed nothing at all.
/// </remarks>
public sealed class FinancialDqnOptionTests
{
    private const int StateSize = 4;
    private const int ActionSize = 3;

    private static FinancialDQNAgentOptions<double> DqnOptions(int seed)
    {
        var options = (FinancialDQNAgentOptions<double>)Options(Dqn, StateSize, ActionSize, seed);
        options.WarmupSteps = 0;
        options.BatchSize = 4;
        options.LearningRate = 0.01;
        options.EpsilonStart = 0.0;
        options.EpsilonEnd = 0.0;
        options.TargetUpdateFrequency = 1000; // keep the target fixed so only the target RULE differs
        return options;
    }

    private static void StoreTransitions(FinancialDQNAgent<double> agent, int count)
    {
        for (int i = 0; i < count; i++)
        {
            agent.StoreExperience(
                State(StateSize, salt: i),
                OneHot(ActionSize, i % ActionSize),
                (i % 2 == 0) ? 1.0 : -1.0,
                State(StateSize, salt: i + 1),
                done: false);
        }
    }

    private static Vector<double> TrainAndGetParameters(bool useDoubleDqn, int seed)
    {
        var options = DqnOptions(seed);
        options.UseDoubleDQN = useDoubleDqn;
        using var agent = (FinancialDQNAgent<double>)Create(Dqn, options);
        StoreTransitions(agent, 16);
        for (int i = 0; i < 20; i++)
        {
            agent.Train();
        }

        return agent.GetParameters().Clone();
    }

    private static bool SameParameters(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i]) return false;
        }

        return true;
    }

    [Fact]
    [Trait("category", "unit")]
    public void UseDoubleDQN_changes_the_temporal_difference_target()
    {
        var doubled = TrainAndGetParameters(useDoubleDqn: true, seed: 61);
        var plain = TrainAndGetParameters(useDoubleDqn: false, seed: 61);

        Assert.False(SameParameters(doubled, plain),
            "UseDoubleDQN changed nothing: selection and evaluation are still done by the same network.");
    }

    [Fact]
    [Trait("category", "unit")]
    public void UseDuelingNetwork_builds_a_dueling_head()
    {
        var options = DqnOptions(seed: 62);
        options.UseDuelingNetwork = true;
        options.HiddenLayers = new[] { 8, 6 };
        var architecture = Arch(StateSize, ActionSize);
        using var agent = (FinancialDQNAgent<double>)Create(Dqn, options, architecture);

        Assert.True(agent.UsesDuelingNetwork, "the agent did not report a dueling network.");
        Assert.Contains(architecture.Layers, layer => layer is DuelingCombinationLayer<double>);

        // The dueling head still emits one Q-value per action, so the agent behaves like any other DQN.
        var action = agent.SelectAction(State(StateSize, salt: 1), training: false);
        Assert.Equal(ActionSize, action.Length);
    }

    [Fact]
    [Trait("category", "unit")]
    public void A_plain_DQN_has_no_dueling_head()
    {
        var options = DqnOptions(seed: 63);
        options.UseDuelingNetwork = false;
        options.HiddenLayers = new[] { 8, 6 };
        var architecture = Arch(StateSize, ActionSize);
        using var agent = (FinancialDQNAgent<double>)Create(Dqn, options, architecture);

        Assert.False(agent.UsesDuelingNetwork);
        Assert.DoesNotContain(architecture.Layers, layer => layer is DuelingCombinationLayer<double>);
    }

    [Fact]
    [Trait("category", "unit")]
    public void A_dueling_DQN_still_learns_from_reward()
    {
        var options = DqnOptions(seed: 64);
        options.UseDuelingNetwork = true;
        using var agent = (FinancialDQNAgent<double>)Create(Dqn, options);

        var initial = agent.GetParameters().Clone();
        StoreTransitions(agent, 16);
        for (int i = 0; i < 20; i++)
        {
            agent.Train();
        }

        Assert.False(SameParameters(initial, agent.GetParameters()),
            "the dueling Q-network never trained.");
    }
}
