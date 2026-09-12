using System.Linq;
using AiDotNet.Finance.Trading.Agents;
using Xunit;
using static AiDotNet.Tests.UnitTests.Finance.FinancialAgentTestKit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// <c>TradingAgentOptions.WarmupSteps</c> ("number of steps before training begins") and
/// <c>TradingAgentOptions.HiddenLayers</c> ("hidden layer sizes for the neural network") were settable but
/// read by nothing: replay agents started updating as soon as one minibatch was stored, and every
/// agent-built network was a hard-coded 64x64 MLP whatever the option said.
/// </summary>
public sealed class TradingAgentWarmupAndHiddenLayersTests
{
    private const int StateSize = 4;
    private const int ActionSize = 3;

    [Theory]
    [InlineData(Dqn)]
    [InlineData(A2C)]
    [InlineData(Sac)]
    [InlineData(MarketMaking)]
    [Trait("category", "unit")]
    public void Replay_agents_do_not_update_until_WarmupSteps_transitions_are_stored(FinancialAgentKind kind)
    {
        var options = Options(kind, StateSize, ActionSize, seed: 8);
        options.BatchSize = 2;
        options.WarmupSteps = 10;
        using var agent = Create(kind, options);

        var initial = agent.GetParameters().Clone();
        StoreTransitions(agent, kind, count: 5);
        agent.Train();
        Assert.True(SameParameters(initial, agent.GetParameters()),
            $"{kind} updated its networks after 5 of {options.WarmupSteps} warmup transitions.");

        StoreTransitions(agent, kind, count: 5);
        agent.Train();
        Assert.False(SameParameters(initial, agent.GetParameters()),
            $"{kind} did not start training once {options.WarmupSteps} warmup transitions were stored.");
    }

    [Theory]
    [InlineData(Dqn)]
    [InlineData(A2C)]
    [InlineData(Ppo)]
    [InlineData(Sac)]
    [InlineData(MarketMaking)]
    [Trait("category", "unit")]
    public void Agent_built_networks_use_the_HiddenLayers_option(FinancialAgentKind kind)
    {
        var options = Options(kind, StateSize, ActionSize, seed: 9);
        options.HiddenLayers = new[] { 7, 5 };
        var architecture = Arch(StateSize, ActionSize);
        using var agent = Create(kind, options, architecture);

        var widths = architecture.Layers.Select(layer => layer.GetOutputShape()[0]).ToArray();
        Assert.Equal(new[] { 7, 5, ActionSize }, widths);
        var networks = Networks(agent, kind, StateSize, ActionSize);
        Assert.Equal(kind switch { Sac => 5, MarketMaking => 1, _ => 2 }, networks.Count);
        foreach (var entry in networks)
        {
            Assert.Equal(entry.Inputs, entry.Network.Architecture.CalculatedInputSize);
            Assert.Equal(entry.Outputs, entry.Network.Architecture.OutputSize);
            Assert.Equal(new[] { 7, 5, entry.Outputs },
                entry.Network.Layers.Select(layer => layer.GetOutputShape()[0]).ToArray());
            Assert.Equal(new[] { 7, 5, entry.Outputs },
                entry.Network.Architecture.Layers.Select(layer => layer.GetOutputShape()[0]).ToArray());
        }

        // Equal widths must not conceal shared mutable layers between online/target/twin networks.
        for (int i = 0; i < networks.Count; i++)
        for (int j = i + 1; j < networks.Count; j++)
        {
            Assert.NotSame(networks[i].Network.Architecture, networks[j].Network.Architecture);
            for (int layer = 0; layer < 3; layer++)
                Assert.NotSame(networks[i].Network.Layers[layer], networks[j].Network.Layers[layer]);
        }
    }

    [Fact]
    [Trait("category", "unit")]
    public void Default_HiddenLayers_preserve_the_historical_64x64_default_network()
    {
        var options = Options(Dqn, StateSize, ActionSize, seed: 10);
        var architecture = Arch(StateSize, ActionSize);
        using var agent = Create(Dqn, options, architecture);

        var widths = architecture.Layers.Select(layer => layer.GetOutputShape()[0]).ToArray();
        Assert.Equal(new[] { 64, 64, ActionSize }, widths);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Caller_supplied_layers_are_not_replaced_by_HiddenLayers()
    {
        var options = Options(Dqn, StateSize, ActionSize, seed: 11);
        options.HiddenLayers = new[] { 7, 5 };
        var architecture = Arch(StateSize, ActionSize);
        var hidden = new AiDotNet.NeuralNetworks.Layers.DenseLayer<double>(
            12, (AiDotNet.Interfaces.IActivationFunction<double>)new AiDotNet.ActivationFunctions.ReLUActivation<double>());
        var output = new AiDotNet.NeuralNetworks.Layers.DenseLayer<double>(
            ActionSize, (AiDotNet.Interfaces.IActivationFunction<double>)new AiDotNet.ActivationFunctions.IdentityActivation<double>());
        architecture.Layers.Add(hidden);
        architecture.Layers.Add(output);
        using var agent = Create(Dqn, options, architecture);

        Assert.Equal(2, architecture.Layers.Count);
        Assert.Same(hidden, architecture.Layers[0]);
        Assert.Same(output, architecture.Layers[1]);
    }

    private static void StoreTransitions(TradingAgentBase<double> agent, FinancialAgentKind kind, int count)
    {
        for (int i = 0; i < count; i++)
        {
            var state = State(StateSize, salt: i);
            var action = kind is Sac or MarketMaking
                ? State(ActionSize, salt: 100 + i)
                : OneHot(ActionSize, i % ActionSize);
            agent.StoreExperience(state, action, 0.1 * (i + 1), State(StateSize, salt: i + 1), done: false);
        }
    }

    private static bool SameParameters(Vector<double> a, Vector<double> b)
    {
        if (a.Length != b.Length)
        {
            return false;
        }

        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] != b[i])
            {
                return false;
            }
        }

        return true;
    }
}
