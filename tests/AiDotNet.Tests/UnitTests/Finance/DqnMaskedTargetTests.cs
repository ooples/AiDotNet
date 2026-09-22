using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

public sealed class DqnMaskedTargetTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Mask_changes_actual_training_target_through_shared_storage_boundary(bool doubleDqn, bool tradingBoundary)
    {
        var unrestricted = Train(doubleDqn, tradingBoundary, masked: false);
        var restricted = Train(doubleDqn, tradingBoundary, masked: true);
        Assert.False(unrestricted.SequenceEqual(restricted), "The forbidden greedy next action still controls the TD target.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Replay_snapshots_mask_before_caller_reuses_array(bool doubleDqn)
    {
        Assert.Equal(Train(doubleDqn, false, masked: true),
            Train(doubleDqn, false, masked: true, mutateMask: true));
    }

    [Fact]
    public void Terminal_transition_ignores_empty_next_legal_set()
    {
        Assert.Equal(Train(false, false, masked: false, terminal: true),
            Train(false, false, masked: true, terminal: true));
    }

    [Fact]
    public void Nonterminal_transition_rejects_empty_or_wrong_size_mask()
    {
        using var agent = Create(false);
        AiDotNet.ReinforcementLearning.Agents.ReinforcementLearningAgentBase<double> shared = agent;
        var state = new Vector<double>(new[] { 0.1, 0.2, -0.3, 0.4 });
        var action = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
        Assert.Throws<InvalidOperationException>(() => shared.StoreExperience(state, action, 0.5, state, false, new bool[3]));
        Assert.Throws<ArgumentException>(() => shared.StoreExperience(state, action, 0.5, state, false, new[] { true }));
    }

    private static double[] Train(bool doubleDqn, bool tradingBoundary, bool masked,
        bool mutateMask = false, bool terminal = false)
    {
        using var agent = Create(doubleDqn);
        var state = new Vector<double>(new[] { 0.1, 0.2, -0.3, 0.4 });
        var next = new Vector<double>(new[] { -0.2, 0.4, 0.3, 0.7 });
        var greedy = agent.SelectAction(next, training: false);
        var greedyIndex = Enumerable.Range(0, 3).Single(i => greedy[i] == 1.0);
        bool[]? mask = null;
        if (masked)
        {
            mask = new bool[3];
            if (!terminal) mask[(greedyIndex + 1) % 3] = true;
        }
        var action = new Vector<double>(new[] { 1.0, 0.0, 0.0 });
        if (tradingBoundary)
        {
            TradingAgentBase<double> shared = agent;
            shared.StoreTradingExperience(state, action, 0.5, next, terminal, 0.0, mask);
        }
        else
        {
            agent.StoreExperience(state, action, 0.5, next, terminal, mask);
        }
        if (mutateMask && mask is not null)
            for (var i = 0; i < mask.Length; i++) mask[i] = true;
        agent.Train();
        return agent.GetParameters().ToArray();
    }

    private static FinancialDQNAgent<double> Create(bool doubleDqn) => new(
        new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 3, complexity: NetworkComplexity.Simple) { RandomSeed = 61 },
        new FinancialDQNAgentOptions<double>
        {
            StateSize = 4, ActionSize = 3, Seed = 61, BatchSize = 1, WarmupSteps = 0,
            LearningRate = 0.01, UseDoubleDQN = doubleDqn, UseDuelingNetwork = false,
            EpsilonStart = 0, EpsilonEnd = 0, TargetUpdateFrequency = 1000,
        });
}
