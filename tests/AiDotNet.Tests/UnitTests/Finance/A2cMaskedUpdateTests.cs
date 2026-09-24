using AiDotNet.Enums;
using AiDotNet.Finance.Trading.Agents;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

public sealed class A2cMaskedUpdateTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Singleton_legal_policy_has_zero_policy_loss_even_if_caller_mutates_mask(bool mutateMask)
    {
        using var agent = CreateAgent();
        var loss = CollectAndTrain(agent, new[] { false, false, true, false }, mutateMask);
        Assert.Equal(0.0, loss, tolerance: 1e-6);
    }

    [Fact]
    public void All_legal_mask_matches_unmasked_training()
    {
        using var masked = CreateAgent();
        using var unmasked = CreateAgent();
        Assert.Equal(CollectAndTrain(unmasked, null, false),
            CollectAndTrain(masked, new[] { true, true, true, true }, false), tolerance: 1e-8);
    }

    private static double CollectAndTrain(FinancialA2CAgent<double> agent, bool[]? template, bool mutateMask)
    {
        for (var step = 0; step < 8; step++)
        {
            var state = State(step);
            var mask = template is null ? null : (bool[])template.Clone();
            var action = agent.SelectAction(state, training: true, mask);
            if (mutateMask && mask is not null)
                for (var i = 0; i < mask.Length; i++) mask[i] = true;
            agent.StoreExperience(state, action, 0.2 + step * 0.1, State(step + 1), done: step == 7);
        }
        return agent.Train();
    }

    private static Vector<double> State(int step) => new(new[] { 0.1 + step * 0.01, -0.2, 0.3, 0.4 });

    private static FinancialA2CAgent<double> CreateAgent()
    {
        var options = new FinancialA2CAgentOptions<double>
        {
            StateSize = 4, ActionSize = 4, Seed = 17, BatchSize = 8, NSteps = 8,
            WarmupSteps = 0, ValueCoefficient = 0, EntropyCoefficient = 0.1,
        };
        return new FinancialA2CAgent<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 4, complexity: NetworkComplexity.Simple) { RandomSeed = 17 },
            new NeuralNetworkArchitecture<double>(inputFeatures: 4, outputSize: 1, complexity: NetworkComplexity.Simple) { RandomSeed = 17 }, options);
    }
}
