using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;
using AiDotNet.ReinforcementLearning.Agents.QMIX;
using AiDotNet.ReinforcementLearning.Agents.TRPO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

public sealed class MergedOptimizerContractTests
{
    [Fact]
    public void DecisionTransformerWarmupAliasUsesTheInheritedSlot()
    {
        var options = new DecisionTransformerOptions<double> { LearningRateWarmupSteps = 9 };
        Assert.Equal(9, options.WarmupSteps);
        ReinforcementLearningOptions<double> inherited = new DecisionTransformerOptions<double> { WarmupSteps = 5 };
        Assert.Equal(5, Assert.IsType<DecisionTransformerOptions<double>>(inherited).LearningRateWarmupSteps);
        Assert.Equal(1e-4, options.LearningRate);
        Assert.Equal(64, options.BatchSize);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EachAgentCreatesItsOwnFactoryOptimizer(bool qmix)
    {
        var created = new List<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>();
        IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>> Factory()
        {
            var optimizer = new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>> { InitialLearningRate = 0.01 });
            created.Add(optimizer);
            return optimizer;
        }
        for (int i = 0; i < 2; i++)
        {
            if (qmix)
            {
                using var agent = new QMIXAgent<double>(new QMIXOptions<double>
                {
                    NumAgents = 2, StateSize = 3, ActionSize = 2, GlobalStateSize = 6,
                    RnnHiddenDim = 4, MixingEmbedDim = 4, OptimizerFactory = Factory,
                });
            }
            else
            {
                using var agent = new DecisionTransformerAgent<double>(new DecisionTransformerOptions<double>
                {
                    StateSize = 2, ActionSize = 2, EmbeddingDim = 8, NumHeads = 1,
                    NumLayers = 1, ContextLength = 2, MaxEpisodeLength = 8, OptimizerFactory = Factory,
                });
            }
        }
        Assert.Equal(2, created.Count);
        Assert.NotSame(created[0], created[1]);
    }

    [Fact]
    public void ContinuousTrpoSeedsBothNetworkWeightsAndExploration()
    {
        TRPOAgent<double> Create() => new(new TRPOOptions<double>
        {
            StateSize = 2, ActionSize = 1, IsContinuous = true, Seed = 23,
            PolicyHiddenLayers = new List<int> { 3 }, ValueHiddenLayers = new List<int> { 3 },
        });
        using var first = Create();
        using var second = Create();
        var state = new Vector<double>(new[] { 0.2, 0.4 });
        for (int i = 0; i < 4; i++)
            Assert.Equal(first.SelectAction(state).ToArray(), second.SelectAction(state).ToArray());
    }
}
