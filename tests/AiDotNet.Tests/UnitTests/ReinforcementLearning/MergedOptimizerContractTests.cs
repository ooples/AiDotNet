using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.ReinforcementLearning.Agents;
using AiDotNet.ReinforcementLearning.Agents.DecisionTransformer;
using AiDotNet.ReinforcementLearning.Agents.QMIX;
using AiDotNet.ReinforcementLearning.Agents.TRPO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ReinforcementLearning;

[Collection("FusedOptimizerGlobalState")]
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
                    RnnHiddenDim = 4, MixingEmbedDim = 4, BatchSize = 2, ReplayBufferSize = 2, OptimizerFactory = Factory,
                });
                Assert.Equal(i, created.Count);
                var observations = new List<Vector<double>> { new(3), new(3) };
                var actions = new List<Vector<double>> { new(new[] { 1.0, 0.0 }), new(new[] { 1.0, 0.0 }) };
                var state = new Vector<double>(6);
                for (int episode = 0; episode < 2; episode++)
                    agent.StoreMultiAgentExperience(observations, actions, 1.0, observations, state, state, done: true);
                agent.Train();
            }
            else
            {
                using var agent = new DecisionTransformerAgent<double>(new DecisionTransformerOptions<double>
                {
                    StateSize = 2, ActionSize = 2, EmbeddingDim = 8, NumHeads = 1,
                    NumLayers = 1, ContextLength = 2, MaxEpisodeLength = 8, OptimizerFactory = Factory,
                });
                Assert.Same(created[created.Count - 1], ((NeuralNetworkBase<double>)agent.TransformerNetwork).TrainingOptimizer);
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
