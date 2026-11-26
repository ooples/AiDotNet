using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.DQN;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class RLBestRewardTests
{
    /// <summary>One-step episodes whose reward is -(episode number), so every episode reward is negative.</summary>
    private sealed class NegativeRewardEnvironment : IEnvironment<double>
    {
        private int _episodes;

        public int ObservationSpaceDimension => 1;
        public int ActionSpaceSize => 2;
        public bool IsContinuousActionSpace => false;

        public Vector<double> Reset()
        {
            _episodes++;
            return new Vector<double>(1);
        }

        public (Vector<double> NextState, double Reward, bool Done, Dictionary<string, object> Info) Step(Vector<double> action)
        {
            return (new Vector<double>(1), -_episodes, true, new Dictionary<string, object>());
        }

        public void Seed(int seed)
        {
        }

        public void Close()
        {
        }
    }

    [Fact(Timeout = 120000)]
    public async Task BuildAsync_AllNegativeRewards_ReportsTheHighestEpisodeReward()
    {
        // Episode rewards are -1, -2, -3: the best is -1. Seeding the running max with 0 reported 0.
        var environment = new NegativeRewardEnvironment();
        var agent = new DQNAgent<double>(new DQNOptions<double>
        {
            StateSize = 1,
            ActionSize = 2,
            BatchSize = 1,
            ReplayBufferSize = 8,
            TargetUpdateFrequency = 1,
            WarmupSteps = 0,
            EpsilonStart = 0.0,
            EpsilonEnd = 0.0,
            EpsilonDecay = 1.0,
            HiddenLayers = new List<int> { 4 },
            Seed = 123
        });

        RLTrainingSummary<double>? summary = null;
        var rlOptions = new RLTrainingOptions<double>
        {
            Environment = environment,
            Episodes = 3,
            MaxStepsPerEpisode = 1,
            LogFrequency = 0,
            OnTrainingComplete = metrics => summary = metrics
        };

        await new AiModelBuilder<double, Vector<double>, Vector<double>>()
            .ConfigureReinforcementLearning(rlOptions)
            .ConfigureModel(agent)
            .BuildAsync();

        Assert.NotNull(summary);
        Assert.Equal(-1.0, summary?.BestReward ?? double.NaN, 12);
        Assert.Equal(-2.0, summary?.AverageReward ?? double.NaN, 12);
    }
}
