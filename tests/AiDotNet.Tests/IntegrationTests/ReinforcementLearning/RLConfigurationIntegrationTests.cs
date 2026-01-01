using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.ReinforcementLearning.Environments;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

public class RLConfigurationIntegrationTests
{
    [Fact]
    public void RLTrainingOptions_Default_AssignsExpectedDefaults()
    {
        var environment = new DeterministicBanditEnvironment<double>(maxSteps: 1);

        var options = RLTrainingOptions<double>.Default(environment);

        Assert.Same(environment, options.Environment);
        Assert.Equal(1000, options.Episodes);
        Assert.Equal(500, options.MaxStepsPerEpisode);
        Assert.Equal(1000, options.WarmupSteps);
        Assert.Equal(1, options.TrainFrequency);
        Assert.Equal(64, options.BatchSize);
    }

    [Fact]
    public void RLStepMetrics_Defaults_AreZero()
    {
        var metrics = new RLStepMetrics<double>();

        Assert.Equal(0.0, metrics.Reward, precision: 10);
    }

    [Fact]
    public void RLEpisodeMetrics_Defaults_AreZero()
    {
        var metrics = new RLEpisodeMetrics<double>();

        Assert.Equal(0.0, metrics.TotalReward, precision: 10);
        Assert.Equal(0.0, metrics.AverageLoss, precision: 10);
        Assert.Equal(0.0, metrics.AverageRewardRecent, precision: 10);
    }

    [Fact]
    public void RLTrainingSummary_Defaults_AreZero()
    {
        var summary = new RLTrainingSummary<double>();

        Assert.Equal(0.0, summary.AverageReward, precision: 10);
        Assert.Equal(0.0, summary.BestReward, precision: 10);
        Assert.Equal(0.0, summary.FinalAverageReward, precision: 10);
        Assert.Equal(0.0, summary.AverageLoss, precision: 10);
    }

    [Fact]
    public void RLEvaluationConfig_Defaults_AreExpected()
    {
        var config = new RLEvaluationConfig();

        Assert.Equal(100, config.EvaluateEveryEpisodes);
        Assert.Equal(10, config.EvaluationEpisodes);
        Assert.True(config.Deterministic);
    }

    [Fact]
    public void RLCheckpointConfig_Defaults_AreExpected()
    {
        var config = new RLCheckpointConfig();

        Assert.Equal("./checkpoints", config.CheckpointDirectory);
        Assert.Equal(100, config.SaveEveryEpisodes);
        Assert.Equal(3, config.KeepBestN);
        Assert.True(config.SaveOnBestReward);
    }

    [Fact]
    public void RLEarlyStoppingConfig_Defaults_AreExpected()
    {
        var config = new RLEarlyStoppingConfig<double>();

        Assert.Equal(100, config.PatienceEpisodes);
        Assert.Equal(0.01, config.MinImprovement, precision: 10);
    }

    [Fact]
    public void TargetNetworkConfig_Defaults_AreExpected()
    {
        var config = new TargetNetworkConfig<double>();

        Assert.Equal(1000, config.UpdateFrequency);
        Assert.False(config.UseSoftUpdate);
        Assert.Equal(0.005, config.Tau, precision: 10);
    }

    [Fact]
    public void ExplorationScheduleConfig_Defaults_AreExpected()
    {
        var config = new ExplorationScheduleConfig<double>();

        Assert.Equal(1.0, config.InitialEpsilon, precision: 10);
        Assert.Equal(0.01, config.FinalEpsilon, precision: 10);
        Assert.Equal(100000, config.DecaySteps);
        Assert.Equal(ExplorationDecayType.Linear, config.DecayType);
    }

    [Fact]
    public void RewardClippingConfig_Defaults_AreExpected()
    {
        var config = new RewardClippingConfig<double>();

        Assert.Equal(-1.0, config.MinReward, precision: 10);
        Assert.Equal(1.0, config.MaxReward, precision: 10);
        Assert.True(config.UseClipping);
    }

    [Fact]
    public void PrioritizedReplayConfig_Defaults_AreExpected()
    {
        var config = new PrioritizedReplayConfig<double>();

        Assert.Equal(0.6, config.Alpha, precision: 10);
        Assert.Equal(0.4, config.InitialBeta, precision: 10);
        Assert.Equal(1.0, config.FinalBeta, precision: 10);
        Assert.Equal(100000, config.BetaAnnealingSteps);
        Assert.Equal(1e-6, config.PriorityEpsilon, precision: 10);
    }

    [Fact]
    public void RLAutoMLOptions_Defaults_AreExpected()
    {
        var options = new RLAutoMLOptions<double>();

        Assert.Equal(50, options.TrainingEpisodesPerTrial);
        Assert.Equal(10, options.EvaluationEpisodesPerTrial);
        Assert.NotNull(options.SearchSpaceOverrides);
    }
}
