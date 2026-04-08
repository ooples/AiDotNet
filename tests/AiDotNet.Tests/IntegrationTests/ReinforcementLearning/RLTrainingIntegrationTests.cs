using System.Collections.Generic;
using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.DQN;
using AiDotNet.ReinforcementLearning.Agents.PPO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class RLTrainingIntegrationTests
{
    [Fact]
    public async Task BuildAsync_DqnAgent_RunsEpisodesAndInvokesCallbacks()
    {
        var environment = new DeterministicBanditEnvironment<double>(
            actionSpaceSize: 2,
            observationSpaceDimension: 1,
            maxSteps: 1);

        var dqnOptions = new DQNOptions<double>
        {
            StateSize = environment.ObservationSpaceDimension,
            ActionSize = environment.ActionSpaceSize,
            BatchSize = 1,
            ReplayBufferSize = 8,
            TargetUpdateFrequency = 1,
            WarmupSteps = 0,
            EpsilonStart = 0.0,
            EpsilonEnd = 0.0,
            EpsilonDecay = 1.0,
            HiddenLayers = new List<int> { 4 },
            Seed = 123
        };

        var agent = new DQNAgent<double>(dqnOptions);

        int stepCallbacks = 0;
        int episodeCallbacks = 0;
        int trainingCallbacks = 0;
        RLTrainingSummary<double>? summary = null;

        var rlOptions = new RLTrainingOptions<double>
        {
            Environment = environment,
            Episodes = 3,
            MaxStepsPerEpisode = 1,
            LogFrequency = 0,
            OnStepComplete = _ => stepCallbacks++,
            OnEpisodeComplete = _ => episodeCallbacks++,
            OnTrainingComplete = metrics =>
            {
                trainingCallbacks++;
                summary = metrics;
            }
        };

        var builder = new AiModelBuilder<double, Vector<double>, Vector<double>>()
            .ConfigureReinforcementLearning(rlOptions)
            .ConfigureModel(agent);

        var result = await builder.BuildAsync();

        Assert.NotNull(result);
        Assert.Equal(3, stepCallbacks);
        Assert.Equal(3, episodeCallbacks);
        Assert.Equal(1, trainingCallbacks);
        Assert.NotNull(summary);
        Assert.Equal(3, summary!.TotalEpisodes);
        Assert.Equal(3, summary.TotalSteps);

        var action = result.Predict(new Vector<double>(environment.ObservationSpaceDimension));
        Assert.Equal(environment.ActionSpaceSize, action.Length);
    }

    [Fact]
    public async Task BuildAsync_PpoAgent_RunsEpisodesAndInvokesCallbacks()
    {
        var environment = new DeterministicBanditEnvironment<double>(
            actionSpaceSize: 2,
            observationSpaceDimension: 1,
            maxSteps: 2);

        var ppoOptions = new PPOOptions<double>
        {
            StateSize = environment.ObservationSpaceDimension,
            ActionSize = environment.ActionSpaceSize,
            StepsPerUpdate = 2,
            MiniBatchSize = 1,
            TrainingEpochs = 1,
            PolicyHiddenLayers = new List<int> { 4 },
            ValueHiddenLayers = new List<int> { 4 },
            Seed = 7
        };

        var agent = new PPOAgent<double>(ppoOptions);

        int stepCallbacks = 0;
        int episodeCallbacks = 0;
        int trainingCallbacks = 0;
        RLTrainingSummary<double>? summary = null;

        var rlOptions = new RLTrainingOptions<double>
        {
            Environment = environment,
            Episodes = 2,
            MaxStepsPerEpisode = 2,
            LogFrequency = 0,
            OnStepComplete = _ => stepCallbacks++,
            OnEpisodeComplete = _ => episodeCallbacks++,
            OnTrainingComplete = metrics =>
            {
                trainingCallbacks++;
                summary = metrics;
            }
        };

        var builder = new AiModelBuilder<double, Vector<double>, Vector<double>>()
            .ConfigureReinforcementLearning(rlOptions)
            .ConfigureModel(agent);

        var result = await builder.BuildAsync();

        Assert.NotNull(result);
        Assert.Equal(4, stepCallbacks);
        Assert.Equal(2, episodeCallbacks);
        Assert.Equal(1, trainingCallbacks);
        Assert.NotNull(summary);
        Assert.Equal(2, summary!.TotalEpisodes);
        Assert.Equal(4, summary.TotalSteps);

        var action = result.Predict(new Vector<double>(environment.ObservationSpaceDimension));
        Assert.Equal(environment.ActionSpaceSize, action.Length);
    }
}
