using System.Collections.Generic;
using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.DQN;
using AiDotNet.ReinforcementLearning.Agents.PPO;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

[Collection("NonParallelIntegration")]
public class RLTrainingIntegrationTests
{
    [Fact(Timeout = 120000)]
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

    [Fact(Timeout = 120000)]
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

    /// <summary>
    /// A configured exploration strategy must actually drive action selection during RL training,
    /// not merely land in an unread field. This stub forces a fixed arm and records every call, so
    /// the assertions prove the training loop consulted it for each action (and ran its Reset/Update
    /// schedule). Without the wiring in BuildRLInternalAsync the agent's own exploration is used and
    /// the stub is never called, so GetExplorationAction stays at 0 and this test fails.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ExplorationStrategyInOptions_DrivesActionSelectionDuringTraining()
    {
        var environment = new DeterministicBanditEnvironment<double>(
            actionSpaceSize: 2,
            observationSpaceDimension: 1,
            maxSteps: 2);

        // Arm 0 yields reward 1, arm 1 yields reward 0. Force arm 1 so a non-zero episode reward
        // would only be possible if the forced action were ignored.
        var strategy = new RecordingExplorationStrategy<double>(forcedActionIndex: 1, actionSpaceSize: 2);

        double lastEpisodeReward = double.NaN;

        var rlOptions = new RLTrainingOptions<double>
        {
            Environment = environment,
            Episodes = 3,
            MaxStepsPerEpisode = 2,
            LogFrequency = 0,
            ExplorationStrategy = strategy,
            OnEpisodeComplete = m => lastEpisodeReward = m.TotalReward
        };

        var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
            .ConfigureReinforcementLearning(rlOptions)
            .ConfigureModel(CreateBanditAgent(environment))
            .BuildAsync();

        Assert.NotNull(result);
        // 3 episodes x 2 steps = 6 action selections, each routed through the strategy.
        Assert.Equal(6, strategy.GetExplorationActionCalls);
        Assert.Equal(6, strategy.UpdateCalls);       // decay advanced once per step
        Assert.Equal(3, strategy.ResetCalls);        // reset once per episode
        Assert.Equal(6, environment.TotalSteps);
        Assert.All(strategy.ObservedActionSpaceSizes, s => Assert.Equal(2, s)); // greedy action length wired through
        // Forced arm 1 -> reward 0 every step -> episode reward exactly 0.
        Assert.Equal(0.0, lastEpisodeReward);
    }

    /// <summary>
    /// The same override delivered via the fluent ConfigureExplorationStrategy() rather than through
    /// the options object must reach training identically. Pins the ConfigureExplorationStrategy wiring.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ConfigureExplorationStrategy_DrivesActionSelectionDuringTraining()
    {
        var environment = new DeterministicBanditEnvironment<double>(
            actionSpaceSize: 2,
            observationSpaceDimension: 1,
            maxSteps: 2);

        var strategy = new RecordingExplorationStrategy<double>(forcedActionIndex: 1, actionSpaceSize: 2);

        var rlOptions = new RLTrainingOptions<double>
        {
            Environment = environment,
            Episodes = 2,
            MaxStepsPerEpisode = 2,
            LogFrequency = 0
        };

        var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
            .ConfigureReinforcementLearning(rlOptions)
            .ConfigureExplorationStrategy(strategy)
            .ConfigureModel(CreateBanditAgent(environment))
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Equal(4, strategy.GetExplorationActionCalls); // 2 episodes x 2 steps
        Assert.Equal(2, strategy.ResetCalls);
        Assert.Equal(4, environment.TotalSteps);
    }

    private static DQNAgent<double> CreateBanditAgent(DeterministicBanditEnvironment<double> environment)
    {
        return new DQNAgent<double>(new DQNOptions<double>
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
        });
    }

    /// <summary>
    /// ConfigureEnvironment before ConfigureReinforcementLearning: the options carry no Environment
    /// of their own, so the one named earlier must survive the wholesale _rlOptions assignment.
    /// Pins the carry-over in ConfigureReinforcementLearning.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ConfigureEnvironment_BeforeRLOptions_TrainsInThatEnvironment()
    {
        var environment = new DeterministicBanditEnvironment<double>(
            actionSpaceSize: 2,
            observationSpaceDimension: 1,
            maxSteps: 1);

        int stepCallbacks = 0;
        int episodeCallbacks = 0;

        // Environment deliberately left null here: it must arrive from ConfigureEnvironment.
        var rlOptions = new RLTrainingOptions<double>
        {
            Episodes = 3,
            MaxStepsPerEpisode = 1,
            LogFrequency = 0,
            OnStepComplete = _ => stepCallbacks++,
            OnEpisodeComplete = _ => episodeCallbacks++
        };

        var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
            .ConfigureEnvironment(environment)
            .ConfigureReinforcementLearning(rlOptions)
            .ConfigureModel(CreateBanditAgent(environment))
            .BuildAsync();

        Assert.NotNull(result);

        // These callbacks fire only from inside the RL training loop, so non-zero counts mean the
        // RL path opened at all. The environment's own counters prove THIS instance was the one
        // driven, rather than some other environment reaching the loop.
        Assert.Equal(3, episodeCallbacks);
        Assert.Equal(3, stepCallbacks);
        Assert.Equal(3, environment.TotalResets);
        Assert.Equal(3, environment.TotalSteps);
    }

    /// <summary>
    /// ConfigureReinforcementLearning before ConfigureEnvironment: the options exist but carry no
    /// Environment, so ConfigureEnvironment must deliver it into them.
    /// Pins the routing in ConfigureEnvironment.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ConfigureEnvironment_AfterRLOptions_TrainsInThatEnvironment()
    {
        var environment = new DeterministicBanditEnvironment<double>(
            actionSpaceSize: 2,
            observationSpaceDimension: 1,
            maxSteps: 1);

        int stepCallbacks = 0;
        int episodeCallbacks = 0;

        var rlOptions = new RLTrainingOptions<double>
        {
            Episodes = 3,
            MaxStepsPerEpisode = 1,
            LogFrequency = 0,
            OnStepComplete = _ => stepCallbacks++,
            OnEpisodeComplete = _ => episodeCallbacks++
        };

        var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
            .ConfigureReinforcementLearning(rlOptions)
            .ConfigureEnvironment(environment)
            .ConfigureModel(CreateBanditAgent(environment))
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Equal(3, episodeCallbacks);
        Assert.Equal(3, stepCallbacks);
        Assert.Equal(3, environment.TotalResets);
        Assert.Equal(3, environment.TotalSteps);
    }

    /// <summary>
    /// An Environment supplied inside RLTrainingOptions is the more specific statement of intent, so
    /// it wins over one named earlier by ConfigureEnvironment rather than being clobbered by the
    /// carry-over. Mirrors ConfigureCurriculumLearning's precedence (options passed in win).
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task ConfigureReinforcementLearning_EnvironmentInOptions_WinsOverEarlierConfigureEnvironment()
    {
        var ignoredEnvironment = new DeterministicBanditEnvironment<double>(
            actionSpaceSize: 2,
            observationSpaceDimension: 1,
            maxSteps: 1);
        var optionsEnvironment = new DeterministicBanditEnvironment<double>(
            actionSpaceSize: 2,
            observationSpaceDimension: 1,
            maxSteps: 1);

        var rlOptions = new RLTrainingOptions<double>
        {
            Environment = optionsEnvironment,
            Episodes = 2,
            MaxStepsPerEpisode = 1,
            LogFrequency = 0
        };

        var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
            .ConfigureEnvironment(ignoredEnvironment)
            .ConfigureReinforcementLearning(rlOptions)
            .ConfigureModel(CreateBanditAgent(optionsEnvironment))
            .BuildAsync();

        Assert.NotNull(result);
        Assert.Equal(2, optionsEnvironment.TotalSteps);
        Assert.Equal(0, ignoredEnvironment.TotalSteps);
    }
}
