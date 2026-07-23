using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Models.Options;
using AiDotNet.ReinforcementLearning.Agents.DQN;
using AiDotNet.ReinforcementLearning.Environments;
using AiDotNet.ReinforcementLearning.IntrinsicMotivation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

/// <summary>
/// Covers curiosity (intrinsic-motivation) exploration: Random Network Distillation gives higher novelty
/// to unseen states than to familiarized ones, and ConfigureCuriosity wires it into RL training.
/// </summary>
public class CuriosityIntrinsicRewardTests
{
    [Fact]
    public void RandomNetworkDistillation_FamiliarStateBecomesLessNovelThanUnseen()
    {
        var rnd = new RandomNetworkDistillation<double>(seed: 42);
        var stateA = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var stateB = new Vector<double>(new[] { -3.0, 0.5, 4.0 });

        // Familiarize the module with state A by learning it many times.
        for (int i = 0; i < 200; i++) rnd.Update(stateA);

        double noveltyA = rnd.ComputeIntrinsicReward(stateA); // now familiar
        double noveltyB = rnd.ComputeIntrinsicReward(stateB); // never seen

        Assert.True(noveltyB > noveltyA,
            $"an unseen state should be more novel than a familiarized one (A={noveltyA:F4}, B={noveltyB:F4})");
    }

    [Fact]
    public void RandomNetworkDistillation_NoveltyDropsAfterLearningTheSameState()
    {
        var rnd = new RandomNetworkDistillation<double>(seed: 7);
        var state = new Vector<double>(new[] { 0.4, -0.2, 1.1, 0.9 });

        double before = rnd.ComputeIntrinsicReward(state);
        for (int i = 0; i < 300; i++) rnd.Update(state);
        double after = rnd.ComputeIntrinsicReward(state);

        Assert.True(after < before,
            $"repeatedly learning a state should reduce its novelty (before={before:F4}, after={after:F4})");
    }

    [Fact(Timeout = 120000)]
    public async Task ConfigureCuriosity_SurfacesMeanIntrinsicRewardOnResult()
    {
        var environment = new DeterministicBanditEnvironment<double>(
            actionSpaceSize: 2, observationSpaceDimension: 1, maxSteps: 1);

        var agent = new DQNAgent<double>(new DQNOptions<double>
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
            Seed = 123,
        });

        var rlOptions = new RLTrainingOptions<double>
        {
            Environment = environment,
            Episodes = 3,
            MaxStepsPerEpisode = 1,
            LogFrequency = 0,
        };

        var result = await new AiModelBuilder<double, Vector<double>, Vector<double>>()
            .ConfigureReinforcementLearning(rlOptions)
            .ConfigureModel(agent)
            .ConfigureCuriosity() // null → Random Network Distillation default
            .BuildAsync();

        Assert.True(result.ConfiguredMetrics.ContainsKey("MeanIntrinsicReward"),
            "curiosity ran but the mean intrinsic reward was not surfaced");
    }
}
