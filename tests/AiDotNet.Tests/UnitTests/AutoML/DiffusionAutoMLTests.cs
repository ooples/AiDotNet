using System;
using System.Threading.Tasks;
using AiDotNet.AutoML;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML;

/// <summary>
/// AutoML over latent-diffusion configurations: trials are sampled from the configured predictor and scheduler
/// lists and numeric ranges, reproducibly under a seed, and a trial's configuration survives its dictionary form.
/// </summary>
public class DiffusionAutoMLTests
{
    [Fact]
    public async Task Trials_StayInsideTheConfiguredSearchSpace()
    {
        var automl = new DiffusionAutoML<double>(seed: 11);
        automl.NoisePredictorTypesToTry.Clear();
        automl.NoisePredictorTypesToTry.Add(NoisePredictorType.DiT);

        for (int i = 0; i < 20; i++)
        {
            var config = DiffusionTrialConfig<double>.FromDictionary(await automl.SuggestNextTrialAsync());

            Assert.Equal(NoisePredictorType.DiT, config.NoisePredictorType);
            Assert.Contains(config.SchedulerType, automl.SchedulerTypesToTry);
            Assert.InRange(config.InferenceSteps, 10, 100);
            Assert.InRange(config.GuidanceScale, 1.0, 15.0);
            Assert.InRange(config.LearningRate, 1e-6, 1e-3);
            Assert.InRange(config.BaseChannels, 64, 512);
            Assert.InRange(config.NumResBlocks, 1, 4);
        }
    }

    [Fact]
    public async Task TheSameSeed_SuggestsTheSameTrials()
    {
        var first = new DiffusionAutoML<double>(seed: 5);
        var second = new DiffusionAutoML<double>(seed: 5);

        for (int i = 0; i < 5; i++)
        {
            var a = await first.SuggestNextTrialAsync();
            var b = await second.SuggestNextTrialAsync();
            Assert.Equal(a, b);
        }
    }

    [Theory]
    [InlineData(null)]
    [InlineData(0)]
    [InlineData(17)]
    public void TrialConfig_RoundTripsThroughItsDictionary(int? seed)
    {
        var config = new DiffusionTrialConfig<double>
        {
            NoisePredictorType = NoisePredictorType.DiT,
            SchedulerType = DiffusionSchedulerType.DDIM,
            InferenceSteps = 33,
            GuidanceScale = 4.5,
            LearningRate = 3e-5,
            BaseChannels = 96,
            NumResBlocks = 3,
            LatentDim = 8,
            LatentHeight = 16,
            LatentWidth = 24,
            ConditioningDim = 64,
            ImageChannels = 1,
            NumHeads = 4,
            TransformerDepth = 6,
            ConditioningDropoutProbability = 0.2,
            Seed = seed,
        };

        var restored = DiffusionTrialConfig<double>.FromDictionary(config.ToDictionary());

        Assert.Equal(config.ToDictionary(), restored.ToDictionary());
        Assert.Equal(seed, restored.Seed);
    }
}
