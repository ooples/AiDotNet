using System.Threading.Tasks;
using AiDotNet.Data.Loaders.RL;
using AiDotNet.ReinforcementLearning.Environments;
using AiDotNet.ReinforcementLearning.ReplayBuffers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ReinforcementLearning;

public class RLDataLoaderIntegrationTests
{
    [Fact]
    public void EnvironmentDataLoader_RunEpisode_CollectsExperiences()
    {
        var environment = new DeterministicBanditEnvironment<double>(
            actionSpaceSize: 2,
            observationSpaceDimension: 1,
            maxSteps: 2);

        var loader = new EnvironmentDataLoader<double>(
            environment,
            episodes: 1,
            maxStepsPerEpisode: 2,
            replayBufferCapacity: 10,
            minExperiencesBeforeTraining: 0,
            verbose: false,
            seed: 3);

        var result = loader.RunEpisode();

        Assert.Equal(2, result.Steps);
        Assert.Equal(2, loader.TotalSteps);
        Assert.Equal(2, loader.ReplayBuffer.Count);
        Assert.InRange(result.TotalReward, 0.0, 2.0);
    }

    [Fact]
    public void EnvironmentDataLoader_RunEpisodes_ReturnsRequestedCount()
    {
        var environment = new DeterministicBanditEnvironment<double>(maxSteps: 1);
        var loader = new EnvironmentDataLoader<double>(
            environment,
            episodes: 3,
            maxStepsPerEpisode: 1,
            replayBufferCapacity: 10,
            minExperiencesBeforeTraining: 0,
            verbose: false,
            seed: 5);

        var results = loader.RunEpisodes(3);

        Assert.Equal(3, results.Count);
        Assert.Equal(3, loader.CurrentEpisode);
        Assert.Equal(3, loader.TotalSteps);
    }

    [Fact]
    public void EnvironmentDataLoader_GetNextBatch_ReturnsExperience()
    {
        var environment = new DeterministicBanditEnvironment<double>(maxSteps: 1);
        var loader = new EnvironmentDataLoader<double>(
            environment,
            episodes: 1,
            maxStepsPerEpisode: 1,
            replayBufferCapacity: 10,
            minExperiencesBeforeTraining: 0,
            verbose: false,
            seed: 7);

        loader.RunEpisode();
        loader.BatchSize = 1;

        var batch = loader.GetNextBatch();

        Assert.NotNull(batch);
        Assert.Equal(1, batch.State.Length);
        Assert.Equal(1, batch.NextState.Length);
    }

    [Fact]
    public void EnvironmentDataLoader_TryGetNextBatch_FalseWhenEmpty()
    {
        var environment = new DeterministicBanditEnvironment<double>(maxSteps: 1);
        var loader = new EnvironmentDataLoader<double>(
            environment,
            episodes: 1,
            maxStepsPerEpisode: 1,
            replayBufferCapacity: 10,
            minExperiencesBeforeTraining: 0,
            verbose: false,
            seed: 11);

        var success = loader.TryGetNextBatch(out var batch);

        Assert.False(success);
        Assert.Equal(0, batch.State.Length);
        Assert.Equal(0, batch.NextState.Length);
    }

    [Fact]
    public async Task EnvironmentDataLoader_LoadAndUnload_TogglesLoadedState()
    {
        var environment = new DeterministicBanditEnvironment<double>(maxSteps: 1);
        var loader = new EnvironmentDataLoader<double>(
            environment,
            episodes: 1,
            maxStepsPerEpisode: 1,
            replayBufferCapacity: 10,
            minExperiencesBeforeTraining: 0,
            verbose: false,
            seed: 13);

        await loader.LoadAsync();

        Assert.True(loader.IsLoaded);

        loader.Unload();

        Assert.False(loader.IsLoaded);
    }

    [Fact]
    public async Task EnvironmentDataLoader_GetBatchesAsync_ReturnsSamples()
    {
        var environment = new DeterministicBanditEnvironment<double>(maxSteps: 2);
        var loader = new EnvironmentDataLoader<double>(
            environment,
            episodes: 1,
            maxStepsPerEpisode: 2,
            replayBufferCapacity: 10,
            minExperiencesBeforeTraining: 0,
            verbose: false,
            seed: 17);

        loader.RunEpisode();

        int count = 0;
        await foreach (var _ in loader.GetBatchesAsync(batchSize: 1, prefetchCount: 1))
        {
            count++;
        }

        Assert.True(count >= 1);
    }
}
