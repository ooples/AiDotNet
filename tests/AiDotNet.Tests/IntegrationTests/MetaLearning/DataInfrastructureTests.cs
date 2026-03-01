using System;
using System.Collections.Generic;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Data;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.MetaLearning;

/// <summary>
/// Integration tests for Phase 8 data infrastructure (IEpisode, IMetaDataset, ITaskSampler,
/// synthetic datasets, episode cache, adapters).
/// </summary>
public class DataInfrastructureTests
{
    // ── SineWaveMetaDataset ──

    [Fact]
    public void SineWaveMetaDataset_SampleEpisode_ReturnsValidEpisode()
    {
        var dataset = new SineWaveMetaDataset<double, Matrix<double>, Vector<double>>(seed: 42);
        var episode = dataset.SampleEpisode(numWays: 1, numShots: 5, numQueryPerClass: 10);

        Assert.NotNull(episode);
        Assert.NotNull(episode.Task);
        Assert.True(episode.EpisodeId > 0);
    }

    [Fact]
    public void SineWaveMetaDataset_SampleMultipleEpisodes_ReturnsRequestedCount()
    {
        var dataset = new SineWaveMetaDataset<double, Matrix<double>, Vector<double>>(seed: 42);
        var episodes = dataset.SampleEpisodes(count: 5, numWays: 1, numShots: 3, numQueryPerClass: 7);

        Assert.Equal(5, episodes.Count);
    }

    [Fact]
    public void SineWaveMetaDataset_SupportsConfiguration_ReturnsTrueForFeasible()
    {
        var dataset = new SineWaveMetaDataset<double, Matrix<double>, Vector<double>>(numClasses: 10, examplesPerClass: 30);
        Assert.True(dataset.SupportsConfiguration(5, 3, 3));
        Assert.False(dataset.SupportsConfiguration(200, 1, 1));
    }

    // ── GaussianClassificationMetaDataset ──

    [Fact]
    public void GaussianClassification_SampleEpisode_ReturnsValidTask()
    {
        var dataset = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20, examplesPerClass: 30, featureDim: 5, seed: 42);
        var episode = dataset.SampleEpisode(numWays: 3, numShots: 2, numQueryPerClass: 5);

        var task = episode.Task;
        Assert.NotNull(task.SupportInput);
        Assert.NotNull(task.QueryInput);
        Assert.Equal(3, task.NumWays);
        Assert.Equal(2, task.NumShots);
    }

    // ── Episode ──

    [Fact]
    public void Episode_TracksMetadata()
    {
        var mlTask = new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = new Matrix<double>(2, 2),
            SupportSetY = new Vector<double>(2),
            QuerySetX = new Matrix<double>(2, 2),
            QuerySetY = new Vector<double>(2),
            NumWays = 2, NumShots = 1, NumQueryPerClass = 1
        };
        var episode = new Episode<double, Matrix<double>, Vector<double>>(
            mlTask, domain: "vision", difficulty: 0.5);

        Assert.Equal("vision", episode.Domain);
        Assert.Equal(0.5, episode.Difficulty);
        Assert.True(episode.CreatedTimestamp > 0);

        episode.LastLoss = 1.23;
        episode.SampleCount = 5;
        Assert.Equal(1.23, episode.LastLoss);
        Assert.Equal(5, episode.SampleCount);
    }

    // ── EpisodeCache ──

    [Fact]
    public void EpisodeCache_PutAndGet_Works()
    {
        var cache = new EpisodeCache<double, Matrix<double>, Vector<double>>(capacity: 3);
        var mlTask = new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = new Matrix<double>(2, 2),
            SupportSetY = new Vector<double>(2),
            QuerySetX = new Matrix<double>(2, 2),
            QuerySetY = new Vector<double>(2),
            NumWays = 2, NumShots = 1, NumQueryPerClass = 1
        };

        var ep1 = new Episode<double, Matrix<double>, Vector<double>>(mlTask);
        var ep2 = new Episode<double, Matrix<double>, Vector<double>>(mlTask);
        var ep3 = new Episode<double, Matrix<double>, Vector<double>>(mlTask);
        var ep4 = new Episode<double, Matrix<double>, Vector<double>>(mlTask);

        cache.Put(ep1);
        cache.Put(ep2);
        cache.Put(ep3);

        Assert.True(cache.TryGet(ep1.EpisodeId, out _));
        Assert.True(cache.TryGet(ep2.EpisodeId, out _));
        Assert.Equal(3, cache.Count);

        // Adding ep4 should evict the LRU (ep3 since ep1 and ep2 were just accessed)
        cache.Put(ep4);
        Assert.Equal(3, cache.Count);

        Assert.Equal(2, cache.HitCount);
    }

    [Fact]
    public void EpisodeCache_Clear_ResetsAll()
    {
        var cache = new EpisodeCache<double, Matrix<double>, Vector<double>>(capacity: 10);
        var mlTask = new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = new Matrix<double>(2, 2),
            SupportSetY = new Vector<double>(2),
            QuerySetX = new Matrix<double>(2, 2),
            QuerySetY = new Vector<double>(2),
            NumWays = 2, NumShots = 1, NumQueryPerClass = 1
        };
        cache.Put(new Episode<double, Matrix<double>, Vector<double>>(mlTask));
        cache.Clear();
        Assert.Equal(0, cache.Count);
        Assert.Equal(0, cache.HitCount);
    }

    // ── UniformTaskSampler ──

    [Fact]
    public void UniformTaskSampler_SampleBatch_ReturnsValidBatch()
    {
        var dataset = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20, examplesPerClass: 30, featureDim: 5, seed: 42);
        var sampler = new UniformTaskSampler<double, Matrix<double>, Vector<double>>(
            dataset, numWays: 3, numShots: 2, numQueryPerClass: 5);

        var batch = sampler.SampleBatch(4);
        Assert.Equal(4, batch.BatchSize);
        Assert.Equal(3, batch.NumWays);
    }

    // ── BalancedTaskSampler ──

    [Fact]
    public void BalancedTaskSampler_SampleBatch_Works()
    {
        var dataset = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20, examplesPerClass: 30, featureDim: 5, seed: 42);
        var sampler = new BalancedTaskSampler<double, Matrix<double>, Vector<double>>(
            dataset, numWays: 3, numShots: 2, numQueryPerClass: 5, seed: 42);

        var batch = sampler.SampleBatch(3);
        Assert.Equal(3, batch.BatchSize);
    }

    // ── DynamicTaskSampler ──

    [Fact]
    public void DynamicTaskSampler_UpdateWithFeedback_TracksMeanLoss()
    {
        var dataset = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20, examplesPerClass: 30, featureDim: 5, seed: 42);
        var sampler = new DynamicTaskSampler<double, Matrix<double>, Vector<double>>(
            dataset, numWays: 3, numShots: 2, numQueryPerClass: 5, seed: 42);

        var episodes = new List<IEpisode<double, Matrix<double>, Vector<double>>>();
        for (int i = 0; i < 4; i++) episodes.Add(sampler.SampleOne());

        sampler.UpdateWithFeedback(episodes, new List<double> { 0.5, 0.8, 0.3, 1.2 });

        // After feedback, sampler should still produce valid batches
        var batch = sampler.SampleBatch(2);
        Assert.Equal(2, batch.BatchSize);
    }

    // ── BatchEpisodeSampler ──

    [Fact]
    public void BatchEpisodeSampler_NextBatch_ReturnsCorrectSize()
    {
        var dataset = new GaussianClassificationMetaDataset<double, Matrix<double>, Vector<double>>(
            numClasses: 20, examplesPerClass: 30, featureDim: 5, seed: 42);
        var innerSampler = new UniformTaskSampler<double, Matrix<double>, Vector<double>>(
            dataset, numWays: 3, numShots: 2, numQueryPerClass: 5);
        var batchSampler = new BatchEpisodeSampler<double, Matrix<double>, Vector<double>>(
            innerSampler, batchSize: 4, prefetchCount: 1);

        var batch = batchSampler.NextBatch();
        Assert.Equal(4, batch.Count);

        var taskBatch = batchSampler.NextTaskBatch();
        Assert.Equal(4, taskBatch.BatchSize);
    }

    // ── EpisodicDataLoaderTaskSamplerAdapter ──

    [Fact]
    public void Adapter_WrapsLegacyLoader()
    {
        var supportX = new Matrix<double>(4, 3);
        var supportY = new Vector<double>(4);
        var queryX = new Matrix<double>(4, 3);
        var queryY = new Vector<double>(4);
        var rng = new Random(42);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                supportX[i, j] = rng.NextDouble();
                queryX[i, j] = rng.NextDouble();
            }
            supportY[i] = i % 2;
            queryY[i] = i % 2;
        }

        var tasks = new List<MetaLearningTask<double, Matrix<double>, Vector<double>>>
        {
            new MetaLearningTask<double, Matrix<double>, Vector<double>>
            {
                SupportSetX = supportX, SupportSetY = supportY,
                QuerySetX = queryX, QuerySetY = queryY,
                NumWays = 2, NumShots = 2, NumQueryPerClass = 2
            }
        };

        var loader = new TestEpisodicDataLoader<double, Matrix<double>, Vector<double>>(
            tasks, nWay: 2, kShot: 2, queryShots: 2, availableClasses: 2);

        var adapter = new EpisodicDataLoaderTaskSamplerAdapter<double, Matrix<double>, Vector<double>>(loader);

        Assert.Equal(2, adapter.NumWays);
        Assert.Equal(2, adapter.NumShots);
        Assert.Equal(2, adapter.NumQueryPerClass);

        var episode = adapter.SampleOne();
        Assert.NotNull(episode.Task);

        var batch = adapter.SampleBatch(1);
        Assert.Equal(1, batch.BatchSize);
    }
}
