using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Xunit;
using AiDotNet.Data.Formats;
using AiDotNet.Data.Sampling;
using AiDotNet.Data.Pipeline;
using AiDotNet.Data.Loaders;

namespace AiDotNet.Tests.IntegrationTests.Data;

public class AdvancedPipelineTests
{
    // ==================== LMDB Dataset Tests ====================

    [Fact]
    public void LmdbDataset_DefaultOptions()
    {
        var options = new LmdbDatasetOptions();
        Assert.Equal("", options.DataPath);
        Assert.Equal(1L * 1024 * 1024 * 1024, options.MapSize);
        Assert.True(options.ReadOnly);
        Assert.Equal(128, options.MaxReaders);
    }

    [Fact]
    public void LmdbDataset_WriteAndRead()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"lmdb_test_{Guid.NewGuid():N}");
        try
        {
            // Write test data
            var entries = new List<KeyValuePair<string, byte[]>>
            {
                new("sample_0", BitConverter.GetBytes(1.0)),
                new("sample_1", BitConverter.GetBytes(2.0)),
                new("sample_2", BitConverter.GetBytes(3.0))
            };

            LmdbDataset<double>.WriteDataset(tempDir, entries);

            // Read back
            using var dataset = new LmdbDataset<double>(new LmdbDatasetOptions { DataPath = tempDir });
            Assert.Equal(3, dataset.Count);
            Assert.Equal(3, dataset.Keys.Count);

            var value = dataset.GetAsString("sample_0");
            Assert.NotNull(value);

            var byIndex = dataset.GetByIndex(1);
            Assert.NotNull(byIndex);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    // ==================== HDF5 Dataset Tests ====================

    [Fact]
    public void Hdf5Dataset_DefaultOptions()
    {
        var options = new Hdf5DatasetOptions();
        Assert.Equal("", options.FilePath);
        Assert.Equal("features", options.FeaturesDataset);
        Assert.Equal("labels", options.LabelsDataset);
        Assert.Equal(1000, options.ChunkSize);
    }

    [Fact]
    public void Hdf5Dataset_WriteAndRead()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"hdf5_test_{Guid.NewGuid():N}.h5");
        try
        {
            // Write test data
            var features = new double[] { 1, 2, 3, 4, 5, 6 };
            var labels = new double[] { 0, 1, 0 };

            var datasets = new Dictionary<string, (double[] Data, int[] Shape)>
            {
                ["features"] = (features, new[] { 3, 2 }),
                ["labels"] = (labels, new[] { 3 })
            };

            Hdf5Dataset<double>.WriteFile(tempFile, datasets);

            // Read back
            using var hdf5 = new Hdf5Dataset<double>(new Hdf5DatasetOptions { FilePath = tempFile });

            Assert.Equal(2, hdf5.DatasetNames.Count);
            Assert.Contains("features", hdf5.DatasetNames);
            Assert.Contains("labels", hdf5.DatasetNames);

            var shape = hdf5.GetShape("features");
            Assert.Equal(2, shape.Length);
            Assert.Equal(3, shape[0]);
            Assert.Equal(2, shape[1]);

            var featureTensor = hdf5.ReadDataset("features");
            Assert.Equal(new[] { 3, 2 }, featureTensor.Shape);

            var labelTensor = hdf5.ReadDataset("labels");
            Assert.Equal(new[] { 3 }, labelTensor.Shape);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Fact]
    public void Hdf5Dataset_ReadSlice()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"hdf5_slice_{Guid.NewGuid():N}.h5");
        try
        {
            var data = new double[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            var datasets = new Dictionary<string, (double[] Data, int[] Shape)>
            {
                ["data"] = (data, new[] { 4, 2 })
            };

            Hdf5Dataset<double>.WriteFile(tempFile, datasets);

            using var hdf5 = new Hdf5Dataset<double>(new Hdf5DatasetOptions { FilePath = tempFile });

            var slice = hdf5.ReadSlice("data", 1, 2);
            Assert.Equal(new[] { 2, 2 }, slice.Shape);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    // ==================== Arrow Dataset Tests ====================

    [Fact]
    public void ArrowDataset_DefaultOptions()
    {
        var options = new ArrowDatasetOptions();
        Assert.Equal("", options.DataPath);
        Assert.Equal("features", options.FeatureColumn);
        Assert.Equal("label", options.LabelColumn);
        Assert.True(options.MemoryMap);
        Assert.Equal(1024, options.BatchSize);
    }

    [Fact]
    public void ArrowDataset_WriteAndRead()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"arrow_test_{Guid.NewGuid():N}.arrow");
        try
        {
            int numRows = 5;
            var features = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }; // 5 rows, 2 per row
            var labels = new double[] { 0, 1, 0, 1, 0 }; // 5 rows, 1 per row

            var columns = new Dictionary<string, (double[] Data, int ElementsPerRow)>
            {
                ["features"] = (features, 2),
                ["label"] = (labels, 1)
            };

            ArrowDataset<double>.WriteFile(tempFile, columns, numRows);

            using var arrow = new ArrowDataset<double>(new ArrowDatasetOptions { DataPath = tempFile });

            Assert.Equal(5, arrow.NumRows);
            Assert.Equal(2, arrow.ColumnNames.Count);

            var featureData = arrow.ReadColumn("features");
            Assert.Equal(10, featureData.Length);

            var labelData = arrow.ReadColumn("label");
            Assert.Equal(5, labelData.Length);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Fact]
    public void ArrowDataset_ReadBatch()
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"arrow_batch_{Guid.NewGuid():N}.arrow");
        try
        {
            var features = new double[] { 1, 2, 3, 4, 5, 6 };
            var labels = new double[] { 0, 1, 0 };

            var columns = new Dictionary<string, (double[] Data, int ElementsPerRow)>
            {
                ["features"] = (features, 2),
                ["label"] = (labels, 1)
            };

            ArrowDataset<double>.WriteFile(tempFile, columns, 3);

            using var arrow = new ArrowDataset<double>(new ArrowDatasetOptions { DataPath = tempFile });

            var (featureTensor, labelTensor) = arrow.ReadBatch(0, 2);
            Assert.Equal(new[] { 2, 2 }, featureTensor.Shape);
            Assert.Equal(new[] { 2 }, labelTensor.Shape);
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    // ==================== Elastic Distributed Sampler Tests ====================

    [Fact]
    public void ElasticDistributedSampler_DefaultOptions()
    {
        var options = new ElasticDistributedSamplerOptions();
        Assert.Equal(1, options.NumReplicas);
        Assert.Equal(0, options.Rank);
        Assert.True(options.Shuffle);
        Assert.True(options.DropLast);
    }

    [Fact]
    public void ElasticDistributedSampler_SingleWorker()
    {
        var sampler = new ElasticDistributedSampler(new ElasticDistributedSamplerOptions
        {
            DatasetSize = 10,
            NumReplicas = 1,
            Rank = 0,
            Shuffle = false,
            Seed = 42
        });

        Assert.Equal(10, sampler.Length);
        var indices = sampler.GetIndices().ToList();
        Assert.Equal(10, indices.Count);
    }

    [Fact]
    public void ElasticDistributedSampler_TwoWorkers_NonOverlapping()
    {
        var sampler0 = new ElasticDistributedSampler(new ElasticDistributedSamplerOptions
        {
            DatasetSize = 10,
            NumReplicas = 2,
            Rank = 0,
            Shuffle = false,
            Seed = 42
        });

        var sampler1 = new ElasticDistributedSampler(new ElasticDistributedSamplerOptions
        {
            DatasetSize = 10,
            NumReplicas = 2,
            Rank = 1,
            Shuffle = false,
            Seed = 42
        });

        var indices0 = sampler0.GetIndices().ToList();
        var indices1 = sampler1.GetIndices().ToList();

        Assert.Equal(5, indices0.Count);
        Assert.Equal(5, indices1.Count);

        // All indices should be covered
        var all = indices0.Concat(indices1).OrderBy(x => x).ToList();
        Assert.Equal(10, all.Count);
    }

    [Fact]
    public void ElasticDistributedSampler_Rescale()
    {
        var sampler = new ElasticDistributedSampler(new ElasticDistributedSamplerOptions
        {
            DatasetSize = 100,
            NumReplicas = 2,
            Rank = 0,
            Seed = 42
        });

        Assert.Equal(50, sampler.Length);

        sampler.Rescale(4, 1);
        Assert.Equal(4, sampler.NumReplicas);
        Assert.Equal(1, sampler.Rank);
        Assert.Equal(25, sampler.Length);
    }

    // ==================== Mid-Epoch Checkpointer Tests ====================

    [Fact]
    public void MidEpochCheckpointer_DefaultOptions()
    {
        var options = new MidEpochCheckpointerOptions();
        Assert.Equal(100, options.SaveEveryNBatches);
        Assert.Equal(3, options.MaxCheckpoints);
        Assert.Equal("checkpoint", options.FilePrefix);
    }

    [Fact]
    public void MidEpochCheckpointer_SaveAndLoad()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"ckpt_test_{Guid.NewGuid():N}");
        try
        {
            var checkpointer = new MidEpochCheckpointer(new MidEpochCheckpointerOptions
            {
                CheckpointDirectory = tempDir,
                SaveEveryNBatches = 2
            });

            // Simulate batches
            Assert.False(checkpointer.OnBatchComplete(0, 0));
            Assert.True(checkpointer.OnBatchComplete(0, 1, Encoding.UTF8.GetBytes("test_state")));

            // Load latest
            var checkpoint = checkpointer.LoadLatestCheckpoint();
            Assert.NotNull(checkpoint);
            Assert.Equal(0, checkpoint.Epoch);
            Assert.Equal(1, checkpoint.BatchIndex);
            Assert.NotNull(checkpoint.CustomState);
            Assert.Equal("test_state", Encoding.UTF8.GetString(checkpoint.CustomState));
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void MidEpochCheckpointer_RotatesOldCheckpoints()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"ckpt_rotate_{Guid.NewGuid():N}");
        try
        {
            var checkpointer = new MidEpochCheckpointer(new MidEpochCheckpointerOptions
            {
                CheckpointDirectory = tempDir,
                SaveEveryNBatches = 1,
                MaxCheckpoints = 2
            });

            // Save 4 checkpoints
            checkpointer.OnBatchComplete(0, 0);
            checkpointer.OnBatchComplete(0, 1);
            checkpointer.OnBatchComplete(0, 2);
            checkpointer.OnBatchComplete(0, 3);

            // Should only keep 2
            var files = Directory.GetFiles(tempDir, "checkpoint_*.ckpt");
            Assert.True(files.Length <= 2, $"Expected at most 2 checkpoints but found {files.Length}");
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    // ==================== Multi-Source Mixer Tests ====================

    [Fact]
    public void MultiSourceMixer_DefaultOptions()
    {
        var options = new MultiSourceMixerOptions();
        Assert.Null(options.Weights);
        Assert.Null(options.Seed);
        Assert.False(options.StopOnShortestSource);
        Assert.Equal(1000, options.BufferSize);
    }

    [Fact]
    public void MultiSourceMixer_EqualWeights()
    {
        var mixer = new MultiSourceMixer<int>(3, new MultiSourceMixerOptions { Seed = 42 });

        Assert.Equal(3, mixer.NumSources);
        double w0 = mixer.GetWeight(0);
        double w1 = mixer.GetWeight(1);
        double w2 = mixer.GetWeight(2);

        Assert.Equal(w0, w1, 5);
        Assert.Equal(w1, w2, 5);
        Assert.Equal(1.0 / 3.0, w0, 5);
    }

    [Fact]
    public void MultiSourceMixer_CustomWeights()
    {
        var mixer = new MultiSourceMixer<int>(2, new MultiSourceMixerOptions
        {
            Weights = new[] { 3.0, 1.0 },
            Seed = 42
        });

        Assert.Equal(0.75, mixer.GetWeight(0), 5);
        Assert.Equal(0.25, mixer.GetWeight(1), 5);
    }

    [Fact]
    public void MultiSourceMixer_SelectSourceDistribution()
    {
        var mixer = new MultiSourceMixer<int>(2, new MultiSourceMixerOptions
        {
            Weights = new[] { 0.8, 0.2 },
            Seed = 42
        });

        int[] counts = new int[2];
        for (int i = 0; i < 1000; i++)
            counts[mixer.SelectSource()]++;

        // Source 0 should be selected roughly 80% of the time
        double ratio = (double)counts[0] / 1000;
        Assert.True(ratio > 0.7 && ratio < 0.9,
            $"Expected ~80% source 0, got {ratio * 100:F1}%");
    }

    // ==================== Prefetch Data Loader Tests ====================

    [Fact]
    public void PrefetchDataLoader_DefaultOptions()
    {
        var options = new PrefetchDataLoaderOptions();
        Assert.Equal(2, options.PrefetchCount);
        Assert.True(options.UseBackgroundThread);
        Assert.Equal(30000, options.TimeoutMs);
    }

    [Fact]
    public void PrefetchDataLoader_SynchronousMode()
    {
        using var prefetcher = new PrefetchDataLoader<int>(new PrefetchDataLoaderOptions
        {
            UseBackgroundThread = false
        });

        var source = Enumerable.Range(0, 10);
        var result = prefetcher.Prefetch(source).ToList();

        Assert.Equal(10, result.Count);
        for (int i = 0; i < 10; i++)
            Assert.Equal(i, result[i]);
    }

    [Fact]
    public void PrefetchDataLoader_BackgroundMode()
    {
        using var prefetcher = new PrefetchDataLoader<int>(new PrefetchDataLoaderOptions
        {
            PrefetchCount = 3,
            UseBackgroundThread = true
        });

        var source = Enumerable.Range(0, 20);
        var result = prefetcher.Prefetch(source).ToList();

        Assert.Equal(20, result.Count);
        // All items should be present (order preserved from source)
        for (int i = 0; i < 20; i++)
            Assert.Equal(i, result[i]);
    }

    // ==================== Caching Data Loader Tests ====================

    [Fact]
    public void CachingDataLoader_DefaultOptions()
    {
        var options = new CachingDataLoaderOptions();
        Assert.Equal(100, options.MaxCacheSize);
        Assert.False(options.EnableDiskCache);
        Assert.Equal(MemoryCacheEvictionPolicy.LRU, options.EvictionPolicy);
    }

    [Fact]
    public void CachingDataLoader_CachesValues()
    {
        var cache = new CachingDataLoader<int, string>();
        int loadCount = 0;

        string result1 = cache.GetOrLoad(1, k => { loadCount++; return $"value_{k}"; });
        string result2 = cache.GetOrLoad(1, k => { loadCount++; return $"value_{k}"; });

        Assert.Equal("value_1", result1);
        Assert.Equal("value_1", result2);
        Assert.Equal(1, loadCount); // Only loaded once
        Assert.Equal(1, cache.Count);
        Assert.True(cache.HitRatio > 0.4);
    }

    [Fact]
    public void CachingDataLoader_LRUEviction()
    {
        var cache = new CachingDataLoader<int, string>(new CachingDataLoaderOptions
        {
            MaxCacheSize = 3,
            EvictionPolicy = MemoryCacheEvictionPolicy.LRU
        });

        cache.GetOrLoad(1, k => $"v{k}");
        cache.GetOrLoad(2, k => $"v{k}");
        cache.GetOrLoad(3, k => $"v{k}");

        // Access key 1 to make it recent
        cache.GetOrLoad(1, k => $"v{k}");

        // Add key 4 - should evict key 2 (least recently used)
        cache.GetOrLoad(4, k => $"v{k}");

        Assert.Equal(3, cache.Count);
        Assert.True(cache.Contains(1));
        Assert.False(cache.Contains(2)); // Evicted
        Assert.True(cache.Contains(3));
        Assert.True(cache.Contains(4));
    }

    [Fact]
    public void CachingDataLoader_Invalidate()
    {
        var cache = new CachingDataLoader<string, int>();

        cache.GetOrLoad("a", _ => 1);
        Assert.True(cache.Contains("a"));

        bool removed = cache.Invalidate("a");
        Assert.True(removed);
        Assert.False(cache.Contains("a"));
    }

    [Fact]
    public void CachingDataLoader_Clear()
    {
        var cache = new CachingDataLoader<int, int>();

        cache.GetOrLoad(1, k => k * 10);
        cache.GetOrLoad(2, k => k * 10);
        Assert.Equal(2, cache.Count);

        cache.Clear();
        Assert.Equal(0, cache.Count);
        Assert.Equal(0, cache.HitRatio);
    }
}
