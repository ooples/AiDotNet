using System;
using System.Linq;
using AiDotNet.Inference;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Inference;

/// <summary>
/// Integration tests for KV-Cache used in autoregressive transformer inference.
/// Tests append, truncate, sliding window, statistics, and edge cases.
/// </summary>
public class InferenceKVCacheIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region Basic Construction and Configuration

    [Fact]
    public void KVCache_DefaultConfig_HasExpectedProperties()
    {
        var cache = new KVCache<double>(numLayers: 4, numHeads: 8, headDim: 32, maxSeqLen: 128);

        Assert.Equal(0, cache.CurrentLength);
        Assert.Equal(128, cache.MaxLength);
        Assert.Equal(0, cache.CacheHits);
        Assert.Equal(0, cache.CacheMisses);
    }

    [Fact]
    public void KVCacheConfig_ForGPT2_MatchesKnownArchitecture()
    {
        var config = KVCacheConfig.ForModel("gpt2");

        Assert.Equal(12, config.NumLayers);
        Assert.Equal(12, config.NumHeads);
        Assert.Equal(64, config.HeadDimension);
        Assert.Equal(1024, config.MaxSequenceLength);
    }

    [Fact]
    public void KVCacheConfig_ForLlama7B_MatchesKnownArchitecture()
    {
        var config = KVCacheConfig.ForModel("llama-7b");

        Assert.Equal(32, config.NumLayers);
        Assert.Equal(32, config.NumHeads);
        Assert.Equal(128, config.HeadDimension);
        Assert.Equal(4096, config.MaxSequenceLength);
        Assert.Equal(CacheDataType.Float16, config.DataType);
    }

    [Fact]
    public void KVCacheConfig_EstimateMemory_CalculatesCorrectly()
    {
        var config = new KVCacheConfig
        {
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8,
            MaxSequenceLength = 16,
            MaxBatchSize = 1,
            DataType = CacheDataType.Float32
        };

        // elementsPerLayer = batch(1) * heads(4) * seq(16) * dim(8) = 512
        // totalElements = elementsPerLayer * layers(2) * 2(K+V) = 2048
        // bytes = totalElements * sizeof(float32)(4) = 8192
        long expected = 1L * 4 * 16 * 8 * 2 * 2 * 4;
        Assert.Equal(expected, config.EstimateMemoryBytes());
    }

    [Fact]
    public void KVCacheConfig_EstimateMemory_FP16_HalvesSize()
    {
        var configF32 = new KVCacheConfig
        {
            NumLayers = 1, NumHeads = 1, HeadDimension = 1,
            MaxSequenceLength = 1, MaxBatchSize = 1,
            DataType = CacheDataType.Float32
        };
        var configF16 = new KVCacheConfig
        {
            NumLayers = 1, NumHeads = 1, HeadDimension = 1,
            MaxSequenceLength = 1, MaxBatchSize = 1,
            DataType = CacheDataType.Float16
        };

        Assert.Equal(configF32.EstimateMemoryBytes() / 2, configF16.EstimateMemoryBytes());
    }

    #endregion

    #region Append and Retrieve

    [Fact]
    public void Append_SingleToken_IncreasesLength()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 2, headDim: 4, maxSeqLen: 16);

        var keys = CreateKVTensor(batch: 1, heads: 2, seq: 1, dim: 4, seed: 42);
        var values = CreateKVTensor(batch: 1, heads: 2, seq: 1, dim: 4, seed: 43);

        cache.Append(0, keys, values);

        Assert.Equal(1, cache.CurrentLength);
        Assert.Equal(1, cache.CacheMisses);
    }

    [Fact]
    public void Append_MultipleTokens_AccumulatesLength()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 2, headDim: 4, maxSeqLen: 32);

        // Append 3 tokens
        var k1 = CreateKVTensor(1, 2, 3, 4, seed: 42);
        var v1 = CreateKVTensor(1, 2, 3, 4, seed: 43);
        cache.Append(0, k1, v1);

        Assert.Equal(3, cache.CurrentLength);

        // Append 2 more
        var k2 = CreateKVTensor(1, 2, 2, 4, seed: 44);
        var v2 = CreateKVTensor(1, 2, 2, 4, seed: 45);
        cache.Append(0, k2, v2);

        Assert.Equal(5, cache.CurrentLength);
    }

    [Fact]
    public void Append_ReturnsFullCachedSequence()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 2, headDim: 4, maxSeqLen: 16);

        // Append first token
        var k1 = CreateKVTensor(1, 2, 1, 4, seed: 42);
        var v1 = CreateKVTensor(1, 2, 1, 4, seed: 43);
        var (allKeys1, allValues1) = cache.Append(0, k1, v1);

        Assert.Equal(new[] { 1, 2, 1, 4 }, allKeys1.Shape);

        // Append second token - should return both
        var k2 = CreateKVTensor(1, 2, 1, 4, seed: 44);
        var v2 = CreateKVTensor(1, 2, 1, 4, seed: 45);
        var (allKeys2, allValues2) = cache.Append(0, k2, v2);

        Assert.Equal(new[] { 1, 2, 2, 4 }, allKeys2.Shape);
        Assert.Equal(new[] { 1, 2, 2, 4 }, allValues2.Shape);
    }

    [Fact]
    public void Append_PreservesPreviouslyCachedValues()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 8);

        // First append with known values
        var k1Data = new double[] { 1.0, 2.0 };
        var k1 = new Tensor<double>(k1Data, new[] { 1, 1, 1, 2 });
        var v1Data = new double[] { 3.0, 4.0 };
        var v1 = new Tensor<double>(v1Data, new[] { 1, 1, 1, 2 });
        cache.Append(0, k1, v1);

        // Second append
        var k2Data = new double[] { 5.0, 6.0 };
        var k2 = new Tensor<double>(k2Data, new[] { 1, 1, 1, 2 });
        var v2Data = new double[] { 7.0, 8.0 };
        var v2 = new Tensor<double>(v2Data, new[] { 1, 1, 1, 2 });
        var (allKeys, allValues) = cache.Append(0, k2, v2);

        // Verify first token's values are preserved
        Assert.Equal(1.0, allKeys[new[] { 0, 0, 0, 0 }], Tolerance);
        Assert.Equal(2.0, allKeys[new[] { 0, 0, 0, 1 }], Tolerance);
        Assert.Equal(3.0, allValues[new[] { 0, 0, 0, 0 }], Tolerance);
        Assert.Equal(4.0, allValues[new[] { 0, 0, 0, 1 }], Tolerance);

        // Verify second token's values
        Assert.Equal(5.0, allKeys[new[] { 0, 0, 1, 0 }], Tolerance);
        Assert.Equal(6.0, allKeys[new[] { 0, 0, 1, 1 }], Tolerance);
        Assert.Equal(7.0, allValues[new[] { 0, 0, 1, 0 }], Tolerance);
        Assert.Equal(8.0, allValues[new[] { 0, 0, 1, 1 }], Tolerance);
    }

    [Fact]
    public void Append_MultipleLayers_IndependentCaches()
    {
        var cache = new KVCache<double>(numLayers: 3, numHeads: 2, headDim: 4, maxSeqLen: 16);

        // Append to layer 0
        var k0 = CreateKVTensor(1, 2, 2, 4, seed: 42);
        var v0 = CreateKVTensor(1, 2, 2, 4, seed: 43);
        cache.Append(0, k0, v0);

        // Append to layer 1 (different amount)
        var k1 = CreateKVTensor(1, 2, 3, 4, seed: 44);
        var v1 = CreateKVTensor(1, 2, 3, 4, seed: 45);
        cache.Append(1, k1, v1);

        Assert.Equal(2, cache.GetSequenceLength(0));
        // GetSequenceLength uses layer 0 only, so check cache directly
        var (cachedK0, _) = cache.GetCached(0, 1);
        var (cachedK1, _) = cache.GetCached(1, 1);
        Assert.Equal(2, cachedK0.Shape[2]);
        Assert.Equal(3, cachedK1.Shape[2]);
    }

    #endregion

    #region Overflow and Edge Cases

    [Fact]
    public void Append_ExceedsMaxLength_ThrowsInvalidOperation()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 4);

        // Fill to capacity
        var k = CreateKVTensor(1, 1, 4, 2, seed: 42);
        var v = CreateKVTensor(1, 1, 4, 2, seed: 43);
        cache.Append(0, k, v);

        // Try to add one more
        var kExtra = CreateKVTensor(1, 1, 1, 2, seed: 44);
        var vExtra = CreateKVTensor(1, 1, 1, 2, seed: 45);

        Assert.Throws<InvalidOperationException>(() => cache.Append(0, kExtra, vExtra));
    }

    [Fact]
    public void Append_InvalidLayerIndex_ThrowsArgumentOutOfRange()
    {
        var cache = new KVCache<double>(numLayers: 2, numHeads: 1, headDim: 2, maxSeqLen: 8);
        var k = CreateKVTensor(1, 1, 1, 2, seed: 42);
        var v = CreateKVTensor(1, 1, 1, 2, seed: 43);

        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Append(-1, k, v));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Append(2, k, v));
    }

    [Fact]
    public void Append_MismatchedShapes_ThrowsArgumentException()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 2, headDim: 4, maxSeqLen: 8);

        var keys = CreateKVTensor(1, 2, 1, 4, seed: 42);
        var values = CreateKVTensor(1, 2, 2, 4, seed: 43); // different seq length!

        Assert.Throws<ArgumentException>(() => cache.Append(0, keys, values));
    }

    [Fact]
    public void Append_WrongNumHeads_ThrowsArgumentException()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 4, headDim: 8, maxSeqLen: 16);

        var keys = CreateKVTensor(1, 2, 1, 8, seed: 42); // 2 heads instead of 4
        var values = CreateKVTensor(1, 2, 1, 8, seed: 43);

        Assert.Throws<ArgumentException>(() => cache.Append(0, keys, values));
    }

    #endregion

    #region Truncate

    [Fact]
    public void Truncate_ReducesCacheLength()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);

        var k = CreateKVTensor(1, 1, 5, 2, seed: 42);
        var v = CreateKVTensor(1, 1, 5, 2, seed: 43);
        cache.Append(0, k, v);
        Assert.Equal(5, cache.CurrentLength);

        cache.Truncate(3);
        Assert.Equal(3, cache.CurrentLength);
    }

    [Fact]
    public void Truncate_LargerThanCurrent_DoesNotExpand()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);

        var k = CreateKVTensor(1, 1, 3, 2, seed: 42);
        var v = CreateKVTensor(1, 1, 3, 2, seed: 43);
        cache.Append(0, k, v);

        cache.Truncate(10); // larger than current 3
        Assert.Equal(3, cache.CurrentLength);
    }

    [Fact]
    public void Truncate_ToZero_ClearsLength()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);

        var k = CreateKVTensor(1, 1, 5, 2, seed: 42);
        var v = CreateKVTensor(1, 1, 5, 2, seed: 43);
        cache.Append(0, k, v);

        cache.Truncate(0);
        Assert.Equal(0, cache.CurrentLength);
    }

    [Fact]
    public void Truncate_NegativeLength_Throws()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);

        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Truncate(-1));
    }

    #endregion

    #region Clear

    [Fact]
    public void Clear_ResetsAllState()
    {
        var cache = new KVCache<double>(numLayers: 2, numHeads: 2, headDim: 4, maxSeqLen: 16);

        for (int layer = 0; layer < 2; layer++)
        {
            var k = CreateKVTensor(1, 2, 3, 4, seed: 42 + layer);
            var v = CreateKVTensor(1, 2, 3, 4, seed: 52 + layer);
            cache.Append(layer, k, v);
        }

        cache.Clear();

        Assert.Equal(0, cache.CurrentLength);
        Assert.Equal(0, cache.CacheHits);
        Assert.Equal(0, cache.CacheMisses);
        Assert.Equal(0, cache.Evictions);
    }

    [Fact]
    public void Clear_SpecificBatch_OnlyResetsThatBatch()
    {
        var config = new KVCacheConfig
        {
            NumLayers = 1,
            NumHeads = 1,
            HeadDimension = 2,
            MaxSequenceLength = 16,
            MaxBatchSize = 2,
            PreAllocate = true
        };
        var cache = new KVCache<double>(config);

        // Fill batch 0 and batch 1
        var k = CreateKVTensor(2, 1, 3, 2, seed: 42);
        var v = CreateKVTensor(2, 1, 3, 2, seed: 43);
        cache.Append(0, k, v);

        // Clear only batch 0
        cache.Clear(0);

        Assert.Equal(0, cache.GetSequenceLength(0));
        Assert.Equal(3, cache.GetSequenceLength(1));
    }

    #endregion

    #region Sliding Window

    [Fact]
    public void SlidingWindow_EvictsOldTokens()
    {
        var config = new KVCacheConfig
        {
            NumLayers = 1,
            NumHeads = 1,
            HeadDimension = 2,
            MaxSequenceLength = 32,
            MaxBatchSize = 1,
            UseSlidingWindow = true,
            WindowSize = 4,
            PreAllocate = true
        };
        var cache = new KVCache<double>(config);

        // Fill window (4 tokens)
        var k1 = CreateKVTensor(1, 1, 4, 2, seed: 42);
        var v1 = CreateKVTensor(1, 1, 4, 2, seed: 43);
        cache.Append(0, k1, v1);
        Assert.Equal(4, cache.CurrentLength);

        // Add one more - should evict 1
        var k2 = CreateKVTensor(1, 1, 1, 2, seed: 44);
        var v2 = CreateKVTensor(1, 1, 1, 2, seed: 45);
        cache.Append(0, k2, v2);

        Assert.Equal(4, cache.CurrentLength); // stays at window size
        Assert.True(cache.Evictions > 0, "Should have evictions");
    }

    #endregion

    #region Statistics

    [Fact]
    public void Statistics_TracksCacheHitsAndMisses()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);

        var k = CreateKVTensor(1, 1, 3, 2, seed: 42);
        var v = CreateKVTensor(1, 1, 3, 2, seed: 43);
        cache.Append(0, k, v); // 3 misses (new tokens)

        Assert.Equal(3, cache.CacheMisses);

        // GetCached returns all cached entries
        var (_, _) = cache.GetCached(0, 1);
        Assert.True(cache.CacheHits > 0);
    }

    [Fact]
    public void GetStatistics_ReturnsComprehensiveInfo()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 2, headDim: 4, maxSeqLen: 16);

        var k = CreateKVTensor(1, 2, 2, 4, seed: 42);
        var v = CreateKVTensor(1, 2, 2, 4, seed: 43);
        cache.Append(0, k, v);

        var stats = cache.GetStatistics();

        Assert.True(stats.ContainsKey("CacheHits"));
        Assert.True(stats.ContainsKey("CacheMisses"));
        Assert.True(stats.ContainsKey("HitRate"));
        Assert.True(stats.ContainsKey("CurrentMemoryMB"));
    }

    [Fact]
    public void GetCurrentMemoryUsage_IsPositiveAfterAppend()
    {
        var cache = new KVCache<double>(numLayers: 1, numHeads: 2, headDim: 4, maxSeqLen: 16);

        var k = CreateKVTensor(1, 2, 1, 4, seed: 42);
        var v = CreateKVTensor(1, 2, 1, 4, seed: 43);
        cache.Append(0, k, v);

        Assert.True(cache.GetCurrentMemoryUsage() > 0);
    }

    #endregion

    #region CopyBatchState

    [Fact]
    public void CopyBatchState_DuplicatesCorrectly()
    {
        var config = new KVCacheConfig
        {
            NumLayers = 1,
            NumHeads = 1,
            HeadDimension = 2,
            MaxSequenceLength = 16,
            MaxBatchSize = 2,
            PreAllocate = true
        };
        var cache = new KVCache<double>(config);

        // Append to batch item 0 only (via batch-size-1 tensor, then copy)
        var kData = new double[] { 1.0, 2.0, 3.0, 4.0 }; // batch=1, heads=1, seq=2, dim=2
        var vData = new double[] { 5.0, 6.0, 7.0, 8.0 };
        // Need batch=2 but only populate batch 0
        var k = new Tensor<double>(new double[]
        {
            1.0, 2.0, 3.0, 4.0,  // batch 0
            0.0, 0.0, 0.0, 0.0,  // batch 1
        }, new[] { 2, 1, 2, 2 });
        var v = new Tensor<double>(new double[]
        {
            5.0, 6.0, 7.0, 8.0,
            0.0, 0.0, 0.0, 0.0,
        }, new[] { 2, 1, 2, 2 });
        cache.Append(0, k, v);

        // Copy batch 0 -> batch 1
        cache.CopyBatchState(0, 1);

        Assert.Equal(cache.GetSequenceLength(0), cache.GetSequenceLength(1));
    }

    #endregion

    #region PreAllocate Config

    [Fact]
    public void PreAllocate_True_AllocatesMemoryImmediately()
    {
        var config = new KVCacheConfig
        {
            NumLayers = 1,
            NumHeads = 1,
            HeadDimension = 2,
            MaxSequenceLength = 8,
            MaxBatchSize = 1,
            PreAllocate = true
        };
        var cache = new KVCache<double>(config);

        Assert.True(cache.GetCurrentMemoryUsage() > 0);
    }

    [Fact]
    public void PreAllocate_False_DelaysAllocation()
    {
        var config = new KVCacheConfig
        {
            NumLayers = 1,
            NumHeads = 1,
            HeadDimension = 2,
            MaxSequenceLength = 8,
            MaxBatchSize = 1,
            PreAllocate = false
        };
        var cache = new KVCache<double>(config);

        Assert.Equal(0, cache.GetCurrentMemoryUsage());

        // After first append, should allocate
        var k = CreateKVTensor(1, 1, 1, 2, seed: 42);
        var v = CreateKVTensor(1, 1, 1, 2, seed: 43);
        cache.Append(0, k, v);

        Assert.True(cache.GetCurrentMemoryUsage() > 0);
    }

    #endregion

    #region Autoregressive Simulation

    [Fact]
    public void AutoregressiveGeneration_SimulateTokenByToken()
    {
        // Simulate generating 10 tokens one at a time
        var cache = new KVCache<double>(numLayers: 2, numHeads: 4, headDim: 8, maxSeqLen: 32);

        for (int step = 0; step < 10; step++)
        {
            for (int layer = 0; layer < 2; layer++)
            {
                var k = CreateKVTensor(1, 4, 1, 8, seed: step * 100 + layer);
                var v = CreateKVTensor(1, 4, 1, 8, seed: step * 100 + layer + 50);
                var (allK, allV) = cache.Append(layer, k, v);

                // Returned sequence should grow by 1 each step
                Assert.Equal(step + 1, allK.Shape[2]);
                Assert.Equal(step + 1, allV.Shape[2]);
            }
        }

        Assert.Equal(10, cache.CurrentLength);
    }

    [Fact]
    public void AutoregressiveGeneration_PrefillThenDecode()
    {
        // First: prefill with prompt (multiple tokens at once)
        // Then: decode one token at a time
        var cache = new KVCache<double>(numLayers: 1, numHeads: 2, headDim: 4, maxSeqLen: 32);

        // Prefill with 5 tokens
        var kPrefill = CreateKVTensor(1, 2, 5, 4, seed: 42);
        var vPrefill = CreateKVTensor(1, 2, 5, 4, seed: 43);
        cache.Append(0, kPrefill, vPrefill);
        Assert.Equal(5, cache.CurrentLength);

        // Decode 3 more tokens one at a time
        for (int i = 0; i < 3; i++)
        {
            var kDecode = CreateKVTensor(1, 2, 1, 4, seed: 100 + i);
            var vDecode = CreateKVTensor(1, 2, 1, 4, seed: 200 + i);
            var (allK, _) = cache.Append(0, kDecode, vDecode);

            Assert.Equal(5 + i + 1, allK.Shape[2]);
        }

        Assert.Equal(8, cache.CurrentLength);
    }

    #endregion

    #region Helpers

    private static Tensor<double> CreateKVTensor(int batch, int heads, int seq, int dim, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var data = new double[batch * heads * seq * dim];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble() * 2.0 - 1.0;
        }
        return new Tensor<double>(data, new[] { batch, heads, seq, dim });
    }

    #endregion
}
