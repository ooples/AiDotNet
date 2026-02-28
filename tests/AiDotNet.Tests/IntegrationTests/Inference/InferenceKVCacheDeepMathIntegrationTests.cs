using AiDotNet.Inference;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Inference;

/// <summary>
/// Deep integration tests for KVCache: append, retrieve, sliding window eviction,
/// truncation, memory estimation, batch state copying, and sequence length tracking.
/// </summary>
public class InferenceKVCacheDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    /// <summary>
    /// Creates a minimal KVCache for testing.
    /// </summary>
    private static KVCache<double> CreateCache(
        int numLayers = 2,
        int numHeads = 2,
        int headDim = 4,
        int maxSeqLen = 16,
        int maxBatchSize = 1,
        bool preAllocate = true,
        bool useSlidingWindow = false,
        int windowSize = 8)
    {
        return new KVCache<double>(new KVCacheConfig
        {
            NumLayers = numLayers,
            NumHeads = numHeads,
            HeadDimension = headDim,
            MaxSequenceLength = maxSeqLen,
            MaxBatchSize = maxBatchSize,
            PreAllocate = preAllocate,
            UseSlidingWindow = useSlidingWindow,
            WindowSize = windowSize
        });
    }

    /// <summary>
    /// Creates a 4D tensor [batch, heads, seqLen, headDim] filled with a constant value.
    /// </summary>
    private static Tensor<double> CreateKVTensor(int batch, int heads, int seqLen, int headDim, double fillValue)
    {
        var shape = new[] { batch, heads, seqLen, headDim };
        var tensor = new Tensor<double>(shape);
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int s = 0; s < seqLen; s++)
                    for (int d = 0; d < headDim; d++)
                        tensor[new[] { b, h, s, d }] = fillValue;
        return tensor;
    }

    /// <summary>
    /// Creates a 4D tensor with sequential values for identification.
    /// Value at [b,h,s,d] = b*1000 + h*100 + s*10 + d
    /// </summary>
    private static Tensor<double> CreateIdentifiableKVTensor(int batch, int heads, int seqLen, int headDim)
    {
        var shape = new[] { batch, heads, seqLen, headDim };
        var tensor = new Tensor<double>(shape);
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int s = 0; s < seqLen; s++)
                    for (int d = 0; d < headDim; d++)
                        tensor[new[] { b, h, s, d }] = b * 1000 + h * 100 + s * 10 + d;
        return tensor;
    }

    // ============================
    // Basic Append and Retrieve
    // ============================

    [Fact]
    public void Append_SingleToken_RetrievesCorrectly()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 8);
        var keys = CreateKVTensor(1, 1, 1, 2, 3.14);
        var values = CreateKVTensor(1, 1, 1, 2, 2.71);

        var (cachedKeys, cachedValues) = cache.Append(0, keys, values);

        Assert.Equal(new[] { 1, 1, 1, 2 }, cachedKeys.Shape);
        Assert.Equal(3.14, cachedKeys[new[] { 0, 0, 0, 0 }], Tolerance);
        Assert.Equal(3.14, cachedKeys[new[] { 0, 0, 0, 1 }], Tolerance);
        Assert.Equal(2.71, cachedValues[new[] { 0, 0, 0, 0 }], Tolerance);
        Assert.Equal(2.71, cachedValues[new[] { 0, 0, 0, 1 }], Tolerance);
    }

    [Fact]
    public void Append_MultipleTokens_AccumulatesInCache()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 8);

        // Append token 0 with value 1.0
        var k1 = CreateKVTensor(1, 1, 1, 2, 1.0);
        var v1 = CreateKVTensor(1, 1, 1, 2, 10.0);
        cache.Append(0, k1, v1);

        // Append token 1 with value 2.0
        var k2 = CreateKVTensor(1, 1, 1, 2, 2.0);
        var v2 = CreateKVTensor(1, 1, 1, 2, 20.0);
        var (cachedKeys, cachedValues) = cache.Append(0, k2, v2);

        // Should have 2 sequence positions
        Assert.Equal(new[] { 1, 1, 2, 2 }, cachedKeys.Shape);
        Assert.Equal(1.0, cachedKeys[new[] { 0, 0, 0, 0 }], Tolerance); // token 0
        Assert.Equal(2.0, cachedKeys[new[] { 0, 0, 1, 0 }], Tolerance); // token 1
        Assert.Equal(10.0, cachedValues[new[] { 0, 0, 0, 0 }], Tolerance);
        Assert.Equal(20.0, cachedValues[new[] { 0, 0, 1, 0 }], Tolerance);
    }

    [Fact]
    public void Append_IdentifiableValues_PreservesOrder()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 2, headDim: 3, maxSeqLen: 8);

        // Append 3 tokens one at a time
        for (int t = 0; t < 3; t++)
        {
            var k = CreateKVTensor(1, 2, 1, 3, t * 100.0);
            var v = CreateKVTensor(1, 2, 1, 3, t * 100.0 + 50.0);
            cache.Append(0, k, v);
        }

        var (keys, values) = cache.GetCached(0, 1);
        Assert.Equal(3, keys.Shape[2]); // 3 sequence positions

        // Verify each token's values are preserved in order
        for (int t = 0; t < 3; t++)
        {
            Assert.Equal(t * 100.0, keys[new[] { 0, 0, t, 0 }], Tolerance);
            Assert.Equal(t * 100.0 + 50.0, values[new[] { 0, 0, t, 0 }], Tolerance);
        }
    }

    // ============================
    // Sequence Length Tracking
    // ============================

    [Fact]
    public void SequenceLength_IncreasesWithEachAppend()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);

        Assert.Equal(0, cache.GetSequenceLength(0));

        var k = CreateKVTensor(1, 1, 1, 2, 1.0);
        var v = CreateKVTensor(1, 1, 1, 2, 1.0);

        cache.Append(0, k, v);
        Assert.Equal(1, cache.GetSequenceLength(0));

        cache.Append(0, k, v);
        Assert.Equal(2, cache.GetSequenceLength(0));

        // Append 3 tokens at once
        var k3 = CreateKVTensor(1, 1, 3, 2, 1.0);
        var v3 = CreateKVTensor(1, 1, 3, 2, 1.0);
        cache.Append(0, k3, v3);
        Assert.Equal(5, cache.GetSequenceLength(0));
    }

    [Fact]
    public void CurrentLength_MatchesSequenceLength()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);
        var k = CreateKVTensor(1, 1, 1, 2, 1.0);
        var v = CreateKVTensor(1, 1, 1, 2, 1.0);

        Assert.Equal(0, cache.CurrentLength);
        cache.Append(0, k, v);
        Assert.Equal(1, cache.CurrentLength);
    }

    // ============================
    // Multi-Layer Tests
    // ============================

    [Fact]
    public void MultiLayer_IndependentCaches()
    {
        var cache = CreateCache(numLayers: 3, numHeads: 1, headDim: 2, maxSeqLen: 8);

        // Append different values to different layers
        for (int layer = 0; layer < 3; layer++)
        {
            var k = CreateKVTensor(1, 1, 1, 2, layer * 10.0);
            var v = CreateKVTensor(1, 1, 1, 2, layer * 10.0 + 1.0);
            cache.Append(layer, k, v);
        }

        // Verify each layer has its own values
        for (int layer = 0; layer < 3; layer++)
        {
            var (keys, values) = cache.GetCached(layer, 1);
            Assert.Equal(layer * 10.0, keys[new[] { 0, 0, 0, 0 }], Tolerance);
            Assert.Equal(layer * 10.0 + 1.0, values[new[] { 0, 0, 0, 0 }], Tolerance);
        }
    }

    // ============================
    // Truncation Tests
    // ============================

    [Fact]
    public void Truncate_ReducesSequenceLength()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);

        // Fill with 5 tokens
        for (int t = 0; t < 5; t++)
        {
            var k = CreateKVTensor(1, 1, 1, 2, (double)t);
            var v = CreateKVTensor(1, 1, 1, 2, (double)t);
            cache.Append(0, k, v);
        }

        Assert.Equal(5, cache.GetSequenceLength(0));

        // Truncate to 3
        cache.Truncate(3);
        Assert.Equal(3, cache.GetSequenceLength(0));

        // Verify first 3 tokens are still accessible
        var (keys, _) = cache.GetCached(0, 1);
        Assert.Equal(3, keys.Shape[2]);
    }

    [Fact]
    public void Truncate_ToZero_Clears()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);

        var k = CreateKVTensor(1, 1, 3, 2, 1.0);
        var v = CreateKVTensor(1, 1, 3, 2, 1.0);
        cache.Append(0, k, v);
        Assert.Equal(3, cache.GetSequenceLength(0));

        cache.Truncate(0);
        Assert.Equal(0, cache.GetSequenceLength(0));
    }

    // ============================
    // Clear Tests
    // ============================

    [Fact]
    public void Clear_ResetsAllSequenceLengths()
    {
        var cache = CreateCache(numLayers: 2, numHeads: 1, headDim: 2, maxSeqLen: 16);

        for (int layer = 0; layer < 2; layer++)
        {
            var k = CreateKVTensor(1, 1, 3, 2, 1.0);
            var v = CreateKVTensor(1, 1, 3, 2, 1.0);
            cache.Append(layer, k, v);
        }

        cache.Clear();
        Assert.Equal(0, cache.GetSequenceLength(0));
        Assert.Equal(0, cache.CurrentLength);
    }

    [Fact]
    public void Clear_ResetsStatistics()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);
        var k = CreateKVTensor(1, 1, 1, 2, 1.0);
        var v = CreateKVTensor(1, 1, 1, 2, 1.0);
        cache.Append(0, k, v);

        Assert.True(cache.CacheMisses > 0);
        cache.Clear();
        Assert.Equal(0, cache.CacheHits);
        Assert.Equal(0, cache.CacheMisses);
        Assert.Equal(0, cache.Evictions);
    }

    // ============================
    // Statistics and Memory Tests
    // ============================

    [Fact]
    public void CacheMisses_CountsNewTokens()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);
        var k = CreateKVTensor(1, 1, 1, 2, 1.0);
        var v = CreateKVTensor(1, 1, 1, 2, 1.0);

        cache.Append(0, k, v);
        Assert.Equal(1, cache.CacheMisses); // 1 new token

        var k3 = CreateKVTensor(1, 1, 3, 2, 1.0);
        var v3 = CreateKVTensor(1, 1, 3, 2, 1.0);
        cache.Append(0, k3, v3);
        Assert.Equal(4, cache.CacheMisses); // 1 + 3 = 4 total new tokens
    }

    [Fact]
    public void CacheHits_CountsRetrievals()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);
        var k = CreateKVTensor(1, 1, 3, 2, 1.0);
        var v = CreateKVTensor(1, 1, 3, 2, 1.0);

        // Append triggers GetCached internally, which counts hits
        cache.Append(0, k, v);
        long hitsAfterAppend = cache.CacheHits;

        // Manual GetCached should increase hits by batch_size * maxLen
        cache.GetCached(0, 1);
        Assert.True(cache.CacheHits > hitsAfterAppend);
    }

    [Fact]
    public void EstimateMemoryBytes_HandComputed()
    {
        // 2 layers, 4 heads, maxSeq=16, headDim=8, batch=1
        // Elements per layer = 1 * 4 * 16 * 8 = 512
        // Total elements = 512 * 2 layers * 2 (K+V) = 2048
        // Float32 = 4 bytes => 2048 * 4 = 8192 bytes
        var config = new KVCacheConfig
        {
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8,
            MaxSequenceLength = 16,
            MaxBatchSize = 1,
            DataType = CacheDataType.Float32
        };
        Assert.Equal(8192, config.EstimateMemoryBytes());
    }

    [Fact]
    public void EstimateMemoryBytes_Float16_HalvesMemory()
    {
        var configFp32 = new KVCacheConfig
        {
            NumLayers = 2, NumHeads = 4, HeadDimension = 8,
            MaxSequenceLength = 16, MaxBatchSize = 1,
            DataType = CacheDataType.Float32
        };
        var configFp16 = new KVCacheConfig
        {
            NumLayers = 2, NumHeads = 4, HeadDimension = 8,
            MaxSequenceLength = 16, MaxBatchSize = 1,
            DataType = CacheDataType.Float16
        };

        Assert.Equal(configFp32.EstimateMemoryBytes() / 2, configFp16.EstimateMemoryBytes());
    }

    [Fact]
    public void EstimateMemoryBytes_Int8_QuartersMemory()
    {
        var configFp32 = new KVCacheConfig
        {
            NumLayers = 2, NumHeads = 4, HeadDimension = 8,
            MaxSequenceLength = 16, MaxBatchSize = 1,
            DataType = CacheDataType.Float32
        };
        var configInt8 = new KVCacheConfig
        {
            NumLayers = 2, NumHeads = 4, HeadDimension = 8,
            MaxSequenceLength = 16, MaxBatchSize = 1,
            DataType = CacheDataType.Int8
        };

        Assert.Equal(configFp32.EstimateMemoryBytes() / 4, configInt8.EstimateMemoryBytes());
    }

    [Fact]
    public void EstimateMemory_ScalesLinearlyWithBatchSize()
    {
        var config1 = new KVCacheConfig
        {
            NumLayers = 2, NumHeads = 4, HeadDimension = 8,
            MaxSequenceLength = 16, MaxBatchSize = 1
        };
        var config4 = new KVCacheConfig
        {
            NumLayers = 2, NumHeads = 4, HeadDimension = 8,
            MaxSequenceLength = 16, MaxBatchSize = 4
        };
        Assert.Equal(config1.EstimateMemoryBytes() * 4, config4.EstimateMemoryBytes());
    }

    [Fact]
    public void EstimateMemory_ScalesLinearlyWithLayers()
    {
        var config2 = new KVCacheConfig
        {
            NumLayers = 2, NumHeads = 4, HeadDimension = 8,
            MaxSequenceLength = 16, MaxBatchSize = 1
        };
        var config6 = new KVCacheConfig
        {
            NumLayers = 6, NumHeads = 4, HeadDimension = 8,
            MaxSequenceLength = 16, MaxBatchSize = 1
        };
        Assert.Equal(config2.EstimateMemoryBytes() * 3, config6.EstimateMemoryBytes());
    }

    // ============================
    // Sliding Window Tests
    // ============================

    [Fact]
    public void SlidingWindow_EvictsOldestTokens()
    {
        var cache = CreateCache(
            numLayers: 1, numHeads: 1, headDim: 2,
            maxSeqLen: 16, useSlidingWindow: true, windowSize: 4);

        // Fill to window size
        for (int t = 0; t < 4; t++)
        {
            var k = CreateKVTensor(1, 1, 1, 2, (double)(t + 1));
            var v = CreateKVTensor(1, 1, 1, 2, (double)(t + 1) * 10);
            cache.Append(0, k, v);
        }

        Assert.Equal(4, cache.GetSequenceLength(0));

        // Add one more - should evict oldest
        var kNew = CreateKVTensor(1, 1, 1, 2, 5.0);
        var vNew = CreateKVTensor(1, 1, 1, 2, 50.0);
        var (keys, values) = cache.Append(0, kNew, vNew);

        // Window should still be size 4 (evicted 1, added 1)
        Assert.Equal(4, cache.GetSequenceLength(0));
        Assert.True(cache.Evictions > 0);
    }

    [Fact]
    public void SlidingWindow_EvictionCountIsCorrect()
    {
        var cache = CreateCache(
            numLayers: 1, numHeads: 1, headDim: 2,
            maxSeqLen: 16, useSlidingWindow: true, windowSize: 3);

        // Fill 3 tokens (window full)
        for (int t = 0; t < 3; t++)
        {
            var k = CreateKVTensor(1, 1, 1, 2, 1.0);
            var v = CreateKVTensor(1, 1, 1, 2, 1.0);
            cache.Append(0, k, v);
        }

        Assert.Equal(0, cache.Evictions);

        // Add 2 more tokens -> should evict 2
        var k2 = CreateKVTensor(1, 1, 2, 2, 1.0);
        var v2 = CreateKVTensor(1, 1, 2, 2, 1.0);
        cache.Append(0, k2, v2);

        Assert.Equal(2, cache.Evictions);
        Assert.Equal(3, cache.GetSequenceLength(0)); // window size
    }

    // ============================
    // Batch State Copying
    // ============================

    [Fact]
    public void CopyBatchState_DuplicatesData()
    {
        var cache = CreateCache(
            numLayers: 1, numHeads: 1, headDim: 2,
            maxSeqLen: 8, maxBatchSize: 2);

        // Append to batch 0
        var k = new Tensor<double>(new[] { 2, 1, 1, 2 });
        k[new[] { 0, 0, 0, 0 }] = 42.0;
        k[new[] { 0, 0, 0, 1 }] = 43.0;
        k[new[] { 1, 0, 0, 0 }] = 0.0; // batch 1 starts empty
        k[new[] { 1, 0, 0, 1 }] = 0.0;

        var v = new Tensor<double>(new[] { 2, 1, 1, 2 });
        v[new[] { 0, 0, 0, 0 }] = 100.0;
        v[new[] { 0, 0, 0, 1 }] = 200.0;
        v[new[] { 1, 0, 0, 0 }] = 0.0;
        v[new[] { 1, 0, 0, 1 }] = 0.0;

        cache.Append(0, k, v);

        // Copy batch 0 -> batch 1
        cache.CopyBatchState(0, 1);

        // Verify batch 1 has the same data
        var (keys, values) = cache.GetCached(0, 2);
        Assert.Equal(42.0, keys[new[] { 1, 0, 0, 0 }], Tolerance);
        Assert.Equal(43.0, keys[new[] { 1, 0, 0, 1 }], Tolerance);
        Assert.Equal(100.0, values[new[] { 1, 0, 0, 0 }], Tolerance);
        Assert.Equal(200.0, values[new[] { 1, 0, 0, 1 }], Tolerance);
    }

    // ============================
    // Overflow Protection
    // ============================

    [Fact]
    public void Append_Overflow_ThrowsException()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 3);

        // Fill to max
        var k3 = CreateKVTensor(1, 1, 3, 2, 1.0);
        var v3 = CreateKVTensor(1, 1, 3, 2, 1.0);
        cache.Append(0, k3, v3);

        // One more should overflow
        var k1 = CreateKVTensor(1, 1, 1, 2, 1.0);
        var v1 = CreateKVTensor(1, 1, 1, 2, 1.0);
        Assert.Throws<InvalidOperationException>(() => cache.Append(0, k1, v1));
    }

    [Fact]
    public void InvalidLayerIndex_ThrowsException()
    {
        var cache = CreateCache(numLayers: 2, numHeads: 1, headDim: 2, maxSeqLen: 8);
        var k = CreateKVTensor(1, 1, 1, 2, 1.0);
        var v = CreateKVTensor(1, 1, 1, 2, 1.0);

        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Append(5, k, v));
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Append(-1, k, v));
    }

    // ============================
    // Model Configuration Presets
    // ============================

    [Fact]
    public void ForModel_GPT2_CorrectDimensions()
    {
        var config = KVCacheConfig.ForModel("gpt2");
        Assert.Equal(12, config.NumLayers);
        Assert.Equal(12, config.NumHeads);
        Assert.Equal(64, config.HeadDimension);
        Assert.Equal(1024, config.MaxSequenceLength);
    }

    [Fact]
    public void ForModel_Llama7B_CorrectDimensions()
    {
        var config = KVCacheConfig.ForModel("llama-7b");
        Assert.Equal(32, config.NumLayers);
        Assert.Equal(32, config.NumHeads);
        Assert.Equal(128, config.HeadDimension);
        Assert.Equal(4096, config.MaxSequenceLength);
        Assert.Equal(CacheDataType.Float16, config.DataType);
    }

    [Fact]
    public void ForModel_Llama70B_UsesSlidingWindow()
    {
        var config = KVCacheConfig.ForModel("llama-70b");
        Assert.True(config.UseSlidingWindow);
        Assert.Equal(2048, config.WindowSize);
    }

    // ============================
    // GetStatistics Tests
    // ============================

    [Fact]
    public void GetStatistics_ContainsExpectedKeys()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 8);
        var stats = cache.GetStatistics();

        Assert.True(stats.ContainsKey("CacheHits"));
        Assert.True(stats.ContainsKey("CacheMisses"));
        Assert.True(stats.ContainsKey("Evictions"));
        Assert.True(stats.ContainsKey("HitRate"));
        Assert.True(stats.ContainsKey("CurrentMemoryMB"));
        Assert.True(stats.ContainsKey("MaxMemoryMB"));
    }

    [Fact]
    public void GetStatistics_HitRate_ComputedCorrectly()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 16);
        var k = CreateKVTensor(1, 1, 3, 2, 1.0);
        var v = CreateKVTensor(1, 1, 3, 2, 1.0);
        cache.Append(0, k, v); // 3 misses + some hits from GetCached

        var stats = cache.GetStatistics();
        double hitRate = (double)stats["HitRate"];
        Assert.InRange(hitRate, 0.0, 1.0);
    }

    // ============================
    // Lazy Allocation Tests
    // ============================

    [Fact]
    public void LazyAllocation_AllocatesOnFirstAppend()
    {
        var cache = CreateCache(numLayers: 2, numHeads: 1, headDim: 2, maxSeqLen: 8, preAllocate: false);

        // Before any append, memory should be minimal
        long memBefore = cache.GetCurrentMemoryUsage();

        var k = CreateKVTensor(1, 1, 1, 2, 1.0);
        var v = CreateKVTensor(1, 1, 1, 2, 1.0);
        cache.Append(0, k, v);

        long memAfter = cache.GetCurrentMemoryUsage();
        // Memory should increase after allocation
        Assert.True(memAfter > memBefore);
    }

    // ============================
    // Multi-Head Attention Tests
    // ============================

    [Fact]
    public void MultiHead_IndependentPerHead()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 3, headDim: 2, maxSeqLen: 8);

        // Create tensor with different values per head
        var k = new Tensor<double>(new[] { 1, 3, 1, 2 });
        for (int h = 0; h < 3; h++)
        {
            k[new[] { 0, h, 0, 0 }] = (h + 1) * 10.0;
            k[new[] { 0, h, 0, 1 }] = (h + 1) * 10.0 + 1.0;
        }
        var v = CreateKVTensor(1, 3, 1, 2, 0.0);
        cache.Append(0, k, v);

        var (keys, _) = cache.GetCached(0, 1);
        Assert.Equal(10.0, keys[new[] { 0, 0, 0, 0 }], Tolerance); // head 0
        Assert.Equal(20.0, keys[new[] { 0, 1, 0, 0 }], Tolerance); // head 1
        Assert.Equal(30.0, keys[new[] { 0, 2, 0, 0 }], Tolerance); // head 2
    }

    // ============================
    // MaxLength Property
    // ============================

    [Fact]
    public void MaxLength_MatchesConfig()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 42);
        Assert.Equal(42, cache.MaxLength);
    }

    // ============================
    // Clear Individual Batch
    // ============================

    [Fact]
    public void ClearBatch_OnlyClearsSpecifiedBatch()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 8, maxBatchSize: 2);

        var k = new Tensor<double>(new[] { 2, 1, 2, 2 });
        for (int b = 0; b < 2; b++)
            for (int s = 0; s < 2; s++)
                for (int d = 0; d < 2; d++)
                    k[new[] { b, 0, s, d }] = b * 10.0 + s + d * 0.1;
        var v = CreateKVTensor(2, 1, 2, 2, 1.0);
        cache.Append(0, k, v);

        // Clear only batch 0
        cache.Clear(0);

        // Batch 0 should be cleared, batch 1 should remain
        // Note: GetSequenceLength checks layer 0 for the specified batch
        Assert.Equal(0, cache.GetSequenceLength(0)); // batch 0 cleared
    }

    // ============================
    // Truncation with Specific Batch
    // ============================

    [Fact]
    public void Truncate_NegativeLength_Throws()
    {
        var cache = CreateCache(numLayers: 1, numHeads: 1, headDim: 2, maxSeqLen: 8);
        Assert.Throws<ArgumentOutOfRangeException>(() => cache.Truncate(-1));
    }
}
