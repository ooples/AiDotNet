using AiDotNet.Inference;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

public class KVCacheTests
{
    [Fact]
    public void KVCache_AppendAcrossLayers_MaintainsIndependentLengths()
    {
        var config = new KVCacheConfig
        {
            NumLayers = 2,
            NumHeads = 1,
            HeadDimension = 2,
            MaxSequenceLength = 8,
            MaxBatchSize = 1,
            PreAllocate = true
        };

        var cache = new KVCache<float>(config);

        var keys0 = new Tensor<float>(new[] { 1, 1, 2, 2 });
        var values0 = new Tensor<float>(new[] { 1, 1, 2, 2 });
        keys0[new[] { 0, 0, 0, 0 }] = 1f;
        keys0[new[] { 0, 0, 0, 1 }] = 2f;
        keys0[new[] { 0, 0, 1, 0 }] = 3f;
        keys0[new[] { 0, 0, 1, 1 }] = 4f;
        values0[new[] { 0, 0, 0, 0 }] = 5f;
        values0[new[] { 0, 0, 0, 1 }] = 6f;
        values0[new[] { 0, 0, 1, 0 }] = 7f;
        values0[new[] { 0, 0, 1, 1 }] = 8f;

        var (layer0Keys, _) = cache.Append(0, keys0, values0);
        Assert.Equal(2, layer0Keys.Shape[2]);

        var keys1 = new Tensor<float>(new[] { 1, 1, 2, 2 });
        var values1 = new Tensor<float>(new[] { 1, 1, 2, 2 });
        keys1[new[] { 0, 0, 0, 0 }] = 10f;
        keys1[new[] { 0, 0, 0, 1 }] = 11f;
        keys1[new[] { 0, 0, 1, 0 }] = 12f;
        keys1[new[] { 0, 0, 1, 1 }] = 13f;
        values1[new[] { 0, 0, 0, 0 }] = 14f;
        values1[new[] { 0, 0, 0, 1 }] = 15f;
        values1[new[] { 0, 0, 1, 0 }] = 16f;
        values1[new[] { 0, 0, 1, 1 }] = 17f;

        var (layer1Keys, _) = cache.Append(1, keys1, values1);
        Assert.Equal(2, layer1Keys.Shape[2]);
        Assert.Equal(10f, layer1Keys[new[] { 0, 0, 0, 0 }]);
        Assert.Equal(13f, layer1Keys[new[] { 0, 0, 1, 1 }]);

        var (layer0KeysAfter, _) = cache.GetCached(0, batchSize: 1);
        Assert.Equal(2, layer0KeysAfter.Shape[2]);
        Assert.Equal(1f, layer0KeysAfter[new[] { 0, 0, 0, 0 }]);
        Assert.Equal(4f, layer0KeysAfter[new[] { 0, 0, 1, 1 }]);
    }

    [Fact]
    public void KVCache_Float16Storage_RoundTripsValues()
    {
        var config = new KVCacheConfig
        {
            NumLayers = 1,
            NumHeads = 1,
            HeadDimension = 2,
            MaxSequenceLength = 8,
            MaxBatchSize = 1,
            PreAllocate = true,
            DataType = CacheDataType.Float16
        };

        var cache = new KVCache<float>(config);

        var keys = new Tensor<float>(new[] { 1, 1, 2, 2 });
        var values = new Tensor<float>(new[] { 1, 1, 2, 2 });
        keys[new[] { 0, 0, 0, 0 }] = 1f;
        keys[new[] { 0, 0, 0, 1 }] = 2f;
        keys[new[] { 0, 0, 1, 0 }] = 3f;
        keys[new[] { 0, 0, 1, 1 }] = 4f;
        values[new[] { 0, 0, 0, 0 }] = 5f;
        values[new[] { 0, 0, 0, 1 }] = 6f;
        values[new[] { 0, 0, 1, 0 }] = 7f;
        values[new[] { 0, 0, 1, 1 }] = 8f;

        var (cachedKeys, cachedValues) = cache.Append(0, keys, values);
        Assert.Equal(2, cachedKeys.Shape[2]);
        Assert.Equal(1f, cachedKeys[new[] { 0, 0, 0, 0 }]);
        Assert.Equal(4f, cachedKeys[new[] { 0, 0, 1, 1 }]);
        Assert.Equal(5f, cachedValues[new[] { 0, 0, 0, 0 }]);
        Assert.Equal(8f, cachedValues[new[] { 0, 0, 1, 1 }]);
    }

    [Fact]
    public void KVCache_Int8Storage_RoundTripsApproximately()
    {
        var config = new KVCacheConfig
        {
            NumLayers = 1,
            NumHeads = 1,
            HeadDimension = 2,
            MaxSequenceLength = 8,
            MaxBatchSize = 1,
            PreAllocate = true,
            DataType = CacheDataType.Int8
        };

        var cache = new KVCache<float>(config);

        var keys = new Tensor<float>(new[] { 1, 1, 2, 2 });
        var values = new Tensor<float>(new[] { 1, 1, 2, 2 });
        keys[new[] { 0, 0, 0, 0 }] = 1f;
        keys[new[] { 0, 0, 0, 1 }] = 2f;
        keys[new[] { 0, 0, 1, 0 }] = 3f;
        keys[new[] { 0, 0, 1, 1 }] = 4f;
        values[new[] { 0, 0, 0, 0 }] = 5f;
        values[new[] { 0, 0, 0, 1 }] = 6f;
        values[new[] { 0, 0, 1, 0 }] = 7f;
        values[new[] { 0, 0, 1, 1 }] = 8f;

        var (cachedKeys, cachedValues) = cache.Append(0, keys, values);
        Assert.Equal(2, cachedKeys.Shape[2]);

        // Int8 quantization is approximate; tolerate small error.
        Assert.InRange(Math.Abs(cachedKeys[new[] { 0, 0, 0, 0 }] - 1f), 0f, 0.1f);
        Assert.InRange(Math.Abs(cachedKeys[new[] { 0, 0, 1, 1 }] - 4f), 0f, 0.1f);
        Assert.InRange(Math.Abs(cachedValues[new[] { 0, 0, 0, 0 }] - 5f), 0f, 0.1f);
        Assert.InRange(Math.Abs(cachedValues[new[] { 0, 0, 1, 1 }] - 8f), 0f, 0.1f);
    }
}
