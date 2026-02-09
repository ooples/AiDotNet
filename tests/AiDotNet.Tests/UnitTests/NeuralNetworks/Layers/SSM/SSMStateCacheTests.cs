using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Unit tests for <see cref="SSMStateCache{T}"/>.
/// </summary>
public class SSMStateCacheTests
{
    [Fact]
    public void Constructor_DefaultParameters_CreatesCache()
    {
        var cache = new SSMStateCache<float>();

        Assert.Equal(0, cache.CachedLayerCount);
        Assert.False(cache.CompressionEnabled);
    }

    [Fact]
    public void Constructor_WithCompression_EnablesCompression()
    {
        var cache = new SSMStateCache<float>(enableCompression: true, compressionBitWidth: 8);

        Assert.True(cache.CompressionEnabled);
    }

    [Fact]
    public void Constructor_ThrowsOnInvalidBitWidth()
    {
        Assert.Throws<ArgumentException>(() => new SSMStateCache<float>(true, 0));
        Assert.Throws<ArgumentException>(() => new SSMStateCache<float>(true, 33));
    }

    [Fact]
    public void CacheSSMState_StoresAndRetrieves()
    {
        var cache = new SSMStateCache<float>();
        var state = CreateRandomTensor(new[] { 1, 16, 4 });

        cache.CacheSSMState(0, state);

        Assert.True(cache.HasSSMState(0));
        Assert.Equal(1, cache.CachedLayerCount);

        var retrieved = cache.GetSSMState(0);
        Assert.NotNull(retrieved);
        Assert.Equal(state.Shape, retrieved.Shape);

        for (int i = 0; i < state.Length; i++)
        {
            Assert.Equal(state[i], retrieved[i]);
        }
    }

    [Fact]
    public void CacheSSMState_OverwritesPreviousState()
    {
        var cache = new SSMStateCache<float>();
        var state1 = new Tensor<float>(new[] { 1, 4, 2 });
        state1.Fill(1.0f);
        var state2 = new Tensor<float>(new[] { 1, 4, 2 });
        state2.Fill(2.0f);

        cache.CacheSSMState(0, state1);
        cache.CacheSSMState(0, state2);

        var retrieved = cache.GetSSMState(0);
        Assert.NotNull(retrieved);
        Assert.Equal(2.0f, retrieved[0]);
    }

    [Fact]
    public void CacheSSMState_ThrowsOnNull()
    {
        var cache = new SSMStateCache<float>();
        Assert.Throws<ArgumentNullException>(() => cache.CacheSSMState(0, null!));
    }

    [Fact]
    public void GetSSMState_ReturnsNullForUncachedLayer()
    {
        var cache = new SSMStateCache<float>();

        Assert.Null(cache.GetSSMState(0));
        Assert.False(cache.HasSSMState(0));
    }

    [Fact]
    public void GetSSMState_ReturnsIndependentCopy()
    {
        var cache = new SSMStateCache<float>();
        var state = new Tensor<float>(new[] { 1, 4, 2 });
        state.Fill(1.0f);
        cache.CacheSSMState(0, state);

        var retrieved1 = cache.GetSSMState(0);
        retrieved1![0] = 999.0f;

        var retrieved2 = cache.GetSSMState(0);
        Assert.Equal(1.0f, retrieved2![0]);
    }

    [Fact]
    public void CacheConvBuffer_StoresAndRetrieves()
    {
        var cache = new SSMStateCache<float>();
        var buffer = CreateRandomTensor(new[] { 1, 16, 3 });

        cache.CacheConvBuffer(0, buffer);

        Assert.True(cache.HasConvBuffer(0));

        var retrieved = cache.GetConvBuffer(0);
        Assert.NotNull(retrieved);
        Assert.Equal(buffer.Shape, retrieved.Shape);
    }

    [Fact]
    public void CacheConvBuffer_ThrowsOnNull()
    {
        var cache = new SSMStateCache<float>();
        Assert.Throws<ArgumentNullException>(() => cache.CacheConvBuffer(0, null!));
    }

    [Fact]
    public void GetConvBuffer_ReturnsNullForUncachedLayer()
    {
        var cache = new SSMStateCache<float>();
        Assert.Null(cache.GetConvBuffer(0));
    }

    [Fact]
    public void Reset_ClearsAllStates()
    {
        var cache = new SSMStateCache<float>();
        cache.CacheSSMState(0, CreateRandomTensor(new[] { 1, 8, 4 }));
        cache.CacheSSMState(1, CreateRandomTensor(new[] { 1, 8, 4 }));
        cache.CacheConvBuffer(0, CreateRandomTensor(new[] { 1, 8, 3 }));

        cache.Reset();

        Assert.Equal(0, cache.CachedLayerCount);
        Assert.Null(cache.GetSSMState(0));
        Assert.Null(cache.GetSSMState(1));
        Assert.Null(cache.GetConvBuffer(0));
    }

    [Fact]
    public void Clone_ProducesIndependentCopy()
    {
        var cache = new SSMStateCache<float>();
        var state = new Tensor<float>(new[] { 1, 4, 2 });
        state.Fill(5.0f);
        cache.CacheSSMState(0, state);
        cache.CacheConvBuffer(0, CreateRandomTensor(new[] { 1, 4, 3 }));

        var clone = cache.Clone();

        // Clone should have same data
        Assert.Equal(cache.CachedLayerCount, clone.CachedLayerCount);
        Assert.True(clone.HasSSMState(0));
        Assert.True(clone.HasConvBuffer(0));

        var clonedState = clone.GetSSMState(0);
        Assert.NotNull(clonedState);
        Assert.Equal(5.0f, clonedState[0]);

        // Modifying clone should not affect original
        clone.Reset();
        Assert.Equal(1, cache.CachedLayerCount);
        Assert.True(cache.HasSSMState(0));
    }

    [Fact]
    public void Clone_DeepCopiesStates()
    {
        var cache = new SSMStateCache<float>();
        var state = new Tensor<float>(new[] { 1, 4, 2 });
        state.Fill(3.0f);
        cache.CacheSSMState(0, state);

        var clone = cache.Clone();

        // Overwrite state in clone
        var newState = new Tensor<float>(new[] { 1, 4, 2 });
        newState.Fill(99.0f);
        clone.CacheSSMState(0, newState);

        // Original should be unchanged
        var originalState = cache.GetSSMState(0);
        Assert.Equal(3.0f, originalState![0]);
    }

    [Fact]
    public void MultipleLayerStates_StoreAndRetrieveIndependently()
    {
        var cache = new SSMStateCache<float>();

        for (int i = 0; i < 5; i++)
        {
            var state = new Tensor<float>(new[] { 1, 8, 4 });
            state.Fill((float)(i + 1));
            cache.CacheSSMState(i, state);
        }

        Assert.Equal(5, cache.CachedLayerCount);

        for (int i = 0; i < 5; i++)
        {
            var retrieved = cache.GetSSMState(i);
            Assert.NotNull(retrieved);
            Assert.Equal((float)(i + 1), retrieved[0], 5);
        }
    }

    [Fact]
    public void StateCompression_PreservesApproximateValues()
    {
        var cache = new SSMStateCache<float>(enableCompression: true, compressionBitWidth: 8);
        var state = CreateRandomTensor(new[] { 1, 8, 4 });

        cache.CacheSSMState(0, state);
        var retrieved = cache.GetSSMState(0);

        Assert.NotNull(retrieved);
        Assert.Equal(state.Shape, retrieved.Shape);

        // Values should be approximately equal (within quantization error)
        for (int i = 0; i < state.Length; i++)
        {
            Assert.True(MathF.Abs(state[i] - retrieved[i]) < 0.02f,
                $"Compressed value at {i}: original={state[i]:G6}, retrieved={retrieved[i]:G6}");
        }
    }

    [Fact]
    public void GetMemoryUsageBytes_ReturnsNonNegative()
    {
        var cache = new SSMStateCache<float>();
        Assert.Equal(0L, cache.GetMemoryUsageBytes());

        cache.CacheSSMState(0, CreateRandomTensor(new[] { 1, 8, 4 }));
        Assert.True(cache.GetMemoryUsageBytes() > 0);
    }

    [Fact]
    public void DoubleType_StoresAndRetrieves()
    {
        var cache = new SSMStateCache<double>();
        var state = new Tensor<double>(new[] { 1, 4, 2 });
        var random = new Random(42);
        for (int i = 0; i < state.Length; i++)
        {
            state[i] = random.NextDouble() * 2 - 1;
        }

        cache.CacheSSMState(0, state);
        var retrieved = cache.GetSSMState(0);

        Assert.NotNull(retrieved);
        for (int i = 0; i < state.Length; i++)
        {
            Assert.Equal(state[i], retrieved[i]);
        }
    }

    #region Helpers

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }

    #endregion
}
