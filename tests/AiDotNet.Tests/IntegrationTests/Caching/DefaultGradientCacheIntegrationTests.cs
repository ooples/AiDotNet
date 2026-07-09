using AiDotNet.Caching;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Caching;

/// <summary>
/// Integration tests for DefaultGradientCache to verify caching behavior,
/// thread safety, and edge case handling.
/// </summary>
public class DefaultGradientCacheIntegrationTests
{
    #region Basic CRUD Operations

    [Fact(Timeout = 120000)]
    public async Task GetCachedGradient_NonExistentKey_ReturnsNull()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();

        // Act
        var result = cache.GetCachedGradient("nonexistent_key");

        // Assert
        Assert.Null(result);
    }

    [Fact(Timeout = 120000)]
    public async Task CacheGradient_ThenGet_ReturnsSameGradient()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var gradient = new TestGradientModel<double>(42.0);
        const string key = "test_gradient";

        // Act
        cache.CacheGradient(key, gradient);
        var retrieved = cache.GetCachedGradient(key);

        // Assert
        Assert.NotNull(retrieved);
        Assert.Same(gradient, retrieved);
    }

    [Fact(Timeout = 120000)]
    public async Task CacheGradient_OverwriteExistingKey_ReturnsNewGradient()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var gradient1 = new TestGradientModel<double>(1.0);
        var gradient2 = new TestGradientModel<double>(2.0);
        const string key = "shared_key";

        // Act
        cache.CacheGradient(key, gradient1);
        cache.CacheGradient(key, gradient2);
        var retrieved = cache.GetCachedGradient(key);

        // Assert
        Assert.Same(gradient2, retrieved);
        Assert.NotSame(gradient1, retrieved);
    }

    [Fact(Timeout = 120000)]
    public async Task ClearCache_RemovesAllEntries()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        cache.CacheGradient("key1", new TestGradientModel<double>(1.0));
        cache.CacheGradient("key2", new TestGradientModel<double>(2.0));
        cache.CacheGradient("key3", new TestGradientModel<double>(3.0));

        // Act
        cache.ClearCache();

        // Assert
        Assert.Null(cache.GetCachedGradient("key1"));
        Assert.Null(cache.GetCachedGradient("key2"));
        Assert.Null(cache.GetCachedGradient("key3"));
    }

    #endregion

    #region Multiple Keys Tests

    [Fact(Timeout = 120000)]
    public async Task Cache_MultipleKeys_EachRetrievableIndependently()
    {
        await Task.Yield();
        // Arrange - capacity sized above the working set so every key stays resident.
        var cache = new DefaultGradientCache<double>(capacity: 200);
        var gradients = new Dictionary<string, TestGradientModel<double>>();

        for (int i = 0; i < 100; i++)
        {
            var key = $"gradient_{i}";
            var gradient = new TestGradientModel<double>(i * 1.5);
            gradients[key] = gradient;
            cache.CacheGradient(key, gradient);
        }

        // Act & Assert
        foreach (var kvp in gradients)
        {
            var retrieved = cache.GetCachedGradient(kvp.Key);
            Assert.NotNull(retrieved);
            Assert.Same(kvp.Value, retrieved);
        }
    }

    #endregion

    #region Bounded-Capacity / Eviction Tests

    // Regression guard for the managed-heap leak in the batched Optimize training
    // loop: GradientBasedOptimizerBase.GenerateGradientCacheKey folds a per-step
    // parameter-state fingerprint plus per-batch tensor identities into every key,
    // so each training step produces a brand-new key that is never looked up again.
    // With an unbounded cache the dictionary grew by one full parameter-sized
    // gradient every step (only ever cleared by Reset(), which the training loop
    // never calls), so per-epoch GC cost — and wall-time — climbed roughly linearly
    // with the epoch index. The cache must therefore retain at most Capacity entries.

    [Fact(Timeout = 120000)]
    public async Task CacheGradient_ExceedingCapacity_EvictsOldestFirst()
    {
        await Task.Yield();
        // Arrange
        const int capacity = 8;
        var cache = new DefaultGradientCache<double>(capacity);

        // Act - insert far more distinct keys than capacity (mirrors a training run
        // where every step is a unique key).
        const int inserted = 500;
        for (int i = 0; i < inserted; i++)
        {
            cache.CacheGradient($"step_{i}", new TestGradientModel<double>(i));
        }

        // Assert - only the most recent `capacity` keys survive; everything older is evicted.
        for (int i = 0; i < inserted - capacity; i++)
        {
            Assert.Null(cache.GetCachedGradient($"step_{i}"));
        }
        for (int i = inserted - capacity; i < inserted; i++)
        {
            Assert.NotNull(cache.GetCachedGradient($"step_{i}"));
        }
    }

    [Fact(Timeout = 120000)]
    public async Task CacheGradient_ManyUniqueKeys_DoesNotGrowUnbounded()
    {
        await Task.Yield();
        // Arrange
        const int capacity = 16;
        var cache = new DefaultGradientCache<double>(capacity);

        // Act - a large number of unique keys, like a long training run.
        const int steps = 100_000;
        for (int i = 0; i < steps; i++)
        {
            cache.CacheGradient($"k{i}", new TestGradientModel<double>(i));
        }

        // Assert - live-entry count is bounded by capacity regardless of how many
        // keys were inserted: everything except the last `capacity` keys is gone.
        int survivors = 0;
        for (int i = steps - capacity - 50; i < steps; i++)
        {
            if (cache.GetCachedGradient($"k{i}") is not null) survivors++;
        }
        Assert.Equal(capacity, survivors);
        // Far-past keys are definitively evicted.
        Assert.Null(cache.GetCachedGradient("k0"));
        Assert.Null(cache.GetCachedGradient($"k{steps / 2}"));
    }

    #endregion

    #region Eviction Policy + Customization

    [Fact(Timeout = 120000)]
    public async Task Constructor_CapacityAndPolicy_AreExposed()
    {
        await Task.Yield();
        var cache = new DefaultGradientCache<double>(4, AiDotNet.Enums.CacheEvictionPolicy.LRU);

        Assert.Equal(4, cache.Capacity);
        Assert.Equal(AiDotNet.Enums.CacheEvictionPolicy.LRU, cache.EvictionPolicy);
        Assert.True(cache.Enabled);
    }

    [Fact(Timeout = 120000)]
    public async Task DefaultEvictionPolicy_IsFifo()
    {
        await Task.Yield();
        Assert.Equal(AiDotNet.Enums.CacheEvictionPolicy.FIFO, new DefaultGradientCache<double>().EvictionPolicy);
    }

    [Fact(Timeout = 120000)]
    public async Task LruPolicy_EvictsLeastRecentlyUsed()
    {
        await Task.Yield();
        var cache = new DefaultGradientCache<double>(3, AiDotNet.Enums.CacheEvictionPolicy.LRU);
        cache.CacheGradient("k0", new TestGradientModel<double>(0));
        cache.CacheGradient("k1", new TestGradientModel<double>(1));
        cache.CacheGradient("k2", new TestGradientModel<double>(2));

        // Touch k0 so it becomes most-recently-used; k1 is now the least-recently-used.
        Assert.NotNull(cache.GetCachedGradient("k0"));

        // Inserting a 4th key evicts the LRU victim (k1), not the oldest-inserted (k0).
        cache.CacheGradient("k3", new TestGradientModel<double>(3));

        Assert.Null(cache.GetCachedGradient("k1"));      // least-recently-used → evicted
        Assert.NotNull(cache.GetCachedGradient("k0"));   // recently touched → retained
        Assert.NotNull(cache.GetCachedGradient("k2"));
        Assert.NotNull(cache.GetCachedGradient("k3"));
    }

    [Fact(Timeout = 120000)]
    public async Task LfuPolicy_EvictsLeastFrequentlyUsed()
    {
        await Task.Yield();
        var cache = new DefaultGradientCache<double>(3, AiDotNet.Enums.CacheEvictionPolicy.LFU);
        cache.CacheGradient("k0", new TestGradientModel<double>(0));
        cache.CacheGradient("k1", new TestGradientModel<double>(1));
        cache.CacheGradient("k2", new TestGradientModel<double>(2));

        // Raise use counts: k0 used most, k1 once, k2 not at all.
        cache.GetCachedGradient("k0");
        cache.GetCachedGradient("k0");
        cache.GetCachedGradient("k1");

        // A 4th key evicts the least-frequently-used (k2), leaving the hot keys.
        cache.CacheGradient("k3", new TestGradientModel<double>(3));

        Assert.Null(cache.GetCachedGradient("k2"));      // never re-used → evicted
        Assert.NotNull(cache.GetCachedGradient("k0"));
        Assert.NotNull(cache.GetCachedGradient("k1"));
        Assert.NotNull(cache.GetCachedGradient("k3"));
    }

    [Fact(Timeout = 120000)]
    public async Task Disabled_Cache_IsPassThrough()
    {
        await Task.Yield();
        var cache = new DefaultGradientCache<double>(8, AiDotNet.Enums.CacheEvictionPolicy.FIFO, enabled: false);

        cache.CacheGradient("k", new TestGradientModel<double>(1));

        Assert.False(cache.Enabled);
        Assert.Null(cache.GetCachedGradient("k")); // stores are dropped → always a miss (recompute)
    }

    [Fact(Timeout = 120000)]
    public async Task CacheGradient_UpdatingSameKey_DoesNotConsumeCapacity()
    {
        await Task.Yield();
        // Arrange - repeatedly refreshing ONE key (e.g. line-search re-evaluation of
        // the same solution) must not evict the other resident entries.
        const int capacity = 4;
        var cache = new DefaultGradientCache<double>(capacity);
        cache.CacheGradient("a", new TestGradientModel<double>(1));
        cache.CacheGradient("b", new TestGradientModel<double>(2));
        cache.CacheGradient("c", new TestGradientModel<double>(3));

        // Act - hammer key "a" many times (updates in place, no new slots).
        for (int i = 0; i < 1000; i++)
        {
            cache.CacheGradient("a", new TestGradientModel<double>(100 + i));
        }

        // Assert - a, b, c all still present (updates never grew the entry count).
        Assert.NotNull(cache.GetCachedGradient("a"));
        Assert.NotNull(cache.GetCachedGradient("b"));
        Assert.NotNull(cache.GetCachedGradient("c"));
    }

    [Fact(Timeout = 120000)]
    public async Task DefaultCapacity_IsPositiveAndBounded()
    {
        await Task.Yield();
        Assert.True(DefaultGradientCache<double>.DefaultCapacity > 0);
        var cache = new DefaultGradientCache<double>();
        Assert.Equal(DefaultGradientCache<double>.DefaultCapacity, cache.Capacity);
    }

    [Theory(Timeout = 120000)]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-100)]
    public async Task Constructor_NonPositiveCapacity_Throws(int capacity)
    {
        await Task.Yield();
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => new DefaultGradientCache<double>(capacity));
        Assert.Equal("capacity", ex.ParamName);
    }

    #endregion

    #region Thread Safety Tests

    [Fact(Timeout = 120000)]
    public async Task ConcurrentCacheOperations_DoesNotThrow()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var exceptions = new List<Exception>();
        var tasks = new List<Task>();

        // Act - Concurrent writes
        for (int i = 0; i < 100; i++)
        {
            int index = i;
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    cache.CacheGradient($"key_{index}", new TestGradientModel<double>(index));
                }
                catch (Exception ex)
                {
                    lock (exceptions) exceptions.Add(ex);
                }
            }));
        }

        // Concurrent reads
        for (int i = 0; i < 100; i++)
        {
            int index = i;
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    cache.GetCachedGradient($"key_{index}");
                }
                catch (Exception ex)
                {
                    lock (exceptions) exceptions.Add(ex);
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        Assert.Empty(exceptions);
    }

    [Fact(Timeout = 120000)]
    public async Task ConcurrentReadWrite_SameKey_ThreadSafe()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        const string sharedKey = "shared";
        var exceptions = new List<Exception>();
        var tasks = new List<Task>();
        const int iterations = 1000;

        // Act - Multiple threads reading and writing to same key
        for (int i = 0; i < 10; i++)
        {
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    for (int j = 0; j < iterations; j++)
                    {
                        cache.CacheGradient(sharedKey, new TestGradientModel<double>(j));
                        cache.GetCachedGradient(sharedKey);
                    }
                }
                catch (Exception ex)
                {
                    lock (exceptions) exceptions.Add(ex);
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        Assert.Empty(exceptions);
    }

    [Fact(Timeout = 120000)]
    public async Task ConcurrentClearAndAccess_ThreadSafe()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var exceptions = new List<Exception>();
        var tasks = new List<Task>();

        // Pre-populate
        for (int i = 0; i < 50; i++)
        {
            cache.CacheGradient($"key_{i}", new TestGradientModel<double>(i));
        }

        // Act - Concurrent clears and accesses
        for (int i = 0; i < 10; i++)
        {
            tasks.Add(Task.Run(() =>
            {
                try
                {
                    for (int j = 0; j < 100; j++)
                    {
                        cache.ClearCache();
                        cache.CacheGradient("temp", new TestGradientModel<double>(j));
                        cache.GetCachedGradient("temp");
                    }
                }
                catch (Exception ex)
                {
                    lock (exceptions) exceptions.Add(ex);
                }
            }));
        }

        Task.WaitAll(tasks.ToArray());

        // Assert
        Assert.Empty(exceptions);
    }

    #endregion

    #region Edge Cases

    [Fact(Timeout = 120000)]
    public async Task CacheGradient_EmptyKey_StoresSuccessfully()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var gradient = new TestGradientModel<double>(1.0);

        // Act
        cache.CacheGradient("", gradient);
        var retrieved = cache.GetCachedGradient("");

        // Assert
        Assert.Same(gradient, retrieved);
    }

    [Fact(Timeout = 120000)]
    public async Task CacheGradient_VeryLongKey_StoresSuccessfully()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var gradient = new TestGradientModel<double>(1.0);
        var longKey = new string('x', 10000);

        // Act
        cache.CacheGradient(longKey, gradient);
        var retrieved = cache.GetCachedGradient(longKey);

        // Assert
        Assert.Same(gradient, retrieved);
    }

    [Fact(Timeout = 120000)]
    public async Task CacheGradient_SpecialCharactersInKey_StoresSuccessfully()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var gradient = new TestGradientModel<double>(1.0);
        const string specialKey = "key!@#$%^&*()_+-=[]{}|;':\",./<>?";

        // Act
        cache.CacheGradient(specialKey, gradient);
        var retrieved = cache.GetCachedGradient(specialKey);

        // Assert
        Assert.Same(gradient, retrieved);
    }

    [Fact(Timeout = 120000)]
    public async Task ClearCache_OnEmptyCache_DoesNotThrow()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();

        // Act & Assert - Should not throw
        var exception = Record.Exception(() => cache.ClearCache());
        Assert.Null(exception);
    }

    [Fact(Timeout = 120000)]
    public async Task ClearCache_CalledMultipleTimes_DoesNotThrow()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        cache.CacheGradient("key", new TestGradientModel<double>(1.0));

        // Act & Assert
        var exception = Record.Exception(() =>
        {
            cache.ClearCache();
            cache.ClearCache();
            cache.ClearCache();
        });
        Assert.Null(exception);
    }

    [Fact(Timeout = 120000)]
    public async Task GetCachedGradient_NullKey_ThrowsArgumentNullException()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();

        // Act & Assert
        var ex = Assert.Throws<ArgumentNullException>(() => cache.GetCachedGradient(null!));
        Assert.Equal("key", ex.ParamName);
    }

    [Fact(Timeout = 120000)]
    public async Task CacheGradient_NullKey_ThrowsArgumentNullException()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var gradient = new TestGradientModel<double>(1.0);

        // Act & Assert
        var ex = Assert.Throws<ArgumentNullException>(() => cache.CacheGradient(null!, gradient));
        Assert.Equal("key", ex.ParamName);
    }

    [Fact(Timeout = 120000)]
    public async Task CacheGradient_NullGradient_ThrowsArgumentNullException()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<double>();

        // Act & Assert
        var ex = Assert.Throws<ArgumentNullException>(() => cache.CacheGradient("key", null!));
        Assert.Equal("gradient", ex.ParamName);
    }

    #endregion

    #region Type Tests

    [Fact(Timeout = 120000)]
    public async Task GradientCache_FloatType_WorksCorrectly()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<float>();
        var gradient = new TestGradientModel<float>(3.14f);

        // Act
        cache.CacheGradient("float_key", gradient);
        var retrieved = cache.GetCachedGradient("float_key");

        // Assert
        Assert.Same(gradient, retrieved);
    }

    [Fact(Timeout = 120000)]
    public async Task GradientCache_DecimalType_WorksCorrectly()
    {
        await Task.Yield();
        // Arrange
        var cache = new DefaultGradientCache<decimal>();
        var gradient = new TestGradientModel<decimal>(3.14159265359m);

        // Act
        cache.CacheGradient("decimal_key", gradient);
        var retrieved = cache.GetCachedGradient("decimal_key");

        // Assert
        Assert.Same(gradient, retrieved);
    }

    #endregion

    #region Test Helpers

    /// <summary>
    /// Simple test implementation of IGradientModel for testing purposes.
    /// </summary>
    private class TestGradientModel<T> : IGradientModel<T>
    {
        private readonly T _value;
        private readonly Vector<T>? _parameters;

        public TestGradientModel(T value)
        {
            _value = value;
            _parameters = null;
        }

        public Vector<T> Parameters => _parameters ?? throw new NotSupportedException("Test gradient has no parameters set.");

        public T Evaluate(Vector<T> input)
        {
            return _value;
        }
    }

    #endregion
}
