using AiDotNet.Caching;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Caching;

/// <summary>
/// Integration tests for DefaultGradientCache to verify caching behavior,
/// thread safety, and edge case handling.
/// </summary>
public class DefaultGradientCacheIntegrationTests
{
    #region Basic CRUD Operations

    [Fact]
    public void GetCachedGradient_NonExistentKey_ReturnsNull()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();

        // Act
        var result = cache.GetCachedGradient("nonexistent_key");

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void CacheGradient_ThenGet_ReturnsSameGradient()
    {
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

    [Fact]
    public void CacheGradient_OverwriteExistingKey_ReturnsNewGradient()
    {
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

    [Fact]
    public void ClearCache_RemovesAllEntries()
    {
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

    [Fact]
    public void Cache_MultipleKeys_EachRetrievableIndependently()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();
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

    #region Thread Safety Tests

    [Fact]
    public void ConcurrentCacheOperations_DoesNotThrow()
    {
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

    [Fact]
    public void ConcurrentReadWrite_SameKey_ThreadSafe()
    {
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

    [Fact]
    public void ConcurrentClearAndAccess_ThreadSafe()
    {
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

    [Fact]
    public void CacheGradient_EmptyKey_StoresSuccessfully()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var gradient = new TestGradientModel<double>(1.0);

        // Act
        cache.CacheGradient("", gradient);
        var retrieved = cache.GetCachedGradient("");

        // Assert
        Assert.Same(gradient, retrieved);
    }

    [Fact]
    public void CacheGradient_VeryLongKey_StoresSuccessfully()
    {
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

    [Fact]
    public void CacheGradient_SpecialCharactersInKey_StoresSuccessfully()
    {
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

    [Fact]
    public void ClearCache_OnEmptyCache_DoesNotThrow()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();

        // Act & Assert - Should not throw
        var exception = Record.Exception(() => cache.ClearCache());
        Assert.Null(exception);
    }

    [Fact]
    public void ClearCache_CalledMultipleTimes_DoesNotThrow()
    {
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

    [Fact]
    public void GetCachedGradient_NullKey_ThrowsArgumentNullException()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => cache.GetCachedGradient(null!));
    }

    [Fact]
    public void CacheGradient_NullKey_ThrowsArgumentNullException()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var gradient = new TestGradientModel<double>(1.0);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => cache.CacheGradient(null!, gradient));
    }

    [Fact]
    public void CacheGradient_NullGradient_ThrowsArgumentNullException()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => cache.CacheGradient("key", null!));
    }

    #endregion

    #region Type Tests

    [Fact]
    public void GradientCache_FloatType_WorksCorrectly()
    {
        // Arrange
        var cache = new DefaultGradientCache<float>();
        var gradient = new TestGradientModel<float>(3.14f);

        // Act
        cache.CacheGradient("float_key", gradient);
        var retrieved = cache.GetCachedGradient("float_key");

        // Assert
        Assert.Same(gradient, retrieved);
    }

    [Fact]
    public void GradientCache_DecimalType_WorksCorrectly()
    {
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
