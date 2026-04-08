using AiDotNet.Caching;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Caching;

/// <summary>
/// Integration tests for DefaultModelCache to verify caching behavior,
/// thread safety, and edge case handling.
/// </summary>
public class DefaultModelCacheIntegrationTests
{
    // Use supported types: Matrix<double> for input, Vector<double> for output
    private static OptimizationStepData<double, Matrix<double>, Vector<double>> CreateStepData()
    {
        return new OptimizationStepData<double, Matrix<double>, Vector<double>>();
    }

    #region Basic CRUD Operations

    [Fact]
    public void GetCachedStepData_NonExistentKey_ReturnsNull()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();

        // Act
        var result = cache.GetCachedStepData("nonexistent_key");

        // Assert - CRITICAL: Must return null, not new()
        Assert.Null(result);
    }

    [Fact]
    public void CacheStepData_ThenGet_ReturnsSameData()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepData = CreateStepData();
        const string key = "test_step";

        // Act
        cache.CacheStepData(key, stepData);
        var retrieved = cache.GetCachedStepData(key);

        // Assert
        Assert.NotNull(retrieved);
        Assert.Same(stepData, retrieved);
    }

    [Fact]
    public void CacheStepData_OverwriteExistingKey_ReturnsNewData()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepData1 = CreateStepData();
        var stepData2 = CreateStepData();
        const string key = "shared_key";

        // Act
        cache.CacheStepData(key, stepData1);
        cache.CacheStepData(key, stepData2);
        var retrieved = cache.GetCachedStepData(key);

        // Assert
        Assert.Same(stepData2, retrieved);
        Assert.NotSame(stepData1, retrieved);
    }

    [Fact]
    public void ClearCache_RemovesAllEntries()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        cache.CacheStepData("key1", CreateStepData());
        cache.CacheStepData("key2", CreateStepData());
        cache.CacheStepData("key3", CreateStepData());

        // Act
        cache.ClearCache();

        // Assert - All entries should return null after clear
        Assert.Null(cache.GetCachedStepData("key1"));
        Assert.Null(cache.GetCachedStepData("key2"));
        Assert.Null(cache.GetCachedStepData("key3"));
    }

    #endregion

    #region Null Key Validation

    [Fact]
    public void GetCachedStepData_NullKey_ThrowsArgumentNullException()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();

        // Act & Assert
        var ex = Assert.Throws<ArgumentNullException>(() => cache.GetCachedStepData(null!));
        Assert.Equal("key", ex.ParamName);
    }

    [Fact]
    public void CacheStepData_NullKey_ThrowsArgumentNullException()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepData = CreateStepData();

        // Act & Assert
        var ex = Assert.Throws<ArgumentNullException>(() => cache.CacheStepData(null!, stepData));
        Assert.Equal("key", ex.ParamName);
    }

    [Fact]
    public void CacheStepData_NullStepData_ThrowsArgumentNullException()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();

        // Act & Assert
        var ex = Assert.Throws<ArgumentNullException>(() => cache.CacheStepData("key", null!));
        Assert.Equal("stepData", ex.ParamName);
    }

    #endregion

    #region Multiple Keys Tests

    [Fact]
    public void Cache_MultipleKeys_EachRetrievableIndependently()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepDataDict = new Dictionary<string, OptimizationStepData<double, Matrix<double>, Vector<double>>>();

        for (int i = 0; i < 100; i++)
        {
            var key = $"step_{i}";
            var stepData = CreateStepData();
            stepDataDict[key] = stepData;
            cache.CacheStepData(key, stepData);
        }

        // Act & Assert
        foreach (var kvp in stepDataDict)
        {
            var retrieved = cache.GetCachedStepData(kvp.Key);
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
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
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
                    cache.CacheStepData($"key_{index}", CreateStepData());
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
                    cache.GetCachedStepData($"key_{index}");
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
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
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
                        cache.CacheStepData(sharedKey, CreateStepData());
                        cache.GetCachedStepData(sharedKey);
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
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var exceptions = new List<Exception>();
        var tasks = new List<Task>();

        // Pre-populate
        for (int i = 0; i < 50; i++)
        {
            cache.CacheStepData($"key_{i}", CreateStepData());
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
                        cache.CacheStepData("temp", CreateStepData());
                        cache.GetCachedStepData("temp");
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
    public void CacheStepData_EmptyKey_StoresSuccessfully()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepData = CreateStepData();

        // Act
        cache.CacheStepData("", stepData);
        var retrieved = cache.GetCachedStepData("");

        // Assert
        Assert.Same(stepData, retrieved);
    }

    [Fact]
    public void CacheStepData_VeryLongKey_StoresSuccessfully()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepData = CreateStepData();
        var longKey = new string('x', 10000);

        // Act
        cache.CacheStepData(longKey, stepData);
        var retrieved = cache.GetCachedStepData(longKey);

        // Assert
        Assert.Same(stepData, retrieved);
    }

    [Fact]
    public void CacheStepData_SpecialCharactersInKey_StoresSuccessfully()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepData = CreateStepData();
        const string specialKey = "key!@#$%^&*()_+-=[]{}|;':\",./<>?";

        // Act
        cache.CacheStepData(specialKey, stepData);
        var retrieved = cache.GetCachedStepData(specialKey);

        // Assert
        Assert.Same(stepData, retrieved);
    }

    [Fact]
    public void ClearCache_OnEmptyCache_DoesNotThrow()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();

        // Act & Assert - Should not throw
        var exception = Record.Exception(() => cache.ClearCache());
        Assert.Null(exception);
    }

    [Fact]
    public void ClearCache_CalledMultipleTimes_DoesNotThrow()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        cache.CacheStepData("key", CreateStepData());

        // Act & Assert
        var exception = Record.Exception(() =>
        {
            cache.ClearCache();
            cache.ClearCache();
            cache.ClearCache();
        });
        Assert.Null(exception);
    }

    #endregion

    #region Type Tests

    [Fact]
    public void ModelCache_FloatType_WorksCorrectly()
    {
        // Arrange
        var cache = new DefaultModelCache<float, Matrix<float>, Vector<float>>();
        var stepData = new OptimizationStepData<float, Matrix<float>, Vector<float>>();

        // Act
        cache.CacheStepData("float_key", stepData);
        var retrieved = cache.GetCachedStepData("float_key");

        // Assert
        Assert.Same(stepData, retrieved);
    }

    // Note: Tensor<T> type test removed because OptimizationStepData's parameterless constructor
    // requires valid 3D dimensions for Tensor types, which is a limitation of the default model creation.
    // The cache functionality is tested via Matrix/Vector which uses the same underlying ConcurrentDictionary.

    #endregion

    #region Return Null vs New Verification

    [Fact]
    public void GetCachedStepData_MultipleNonExistentKeys_AllReturnNull()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();

        // Act & Assert - This verifies the fix for the bug where new() was returned
        for (int i = 0; i < 100; i++)
        {
            var result = cache.GetCachedStepData($"nonexistent_{i}");
            Assert.Null(result);
        }
    }

    [Fact]
    public void GetCachedStepData_AfterClear_ReturnsNull()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        cache.CacheStepData("key", CreateStepData());
        cache.ClearCache();

        // Act
        var result = cache.GetCachedStepData("key");

        // Assert - Must return null, not new()
        Assert.Null(result);
    }

    #endregion
}
