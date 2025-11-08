# Issue #358: Junior Developer Implementation Guide - Caching Infrastructure

## Overview
This guide helps you create **unit tests** for the caching infrastructure: DefaultGradientCache, DefaultModelCache, and DeterministicCacheKeyGenerator. These classes currently have **0% test coverage** despite being critical for performance optimization.

**Goal**: Write comprehensive unit tests to ensure caching works correctly and improves performance.

---

## Understanding the Classes

### DefaultGradientCache<T> (`src/Caching/DefaultGradientCache.cs`)
Caches pre-calculated gradients for machine learning models.

**Key Concepts**:
- **Gradient**: Mathematical calculation showing how to adjust model parameters
- **Cache**: Temporary storage to avoid recalculating the same values
- **Thread-safe**: Multiple threads can access simultaneously using ConcurrentDictionary

**Key Methods**:
- `GetCachedGradient(string key)`: Retrieve cached gradient by key
- `CacheGradient(string key, IGradientModel<T> gradient)`: Store gradient with key
- `ClearCache()`: Remove all cached gradients

**Example Usage**:
```csharp
var cache = new DefaultGradientCache<double>();

// Store gradient
var gradient = ...; // Some IGradientModel<double>
cache.CacheGradient("layer1_weights", gradient);

// Retrieve gradient
var cached = cache.GetCachedGradient("layer1_weights");

// Clear all
cache.ClearCache();
```

**Why It Matters**:
- Gradients are expensive to calculate
- Same gradients are often needed multiple times
- Caching can dramatically speed up training

---

### DefaultModelCache<T, TInput, TOutput> (`src/Caching/DefaultModelCache.cs`)
Caches optimization step data during model training.

**Key Concepts**:
- **Optimization Step**: One iteration of improving the model
- **Step Data**: Information about model state, loss, gradients at that step
- **Deterministic Key**: A unique identifier that stays the same across restarts

**Key Methods**:
- `GetCachedStepData(string key)`: Retrieve cached step data
- `CacheStepData(string key, OptimizationStepData<T, TInput, TOutput> stepData)`: Store step data
- `ClearCache()`: Remove all cached data
- `GenerateCacheKey(IFullModel<T, TInput, TOutput> solution, OptimizationInputData<T, TInput, TOutput> inputData)`: Generate deterministic key

**Example Usage**:
```csharp
var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();

// Generate key from model and data
string key = cache.GenerateCacheKey(model, inputData);

// Store step data
var stepData = new OptimizationStepData<double, Matrix<double>, Vector<double>>
{
    Loss = 0.5,
    Iteration = 100
};
cache.CacheStepData(key, stepData);

// Retrieve later
var cached = cache.GetCachedStepData(key);
```

**Why It Matters**:
- Training can be interrupted and resumed
- Avoid repeating expensive calculations
- Track training progress over time

---

### DeterministicCacheKeyGenerator (`src/Caching/DeterministicCacheKeyGenerator.cs`)
Generates consistent cache keys using SHA-256 hashing.

**Key Concepts**:
- **Deterministic**: Same input always produces same key
- **SHA-256**: Cryptographic hash function
- **Process-independent**: Keys work across restarts
- **Collision-resistant**: Different inputs produce different keys

**Key Methods**:
- `GenerateKey<T>(Vector<T> parameters, string inputDataDescriptor)`: Generate SHA-256 hash key
- `CreateInputDataDescriptor<T, TInput, TOutput>(...)`: Create stable descriptor for input data
- `GetShapeDescriptor<TData>(TData data)`: Get shape string for data structures

**Example Key Format**:
```
Input: parameters = [1.0, 2.0, 3.0], descriptor = "Matrix(100,10)"
Process: "params:3|1.0,2.0,3.0|input:Matrix(100,10)" -> SHA-256
Output: "a3b5c7d9e1f2...64-character hex string"
```

**Why It Matters**:
- Keys must be stable across restarts (unlike GetHashCode())
- Enables persistent caching
- Ensures cache correctness

---

## Phase 1: DefaultGradientCache Tests

### Test File: `tests/UnitTests/Caching/DefaultGradientCacheTests.cs`

```csharp
using AiDotNet.Caching;
using Xunit;
using Moq;

namespace AiDotNet.Tests.UnitTests.Caching;

public class DefaultGradientCacheTests
{
    [Fact]
    public void Constructor_CreatesEmptyCache()
    {
        // Act
        var cache = new DefaultGradientCache<double>();

        // Assert
        Assert.NotNull(cache);
    }

    [Fact]
    public void GetCachedGradient_EmptyCache_ReturnsNull()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();

        // Act
        var result = cache.GetCachedGradient("nonexistent");

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void CacheGradient_StoresGradient()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var mockGradient = new Mock<IGradientModel<double>>();
        string key = "test_gradient";

        // Act
        cache.CacheGradient(key, mockGradient.Object);
        var result = cache.GetCachedGradient(key);

        // Assert
        Assert.NotNull(result);
        Assert.Same(mockGradient.Object, result);
    }

    [Fact]
    public void CacheGradient_OverwritesExisting()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var mockGradient1 = new Mock<IGradientModel<double>>();
        var mockGradient2 = new Mock<IGradientModel<double>>();
        string key = "test_gradient";

        // Act
        cache.CacheGradient(key, mockGradient1.Object);
        cache.CacheGradient(key, mockGradient2.Object); // Overwrite

        var result = cache.GetCachedGradient(key);

        // Assert
        Assert.Same(mockGradient2.Object, result);
        Assert.NotSame(mockGradient1.Object, result);
    }

    [Fact]
    public void ClearCache_RemovesAllEntries()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var mockGradient1 = new Mock<IGradientModel<double>>();
        var mockGradient2 = new Mock<IGradientModel<double>>();

        cache.CacheGradient("key1", mockGradient1.Object);
        cache.CacheGradient("key2", mockGradient2.Object);

        // Act
        cache.ClearCache();

        // Assert
        Assert.Null(cache.GetCachedGradient("key1"));
        Assert.Null(cache.GetCachedGradient("key2"));
    }

    [Fact]
    public void Cache_SupportsMultipleKeys()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var mockGradient1 = new Mock<IGradientModel<double>>();
        var mockGradient2 = new Mock<IGradientModel<double>>();
        var mockGradient3 = new Mock<IGradientModel<double>>();

        // Act
        cache.CacheGradient("layer1", mockGradient1.Object);
        cache.CacheGradient("layer2", mockGradient2.Object);
        cache.CacheGradient("layer3", mockGradient3.Object);

        // Assert
        Assert.Same(mockGradient1.Object, cache.GetCachedGradient("layer1"));
        Assert.Same(mockGradient2.Object, cache.GetCachedGradient("layer2"));
        Assert.Same(mockGradient3.Object, cache.GetCachedGradient("layer3"));
    }

    [Fact]
    public void Cache_IsThreadSafe()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();
        int numThreads = 10;
        int operationsPerThread = 100;
        var tasks = new Task[numThreads];

        // Act
        for (int t = 0; t < numThreads; t++)
        {
            int threadId = t;
            tasks[t] = Task.Run(() =>
            {
                for (int i = 0; i < operationsPerThread; i++)
                {
                    var mockGradient = new Mock<IGradientModel<double>>();
                    string key = $"thread{threadId}_op{i}";
                    cache.CacheGradient(key, mockGradient.Object);
                    var retrieved = cache.GetCachedGradient(key);
                    Assert.NotNull(retrieved);
                }
            });
        }

        Task.WaitAll(tasks);

        // Assert - No exceptions thrown means thread-safe
        Assert.True(true);
    }
}
```

---

## Phase 2: DefaultModelCache Tests

### Test File: `tests/UnitTests/Caching/DefaultModelCacheTests.cs`

```csharp
using AiDotNet.Caching;
using AiDotNet.LinearAlgebra;
using Xunit;
using Moq;

namespace AiDotNet.Tests.UnitTests.Caching;

public class DefaultModelCacheTests
{
    [Fact]
    public void Constructor_CreatesEmptyCache()
    {
        // Act
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();

        // Assert
        Assert.NotNull(cache);
    }

    [Fact]
    public void GetCachedStepData_EmptyCache_ReturnsNewInstance()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();

        // Act
        var result = cache.GetCachedStepData("nonexistent");

        // Assert
        Assert.NotNull(result); // Returns new empty instance, not null
    }

    [Fact]
    public void CacheStepData_StoresData()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepData = new OptimizationStepData<double, Matrix<double>, Vector<double>>
        {
            Loss = 0.5,
            Iteration = 100
        };
        string key = "step_100";

        // Act
        cache.CacheStepData(key, stepData);
        var result = cache.GetCachedStepData(key);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(0.5, result.Loss);
        Assert.Equal(100, result.Iteration);
    }

    [Fact]
    public void CacheStepData_OverwritesExisting()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepData1 = new OptimizationStepData<double, Matrix<double>, Vector<double>>
        {
            Loss = 0.5,
            Iteration = 100
        };
        var stepData2 = new OptimizationStepData<double, Matrix<double>, Vector<double>>
        {
            Loss = 0.3,
            Iteration = 200
        };
        string key = "step";

        // Act
        cache.CacheStepData(key, stepData1);
        cache.CacheStepData(key, stepData2); // Overwrite

        var result = cache.GetCachedStepData(key);

        // Assert
        Assert.Equal(0.3, result.Loss);
        Assert.Equal(200, result.Iteration);
    }

    [Fact]
    public void ClearCache_RemovesAllEntries()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepData1 = new OptimizationStepData<double, Matrix<double>, Vector<double>>
        {
            Loss = 0.5
        };
        var stepData2 = new OptimizationStepData<double, Matrix<double>, Vector<double>>
        {
            Loss = 0.3
        };

        cache.CacheStepData("key1", stepData1);
        cache.CacheStepData("key2", stepData2);

        // Act
        cache.ClearCache();

        // Assert
        var result1 = cache.GetCachedStepData("key1");
        var result2 = cache.GetCachedStepData("key2");

        // Should return new empty instances
        Assert.NotEqual(0.5, result1.Loss);
        Assert.NotEqual(0.3, result2.Loss);
    }

    [Fact]
    public void GenerateCacheKey_SameInputs_ProducesSameKey()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var mockModel = new Mock<IFullModel<double, Matrix<double>, Vector<double>>>();
        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        mockModel.Setup(m => m.GetParameters()).Returns(parameters);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = new Matrix<double>(100, 10),
            YTrain = new Vector<double>(100)
        };

        // Act
        string key1 = cache.GenerateCacheKey(mockModel.Object, inputData);
        string key2 = cache.GenerateCacheKey(mockModel.Object, inputData);

        // Assert
        Assert.Equal(key1, key2);
    }

    [Fact]
    public void GenerateCacheKey_DifferentInputs_ProducesDifferentKeys()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var mockModel1 = new Mock<IFullModel<double, Matrix<double>, Vector<double>>>();
        var mockModel2 = new Mock<IFullModel<double, Matrix<double>, Vector<double>>>();

        var parameters1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var parameters2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });

        mockModel1.Setup(m => m.GetParameters()).Returns(parameters1);
        mockModel2.Setup(m => m.GetParameters()).Returns(parameters2);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = new Matrix<double>(100, 10),
            YTrain = new Vector<double>(100)
        };

        // Act
        string key1 = cache.GenerateCacheKey(mockModel1.Object, inputData);
        string key2 = cache.GenerateCacheKey(mockModel2.Object, inputData);

        // Assert
        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void GenerateCacheKey_NullModel_ThrowsException()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = new Matrix<double>(100, 10),
            YTrain = new Vector<double>(100)
        };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            cache.GenerateCacheKey(null, inputData));
    }

    [Fact]
    public void GenerateCacheKey_NullInputData_ThrowsException()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var mockModel = new Mock<IFullModel<double, Matrix<double>, Vector<double>>>();
        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        mockModel.Setup(m => m.GetParameters()).Returns(parameters);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            cache.GenerateCacheKey(mockModel.Object, null));
    }
}
```

---

## Phase 3: DeterministicCacheKeyGenerator Tests

### Test File: `tests/UnitTests/Caching/DeterministicCacheKeyGeneratorTests.cs`

```csharp
using AiDotNet.Caching;
using AiDotNet.LinearAlgebra;
using Xunit;
using System.Reflection;

namespace AiDotNet.Tests.UnitTests.Caching;

public class DeterministicCacheKeyGeneratorTests
{
    // Helper method to access internal static class
    private static Type GetGeneratorType()
    {
        var assembly = Assembly.GetAssembly(typeof(DefaultModelCache<,,>));
        return assembly.GetType("AiDotNet.Caching.DeterministicCacheKeyGenerator");
    }

    [Fact]
    public void GenerateKey_SameInputs_ProducesSameKey()
    {
        // Arrange
        var type = GetGeneratorType();
        var method = type.GetMethod("GenerateKey", BindingFlags.Public | BindingFlags.Static);

        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        string descriptor = "Matrix(100,10)";

        // Act
        string key1 = (string)method.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { parameters, descriptor });
        string key2 = (string)method.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { parameters, descriptor });

        // Assert
        Assert.Equal(key1, key2);
    }

    [Fact]
    public void GenerateKey_DifferentParameters_ProducesDifferentKeys()
    {
        // Arrange
        var type = GetGeneratorType();
        var method = type.GetMethod("GenerateKey", BindingFlags.Public | BindingFlags.Static);

        var parameters1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var parameters2 = new Vector<double>(new[] { 4.0, 5.0, 6.0 });
        string descriptor = "Matrix(100,10)";

        // Act
        string key1 = (string)method.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { parameters1, descriptor });
        string key2 = (string)method.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { parameters2, descriptor });

        // Assert
        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void GenerateKey_DifferentDescriptors_ProducesDifferentKeys()
    {
        // Arrange
        var type = GetGeneratorType();
        var method = type.GetMethod("GenerateKey", BindingFlags.Public | BindingFlags.Static);

        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        string descriptor1 = "Matrix(100,10)";
        string descriptor2 = "Matrix(200,20)";

        // Act
        string key1 = (string)method.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { parameters, descriptor1 });
        string key2 = (string)method.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { parameters, descriptor2 });

        // Assert
        Assert.NotEqual(key1, key2);
    }

    [Fact]
    public void GenerateKey_ReturnsHexString()
    {
        // Arrange
        var type = GetGeneratorType();
        var method = type.GetMethod("GenerateKey", BindingFlags.Public | BindingFlags.Static);

        var parameters = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        string descriptor = "Matrix(100,10)";

        // Act
        string key = (string)method.MakeGenericMethod(typeof(double))
            .Invoke(null, new object[] { parameters, descriptor });

        // Assert
        Assert.Equal(64, key.Length); // SHA-256 produces 64 hex characters
        Assert.Matches("^[0-9a-f]+$", key); // All lowercase hex
    }

    [Fact]
    public void CreateInputDataDescriptor_Matrix_ReturnsCorrectFormat()
    {
        // Arrange
        var type = GetGeneratorType();
        var method = type.GetMethod("CreateInputDataDescriptor", BindingFlags.Public | BindingFlags.Static);

        var xTrain = new Matrix<double>(100, 10);
        var yTrain = new Vector<double>(100);

        // Act
        string descriptor = (string)method.MakeGenericMethod(typeof(double), typeof(Matrix<double>), typeof(Vector<double>))
            .Invoke(null, new object[] { xTrain, yTrain, null, null, null, null });

        // Assert
        Assert.Contains("train:", descriptor);
        Assert.Contains("Matrix(100,10)", descriptor);
        Assert.Contains("Vector(100)", descriptor);
    }

    [Fact]
    public void CreateInputDataDescriptor_WithValidation_IncludesValidationData()
    {
        // Arrange
        var type = GetGeneratorType();
        var method = type.GetMethod("CreateInputDataDescriptor", BindingFlags.Public | BindingFlags.Static);

        var xTrain = new Matrix<double>(100, 10);
        var yTrain = new Vector<double>(100);
        var xVal = new Matrix<double>(20, 10);
        var yVal = new Vector<double>(20);

        // Act
        string descriptor = (string)method.MakeGenericMethod(typeof(double), typeof(Matrix<double>), typeof(Vector<double>))
            .Invoke(null, new object[] { xTrain, yTrain, xVal, yVal, null, null });

        // Assert
        Assert.Contains("train:", descriptor);
        Assert.Contains("val:", descriptor);
        Assert.Contains("Matrix(20,10)", descriptor);
    }

    [Fact]
    public void GenerateKey_NullParameters_ThrowsException()
    {
        // Arrange
        var type = GetGeneratorType();
        var method = type.GetMethod("GenerateKey", BindingFlags.Public | BindingFlags.Static);

        string descriptor = "Matrix(100,10)";

        // Act & Assert
        var exception = Assert.Throws<TargetInvocationException>(() =>
            method.MakeGenericMethod(typeof(double))
                .Invoke(null, new object[] { null, descriptor }));

        Assert.IsType<ArgumentNullException>(exception.InnerException);
    }
}
```

---

## Phase 4: Cache Hit/Miss Performance Tests

### Test File: `tests/UnitTests/Caching/CachePerformanceTests.cs`

```csharp
using AiDotNet.Caching;
using Xunit;
using Moq;
using System.Diagnostics;

namespace AiDotNet.Tests.UnitTests.Caching;

public class CachePerformanceTests
{
    [Fact]
    public void GradientCache_CacheHit_FasterThanRecalculation()
    {
        // Arrange
        var cache = new DefaultGradientCache<double>();
        var mockGradient = new Mock<IGradientModel<double>>();
        string key = "expensive_gradient";

        cache.CacheGradient(key, mockGradient.Object);

        // Act - Measure cache hit time
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < 10000; i++)
        {
            var result = cache.GetCachedGradient(key);
        }
        sw.Stop();

        // Assert
        Assert.True(sw.ElapsedMilliseconds < 100); // Should be very fast
    }

    [Fact]
    public void ModelCache_StoresAndRetrievesQuickly()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var stepData = new OptimizationStepData<double, Matrix<double>, Vector<double>>
        {
            Loss = 0.5,
            Iteration = 100
        };

        // Act
        var sw = Stopwatch.StartNew();
        cache.CacheStepData("key", stepData);
        var retrieved = cache.GetCachedStepData("key");
        sw.Stop();

        // Assert
        Assert.True(sw.ElapsedMilliseconds < 10);
        Assert.Equal(0.5, retrieved.Loss);
    }

    [Fact]
    public void CacheKeyGeneration_Completes_InReasonableTime()
    {
        // Arrange
        var cache = new DefaultModelCache<double, Matrix<double>, Vector<double>>();
        var mockModel = new Mock<IFullModel<double, Matrix<double>, Vector<double>>>();
        var parameters = new Vector<double>(1000); // Large parameter vector
        for (int i = 0; i < 1000; i++)
            parameters[i] = i * 0.1;

        mockModel.Setup(m => m.GetParameters()).Returns(parameters);

        var inputData = new OptimizationInputData<double, Matrix<double>, Vector<double>>
        {
            XTrain = new Matrix<double>(1000, 100),
            YTrain = new Vector<double>(1000)
        };

        // Act
        var sw = Stopwatch.StartNew();
        string key = cache.GenerateCacheKey(mockModel.Object, inputData);
        sw.Stop();

        // Assert
        Assert.NotNull(key);
        Assert.True(sw.ElapsedMilliseconds < 100); // Should be fast even for large inputs
    }
}
```

---

## Common Testing Patterns

### Cache Behavior Tests

1. **Cache Miss Returns Appropriate Value**
   ```csharp
   var result = cache.GetCached("nonexistent");
   // Should return null or new empty instance
   ```

2. **Cache Overwrite Works**
   ```csharp
   cache.Set("key", value1);
   cache.Set("key", value2);
   Assert.Equal(value2, cache.Get("key"));
   ```

3. **Clear Removes All**
   ```csharp
   cache.Set("key1", value1);
   cache.Set("key2", value2);
   cache.Clear();
   Assert.Null(cache.Get("key1"));
   ```

### Determinism Tests

1. **Same Input Same Key**
   ```csharp
   string key1 = GenerateKey(input);
   string key2 = GenerateKey(input);
   Assert.Equal(key1, key2);
   ```

2. **Different Input Different Key**
   ```csharp
   string key1 = GenerateKey(input1);
   string key2 = GenerateKey(input2);
   Assert.NotEqual(key1, key2);
   ```

---

## Running Tests

```bash
# Run all Caching tests
dotnet test --filter "FullyQualifiedName~Caching"

# Run specific test class
dotnet test --filter "FullyQualifiedName~DefaultGradientCacheTests"

# Run with coverage
dotnet test /p:CollectCoverage=true
```

---

## Success Criteria

- [ ] DefaultGradientCache tests cover: storage, retrieval, clearing, thread safety
- [ ] DefaultModelCache tests cover: step data caching, key generation, null handling
- [ ] DeterministicCacheKeyGenerator tests cover: determinism, hash format, descriptors
- [ ] Performance tests demonstrate caching benefits
- [ ] All tests pass with green checkmarks
- [ ] Code coverage increases from 0% to >80%
- [ ] Thread safety verified with concurrent tests

---

## Common Pitfalls

1. **Don't test implementation details** - Test behavior, not internals
2. **Don't forget thread safety** - Caches must handle concurrent access
3. **Do verify determinism** - Same input must produce same key every time
4. **Do test null/empty cases** - Cache should handle missing data gracefully
5. **Do test performance benefits** - Show that caching actually helps
6. **Do test key uniqueness** - Different inputs must produce different keys

Start with DefaultGradientCache (simplest), then DefaultModelCache, then DeterministicCacheKeyGenerator. Build incrementally!
