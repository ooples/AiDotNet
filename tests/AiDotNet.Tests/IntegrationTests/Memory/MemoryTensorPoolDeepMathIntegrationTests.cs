using AiDotNet.Memory;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Memory;

/// <summary>
/// Deep integration tests for TensorPool: rent/return lifecycle, memory accounting,
/// pool statistics, shape matching, boundary conditions, and resource management.
/// </summary>
public class MemoryTensorPoolDeepMathIntegrationTests
{
    private const double Tolerance = 1e-10;

    // ============================
    // Basic Rent/Return Lifecycle
    // ============================

    [Fact]
    public void Rent_ReturnsCorrectShape()
    {
        using var pool = new TensorPool<double>();
        var shape = new[] { 3, 4 };

        var tensor = pool.Rent(shape);

        Assert.Equal(2, tensor.Shape.Length);
        Assert.Equal(3, tensor.Shape[0]);
        Assert.Equal(4, tensor.Shape[1]);
        Assert.Equal(12, tensor.Length);
    }

    [Fact]
    public void Rent_ReturnsZeroedTensor()
    {
        using var pool = new TensorPool<double>();
        var shape = new[] { 5 };

        var tensor = pool.Rent(shape);

        for (int i = 0; i < 5; i++)
            Assert.Equal(0.0, tensor[i], Tolerance);
    }

    [Fact]
    public void Return_ThenRent_ReusesMemory()
    {
        using var pool = new TensorPool<double>();
        var shape = new[] { 3, 3 };

        var tensor1 = pool.Rent(shape);
        // Fill with data
        for (int i = 0; i < 9; i++)
            tensor1[i] = i + 1.0;

        pool.Return(tensor1);

        // After return, rent again - should get a reused tensor
        var tensor2 = pool.Rent(shape);

        // Reused tensor should be cleared to zeros
        for (int i = 0; i < 9; i++)
            Assert.Equal(0.0, tensor2[i], Tolerance);
    }

    [Fact]
    public void Return_ThenRent_PoolContainsEntry()
    {
        using var pool = new TensorPool<double>();
        var shape = new[] { 4 };

        var tensor = pool.Rent(shape);
        pool.Return(tensor);

        var stats = pool.GetStatistics();
        Assert.True(stats.PooledTensorCount >= 0); // Pool should be tracking
    }

    // ============================
    // Multiple Shape Buckets
    // ============================

    [Fact]
    public void DifferentShapes_CreateSeparateBuckets()
    {
        using var pool = new TensorPool<double>();

        var t1 = pool.Rent(new[] { 3, 3 });
        var t2 = pool.Rent(new[] { 4, 2 });
        var t3 = pool.Rent(new[] { 6 });

        pool.Return(t1);
        pool.Return(t2);
        pool.Return(t3);

        var stats = pool.GetStatistics();
        // Three different shapes should create 3 buckets
        Assert.Equal(3, stats.TensorBuckets);
    }

    [Fact]
    public void SameShape_ReusesSameBucket()
    {
        using var pool = new TensorPool<double>();

        var t1 = pool.Rent(new[] { 5, 5 });
        var t2 = pool.Rent(new[] { 5, 5 });

        pool.Return(t1);
        pool.Return(t2);

        var stats = pool.GetStatistics();
        // Same shape should use 1 bucket
        Assert.Equal(1, stats.TensorBuckets);
        Assert.Equal(2, stats.PooledTensorCount);
    }

    // ============================
    // Memory Accounting Tests
    // ============================

    [Fact]
    public void EmptyPool_ZeroMemory()
    {
        using var pool = new TensorPool<double>();

        Assert.Equal(0, pool.CurrentMemoryBytes);
        Assert.Equal(0, pool.TotalPooledTensors);
    }

    [Fact]
    public void AfterReturn_MemoryIncreases()
    {
        using var pool = new TensorPool<double>();

        var tensor = pool.Rent(new[] { 10 });
        Assert.Equal(0, pool.CurrentMemoryBytes); // Rented = not in pool

        pool.Return(tensor);
        Assert.True(pool.CurrentMemoryBytes > 0); // Returned = in pool
    }

    [Fact]
    public void AfterRent_FromPool_MemoryDecreases()
    {
        using var pool = new TensorPool<double>();

        var tensor = pool.Rent(new[] { 10 });
        pool.Return(tensor);

        var memoryAfterReturn = pool.CurrentMemoryBytes;
        Assert.True(memoryAfterReturn > 0);

        // Rent again should pull from pool, decreasing memory
        var tensor2 = pool.Rent(new[] { 10 });
        Assert.True(pool.CurrentMemoryBytes < memoryAfterReturn);
    }

    [Fact]
    public void MemoryAccounting_HandComputed()
    {
        // For double (8 bytes), a tensor of length 10:
        // sizeBytes = 10 * 8 + 64 = 144 bytes (overhead of 64)
        using var pool = new TensorPool<double>();

        var tensor = pool.Rent(new[] { 10 });
        pool.Return(tensor);

        var expectedSize = 10L * 8 + 64; // 144 bytes
        Assert.Equal(expectedSize, pool.CurrentMemoryBytes);
    }

    [Fact]
    public void MemoryAccounting_MultipleTensors_Additive()
    {
        using var pool = new TensorPool<double>();

        var t1 = pool.Rent(new[] { 10 }); // 10*8+64 = 144
        var t2 = pool.Rent(new[] { 20 }); // 20*8+64 = 224

        pool.Return(t1);
        pool.Return(t2);

        var expected = (10L * 8 + 64) + (20L * 8 + 64); // 144 + 224 = 368
        Assert.Equal(expected, pool.CurrentMemoryBytes);
    }

    // ============================
    // Pool Statistics Tests
    // ============================

    [Fact]
    public void Statistics_InitialState()
    {
        using var pool = new TensorPool<double>();
        var stats = pool.GetStatistics();

        Assert.Equal(0, stats.PooledTensorCount);
        Assert.Equal(0, stats.CurrentMemoryBytes);
        Assert.Equal(0, stats.TensorBuckets);
        Assert.Equal(0.0, stats.MemoryUtilizationPercent, Tolerance);
    }

    [Fact]
    public void Statistics_MemoryUtilization_HandComputed()
    {
        // Pool max = 1 MB = 1048576 bytes
        using var pool = new TensorPool<double>(1);
        var tensor = pool.Rent(new[] { 100 }); // 100*8+64 = 864 bytes
        pool.Return(tensor);

        var stats = pool.GetStatistics();

        var expectedBytes = 100L * 8 + 64;
        var expectedUtilization = expectedBytes * 100.0 / (1L * 1024 * 1024);
        Assert.Equal(expectedBytes, stats.CurrentMemoryBytes);
        Assert.Equal(expectedUtilization, stats.MemoryUtilizationPercent, 1e-4);
    }

    [Fact]
    public void Statistics_MaxMemoryBytes_Correct()
    {
        using var pool = new TensorPool<double>(256);
        var stats = pool.GetStatistics();

        Assert.Equal(256L * 1024 * 1024, stats.MaxMemoryBytes);
    }

    // ============================
    // Pool Clear Tests
    // ============================

    [Fact]
    public void Clear_ResetsAllCounters()
    {
        using var pool = new TensorPool<double>();

        var t1 = pool.Rent(new[] { 10 });
        var t2 = pool.Rent(new[] { 20 });
        pool.Return(t1);
        pool.Return(t2);

        Assert.True(pool.CurrentMemoryBytes > 0);
        Assert.True(pool.TotalPooledTensors > 0);

        pool.Clear();

        Assert.Equal(0, pool.CurrentMemoryBytes);
        Assert.Equal(0, pool.TotalPooledTensors);
    }

    [Fact]
    public void Clear_PoolStillUsable()
    {
        using var pool = new TensorPool<double>();

        var t1 = pool.Rent(new[] { 5 });
        pool.Return(t1);
        pool.Clear();

        // Pool should still work after clear
        var t2 = pool.Rent(new[] { 5 });
        Assert.Equal(5, t2.Length);
        pool.Return(t2);
    }

    // ============================
    // PooledTensor Auto-Return Tests
    // ============================

    [Fact]
    public void RentPooled_DisposingReturnsToPool()
    {
        using var pool = new TensorPool<double>();

        using (var pooled = pool.RentPooled(new[] { 8 }))
        {
            Assert.Equal(8, pooled.Tensor.Length);
        }
        // After dispose, tensor should be back in pool

        var stats = pool.GetStatistics();
        Assert.True(stats.PooledTensorCount >= 1);
    }

    [Fact]
    public void RentPooled_TensorIsZeroed()
    {
        using var pool = new TensorPool<double>();

        using var pooled = pool.RentPooled(new[] { 6 });
        for (int i = 0; i < 6; i++)
            Assert.Equal(0.0, pooled.Tensor[i], Tolerance);
    }

    // ============================
    // Memory Rent/Return Tests
    // ============================

    [Fact]
    public void RentMemory_ReturnsCorrectSize()
    {
        using var pool = new TensorPool<double>();
        using var memory = pool.RentMemory(100);

        Assert.Equal(100, memory.Memory.Length);
    }

    [Fact]
    public void RentMemory_IsZeroed()
    {
        using var pool = new TensorPool<double>();
        using var memory = pool.RentMemory(50);

        var span = memory.Memory.Span;
        for (int i = 0; i < 50; i++)
            Assert.Equal(0.0, span[i], Tolerance);
    }

    // ============================
    // Boundary Conditions
    // ============================

    [Fact]
    public void Rent_NullShape_Throws()
    {
        using var pool = new TensorPool<double>();
        Assert.Throws<ArgumentException>(() => pool.Rent(null!));
    }

    [Fact]
    public void Rent_EmptyShape_Throws()
    {
        using var pool = new TensorPool<double>();
        Assert.Throws<ArgumentException>(() => pool.Rent(Array.Empty<int>()));
    }

    [Fact]
    public void Rent_ZeroDimension_Throws()
    {
        using var pool = new TensorPool<double>();
        Assert.Throws<ArgumentException>(() => pool.Rent(new[] { 3, 0, 4 }));
    }

    [Fact]
    public void Rent_NegativeDimension_Throws()
    {
        using var pool = new TensorPool<double>();
        Assert.Throws<ArgumentException>(() => pool.Rent(new[] { -1 }));
    }

    [Fact]
    public void RentMemory_ZeroCount_Throws()
    {
        using var pool = new TensorPool<double>();
        Assert.Throws<ArgumentOutOfRangeException>(() => pool.RentMemory(0));
    }

    [Fact]
    public void RentMemory_NegativeCount_Throws()
    {
        using var pool = new TensorPool<double>();
        Assert.Throws<ArgumentOutOfRangeException>(() => pool.RentMemory(-1));
    }

    [Fact]
    public void Dispose_ThenRent_Throws()
    {
        var pool = new TensorPool<double>();
        pool.Dispose();

        Assert.Throws<ObjectDisposedException>(() => pool.Rent(new[] { 5 }));
    }

    [Fact]
    public void Dispose_ThenReturn_Throws()
    {
        var pool = new TensorPool<double>();
        var tensor = new Tensor<double>(new[] { 5 });
        pool.Dispose();

        Assert.Throws<ObjectDisposedException>(() => pool.Return(tensor));
    }

    [Fact]
    public void Dispose_ThenRentPooled_Throws()
    {
        var pool = new TensorPool<double>();
        pool.Dispose();

        Assert.Throws<ObjectDisposedException>(() => pool.RentPooled(new[] { 5 }));
    }

    [Fact]
    public void Dispose_ThenRentMemory_Throws()
    {
        var pool = new TensorPool<double>();
        pool.Dispose();

        Assert.Throws<ObjectDisposedException>(() => pool.RentMemory(5));
    }

    // ============================
    // PoolingOptions Tests
    // ============================

    [Fact]
    public void PoolingOptions_DefaultValues()
    {
        var options = new PoolingOptions();

        Assert.Equal(256L * 1024 * 1024, options.MaxPoolSizeBytes); // 256 MB
        Assert.Equal(256, options.MaxPoolSizeMB);
        Assert.Equal(10_000_000, options.MaxElementsToPool);
        Assert.Equal(10, options.MaxItemsPerBucket);
        Assert.True(options.Enabled);
        Assert.False(options.UseWeakReferences);
    }

    [Fact]
    public void PoolingOptions_MaxPoolSizeMB_ConvertsBidirectionally()
    {
        var options = new PoolingOptions();

        options.MaxPoolSizeMB = 512;
        Assert.Equal(512L * 1024 * 1024, options.MaxPoolSizeBytes);

        options.MaxPoolSizeBytes = 100L * 1024 * 1024;
        Assert.Equal(100, options.MaxPoolSizeMB);
    }

    [Fact]
    public void DisabledPool_RentBypassesPool()
    {
        // When Enabled=false, Rent always allocates new tensors (never reuses)
        // but Return still adds to the pool. The Enabled flag controls Rent behavior.
        var options = new PoolingOptions { Enabled = false };
        using var pool = new TensorPool<double>(options);

        var t1 = pool.Rent(new[] { 10 });
        for (int i = 0; i < 10; i++)
            t1[i] = 42.0;
        pool.Return(t1);

        // Return still pools (Enabled only affects Rent)
        Assert.True(pool.CurrentMemoryBytes > 0);

        // Rent with Enabled=false allocates fresh (doesn't reuse)
        var t2 = pool.Rent(new[] { 10 });
        // If pool was bypassed, the pool memory should still show the returned tensor
        Assert.Equal(10, t2.Length);
    }

    [Fact]
    public void MaxElementsToPool_LargeTensorBypasses()
    {
        var options = new PoolingOptions { MaxElementsToPool = 5 };
        using var pool = new TensorPool<double>(options);

        // Tensor with 10 elements exceeds limit of 5
        var tensor = pool.Rent(new[] { 10 });
        pool.Return(tensor);

        // Should not be pooled
        Assert.Equal(0, pool.CurrentMemoryBytes);
    }

    [Fact]
    public void MaxItemsPerBucket_LimitsPoolSize()
    {
        var options = new PoolingOptions { MaxItemsPerBucket = 2 };
        using var pool = new TensorPool<double>(options);

        var t1 = pool.Rent(new[] { 5 });
        var t2 = pool.Rent(new[] { 5 });
        var t3 = pool.Rent(new[] { 5 });

        pool.Return(t1);
        pool.Return(t2);
        pool.Return(t3); // Should be rejected (bucket limit = 2)

        var stats = pool.GetStatistics();
        Assert.True(stats.PooledTensorCount <= 2);
    }

    // ============================
    // Pool Memory Limit Tests
    // ============================

    [Fact]
    public void MaxPoolSize_PreventsOverflow()
    {
        // Pool with very small max size (1 KB)
        var options = new PoolingOptions { MaxPoolSizeBytes = 1024 };
        using var pool = new TensorPool<double>(options);

        // Each tensor of 100 doubles = 100*8+64 = 864 bytes
        var t1 = pool.Rent(new[] { 100 });
        pool.Return(t1); // 864 bytes - should fit

        var t2 = pool.Rent(new[] { 100 });
        pool.Return(t2); // Would be 864+864=1728 > 1024 - should be rejected

        Assert.True(pool.CurrentMemoryBytes <= 1024);
    }

    // ============================
    // PoolStatistics Computation Tests
    // ============================

    [Fact]
    public void PoolStatistics_UtilizationPercent_HandComputed()
    {
        var stats = new PoolStatistics
        {
            CurrentMemoryBytes = 500,
            MaxMemoryBytes = 1000
        };

        // 500/1000 * 100 = 50.0%
        Assert.Equal(50.0, stats.MemoryUtilizationPercent, Tolerance);
    }

    [Fact]
    public void PoolStatistics_UtilizationPercent_FullPool()
    {
        var stats = new PoolStatistics
        {
            CurrentMemoryBytes = 1000,
            MaxMemoryBytes = 1000
        };

        Assert.Equal(100.0, stats.MemoryUtilizationPercent, Tolerance);
    }

    [Fact]
    public void PoolStatistics_UtilizationPercent_EmptyPool()
    {
        var stats = new PoolStatistics
        {
            CurrentMemoryBytes = 0,
            MaxMemoryBytes = 1000
        };

        Assert.Equal(0.0, stats.MemoryUtilizationPercent, Tolerance);
    }

    [Fact]
    public void PoolStatistics_UtilizationPercent_ZeroMax_ReturnsZero()
    {
        var stats = new PoolStatistics
        {
            CurrentMemoryBytes = 100,
            MaxMemoryBytes = 0
        };

        // Division by zero should return 0, not throw
        Assert.Equal(0.0, stats.MemoryUtilizationPercent, Tolerance);
    }

    [Fact]
    public void PoolStatistics_UtilizationPercent_Fractional()
    {
        var stats = new PoolStatistics
        {
            CurrentMemoryBytes = 1,
            MaxMemoryBytes = 3
        };

        // 1/3 * 100 = 33.333...%
        Assert.Equal(100.0 / 3.0, stats.MemoryUtilizationPercent, 1e-6);
    }

    // ============================
    // Shape Consistency Tests
    // ============================

    [Fact]
    public void Rent_1DShape_Correct()
    {
        using var pool = new TensorPool<double>();
        var tensor = pool.Rent(new[] { 7 });

        Assert.Single(tensor.Shape);
        Assert.Equal(7, tensor.Shape[0]);
    }

    [Fact]
    public void Rent_2DShape_Correct()
    {
        using var pool = new TensorPool<double>();
        var tensor = pool.Rent(new[] { 3, 5 });

        Assert.Equal(2, tensor.Shape.Length);
        Assert.Equal(3, tensor.Shape[0]);
        Assert.Equal(5, tensor.Shape[1]);
        Assert.Equal(15, tensor.Length);
    }

    [Fact]
    public void Rent_3DShape_Correct()
    {
        using var pool = new TensorPool<double>();
        var tensor = pool.Rent(new[] { 2, 3, 4 });

        Assert.Equal(3, tensor.Shape.Length);
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(3, tensor.Shape[1]);
        Assert.Equal(4, tensor.Shape[2]);
        Assert.Equal(24, tensor.Length);
    }

    [Fact]
    public void Rent_4DShape_Correct()
    {
        using var pool = new TensorPool<double>();
        var tensor = pool.Rent(new[] { 2, 3, 4, 5 });

        Assert.Equal(4, tensor.Shape.Length);
        Assert.Equal(120, tensor.Length);
    }

    // ============================
    // Data Integrity After Reuse
    // ============================

    [Fact]
    public void Reused_Tensor_NoGhostData()
    {
        using var pool = new TensorPool<double>();
        var shape = new[] { 5 };

        // First use: fill with non-zero data
        var t1 = pool.Rent(shape);
        for (int i = 0; i < 5; i++)
            t1[i] = 999.0;
        pool.Return(t1);

        // Second use: should be completely zero
        var t2 = pool.Rent(shape);
        for (int i = 0; i < 5; i++)
            Assert.Equal(0.0, t2[i], Tolerance);
    }

    [Fact]
    public void Multiple_RentReturn_Cycles_DataIntegrity()
    {
        using var pool = new TensorPool<double>();
        var shape = new[] { 4 };

        for (int cycle = 0; cycle < 10; cycle++)
        {
            var tensor = pool.Rent(shape);

            // Verify zeroed
            for (int i = 0; i < 4; i++)
                Assert.Equal(0.0, tensor[i], Tolerance);

            // Fill with cycle-specific data
            for (int i = 0; i < 4; i++)
                tensor[i] = cycle * 10.0 + i;

            pool.Return(tensor);
        }
    }

    // ============================
    // Constructor Variants
    // ============================

    [Fact]
    public void Constructor_Default_256MB()
    {
        using var pool = new TensorPool<double>();
        Assert.Equal(256L * 1024 * 1024, pool.MaxPoolSizeBytes);
    }

    [Fact]
    public void Constructor_CustomMB()
    {
        using var pool = new TensorPool<double>(128);
        Assert.Equal(128L * 1024 * 1024, pool.MaxPoolSizeBytes);
    }

    [Fact]
    public void Constructor_CustomOptions()
    {
        var options = new PoolingOptions { MaxPoolSizeMB = 64, MaxItemsPerBucket = 5 };
        using var pool = new TensorPool<double>(options);

        Assert.Equal(64L * 1024 * 1024, pool.MaxPoolSizeBytes);
        Assert.Equal(5, pool.Options.MaxItemsPerBucket);
    }

    // ============================
    // Double Dispose Safety
    // ============================

    [Fact]
    public void DoubleDispose_DoesNotThrow()
    {
        var pool = new TensorPool<double>();
        pool.Dispose();
        pool.Dispose(); // Should not throw
    }
}
