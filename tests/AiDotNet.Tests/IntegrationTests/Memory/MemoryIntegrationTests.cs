using AiDotNet.Memory;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Memory;

/// <summary>
/// Integration tests for memory management classes: TensorPool, InferenceContext, PoolingOptions, PoolStatistics.
/// </summary>
public class MemoryIntegrationTests
{
    private const double Tolerance = 1e-6;
    #region TensorPool Tests

    [Fact]
    public void TensorPool_Rent_ReturnsTensorOfRequestedShape()
    {
        using var pool = new TensorPool<double>();
        var tensor = pool.Rent(new[] { 3, 4 });
        Assert.NotNull(tensor);
        Assert.Equal(2, tensor.Shape.Length);
        Assert.Equal(3, tensor.Shape[0]);
        Assert.Equal(4, tensor.Shape[1]);
        pool.Return(tensor);
    }

    [Fact]
    public void TensorPool_ReturnAndReRent_ReusesTensor()
    {
        using var pool = new TensorPool<double>();
        var tensor1 = pool.Rent(new[] { 5 });
        pool.Return(tensor1);

        var stats = pool.GetStatistics();
        Assert.True(stats.PooledTensorCount >= 1, "Pool should contain the returned tensor");

        var tensor2 = pool.Rent(new[] { 5 });
        Assert.NotNull(tensor2);
        Assert.Equal(5, tensor2.Shape[0]);

        // Verify reuse: tensor2 should be the same instance as tensor1
        Assert.Same(tensor1, tensor2);
        pool.Return(tensor2);
    }

    [Fact]
    public void TensorPool_DifferentShapes_TrackedSeparately()
    {
        using var pool = new TensorPool<double>();
        var t1 = pool.Rent(new[] { 3 });
        var t2 = pool.Rent(new[] { 5 });
        pool.Return(t1);
        pool.Return(t2);

        var stats = pool.GetStatistics();
        Assert.True(stats.TensorBuckets >= 2);
    }

    [Fact]
    public void TensorPool_GetStatistics_ReturnsValidData()
    {
        using var pool = new TensorPool<double>();
        var tensor = pool.Rent(new[] { 10, 10 });
        pool.Return(tensor);

        var stats = pool.GetStatistics();
        Assert.True(stats.PooledTensorCount >= 0);
        Assert.True(stats.CurrentMemoryBytes >= 0);
        Assert.True(stats.MaxMemoryBytes > 0);
        Assert.True(stats.MemoryUtilizationPercent >= 0);
    }

    [Fact]
    public void TensorPool_Dispose_ClearsPool()
    {
        var pool = new TensorPool<double>();
        var tensor = pool.Rent(new[] { 5 });
        pool.Return(tensor);
        pool.Dispose();

        Assert.Throws<ObjectDisposedException>(() => pool.Rent(new[] { 5 }));
    }

    [Fact]
    public void TensorPool_Clear_RemovesAllTensors()
    {
        using var pool = new TensorPool<double>();
        var t1 = pool.Rent(new[] { 10 });
        var t2 = pool.Rent(new[] { 20 });
        pool.Return(t1);
        pool.Return(t2);

        pool.Clear();
        var stats = pool.GetStatistics();
        Assert.Equal(0, stats.PooledTensorCount);
    }

    [Fact]
    public void TensorPool_RentPooled_AutoReturnsOnDispose()
    {
        using var pool = new TensorPool<double>();

        using (var pooled = pool.RentPooled(new[] { 8 }))
        {
            Assert.NotNull(pooled.Tensor);
            Assert.Equal(8, pooled.Tensor.Shape[0]);
        }

        // After disposing the PooledTensor, it should be back in the pool
        var stats = pool.GetStatistics();
        Assert.True(stats.PooledTensorCount >= 1,
            $"After auto-return, pool should have at least 1 tensor, got {stats.PooledTensorCount}");
    }

    [Fact]
    public void TensorPool_InvalidShape_ThrowsArgumentException()
    {
        using var pool = new TensorPool<double>();
        Assert.Throws<ArgumentException>(() => pool.Rent(Array.Empty<int>()));
        Assert.Throws<ArgumentException>(() => pool.Rent(new[] { -1 }));
        Assert.Throws<ArgumentException>(() => pool.Rent(new[] { 0 }));
    }

    [Fact]
    public void TensorPool_ConstructorWithSize_SetsMaxPoolSize()
    {
        using var pool = new TensorPool<double>(128);
        Assert.Equal(128L * 1024 * 1024, pool.MaxPoolSizeBytes);
    }

    #endregion

    #region InferenceContext Tests

    [Fact]
    public void InferenceContext_CreateDispose_Lifecycle()
    {
        using var context = new InferenceContext<double>();
        Assert.Equal(0, context.RentedTensorCount);
    }

    [Fact]
    public void InferenceContext_RentTensor_TracksRentedCount()
    {
        using var context = new InferenceContext<double>();
        var t1 = context.Rent(new[] { 5 });
        var t2 = context.Rent(new[] { 10 });
        Assert.Equal(2, context.RentedTensorCount);
    }

    [Fact]
    public void InferenceContext_Release_DecrementsCount()
    {
        using var context = new InferenceContext<double>();
        var tensor = context.Rent(new[] { 5 });
        Assert.Equal(1, context.RentedTensorCount);

        context.Release(tensor);
        Assert.Equal(0, context.RentedTensorCount);
    }

    [Fact]
    public void InferenceContext_Dispose_CleansUpAllTensors()
    {
        var context = new InferenceContext<double>();
        var t1 = context.Rent1D(10);
        var t2 = context.Rent2D(3, 4);
        var t3 = context.Rent3D(2, 3, 4);
        Assert.Equal(3, context.RentedTensorCount);

        context.Dispose();
        // After dispose, attempting to rent throws
        Assert.Throws<ObjectDisposedException>(() => context.Rent(new[] { 5 }));
    }

    [Fact]
    public void InferenceContext_RentLike_MatchesTemplateShape()
    {
        using var context = new InferenceContext<double>();
        var template = context.Rent(new[] { 3, 4, 5 });
        var similar = context.RentLike(template);
        Assert.Equal(template.Shape, similar.Shape);
    }

    [Fact]
    public void InferenceContext_Rent4D_CorrectShape()
    {
        using var context = new InferenceContext<double>();
        var tensor = context.Rent4D(2, 3, 8, 8);
        Assert.Equal(4, tensor.Shape.Length);
        Assert.Equal(2, tensor.Shape[0]);
        Assert.Equal(3, tensor.Shape[1]);
        Assert.Equal(8, tensor.Shape[2]);
        Assert.Equal(8, tensor.Shape[3]);
    }

    [Fact]
    public void InferenceContext_WithExistingPool_SharesPool()
    {
        using var pool = new TensorPool<double>();
        using var context = new InferenceContext<double>(pool);
        Assert.Same(pool, context.Pool);
    }

    [Fact]
    public void InferenceScope_NestedContexts_RestoresPrevious()
    {
        using var context1 = new InferenceContext<double>();
        using var context2 = new InferenceContext<double>();

        using (var scope1 = InferenceScope<double>.Begin(context1))
        {
            Assert.Same(context1, InferenceScope<double>.Current);

            using (var scope2 = InferenceScope<double>.Begin(context2))
            {
                Assert.Same(context2, InferenceScope<double>.Current);
            }

            Assert.Same(context1, InferenceScope<double>.Current);
        }

        Assert.Null(InferenceScope<double>.Current);
    }

    [Fact]
    public void InferenceScope_RentOrCreate_UsesPoolWhenActive()
    {
        using var context = new InferenceContext<double>();
        using var scope = InferenceScope<double>.Begin(context);

        Assert.True(InferenceScope<double>.IsActive);
        var tensor = InferenceScope<double>.RentOrCreate(new[] { 5 });
        Assert.NotNull(tensor);
        Assert.Equal(5, tensor.Shape[0]);
    }

    [Fact]
    public void InferenceScope_RentOrCreate_AllocatesWhenInactive()
    {
        // Ensure no scope is active
        InferenceScope<double>.Current = null;
        Assert.False(InferenceScope<double>.IsActive);

        var tensor = InferenceScope<double>.RentOrCreate(new[] { 5 });
        Assert.NotNull(tensor);
        Assert.Equal(5, tensor.Shape[0]);
    }

    #endregion

    #region PoolingOptions Tests

    [Fact]
    public void PoolingOptions_DefaultValues()
    {
        var opts = new PoolingOptions();
        Assert.Equal(256, opts.MaxPoolSizeMB);
        Assert.Equal(256L * 1024 * 1024, opts.MaxPoolSizeBytes);
        Assert.Equal(10, opts.MaxItemsPerBucket);
        Assert.Equal(10_000_000, opts.MaxElementsToPool);
        Assert.True(opts.Enabled);
        Assert.False(opts.UseWeakReferences);
    }

    [Fact]
    public void PoolingOptions_SetMaxPoolSizeMB_ConvertsToBytes()
    {
        var opts = new PoolingOptions { MaxPoolSizeMB = 512 };
        Assert.Equal(512L * 1024 * 1024, opts.MaxPoolSizeBytes);
    }

    [Fact]
    public void PoolingOptions_CustomConfiguration()
    {
        var opts = new PoolingOptions
        {
            MaxPoolSizeMB = 128,
            MaxItemsPerBucket = 20,
            MaxElementsToPool = 5_000_000,
            Enabled = false,
            UseWeakReferences = true
        };

        Assert.Equal(128, opts.MaxPoolSizeMB);
        Assert.Equal(20, opts.MaxItemsPerBucket);
        Assert.Equal(5_000_000, opts.MaxElementsToPool);
        Assert.False(opts.Enabled);
        Assert.True(opts.UseWeakReferences);
    }

    #endregion

    #region PoolStatistics Tests

    [Fact]
    public void PoolStatistics_Properties_SetAndGet()
    {
        var stats = new PoolStatistics
        {
            PooledTensorCount = 10,
            CurrentMemoryBytes = 1024,
            MaxMemoryBytes = 4096,
            TensorBuckets = 3
        };

        Assert.Equal(10, stats.PooledTensorCount);
        Assert.Equal(1024, stats.CurrentMemoryBytes);
        Assert.Equal(4096, stats.MaxMemoryBytes);
        Assert.Equal(3, stats.TensorBuckets);
    }

    [Fact]
    public void PoolStatistics_MemoryUtilizationPercent_Calculation()
    {
        var stats = new PoolStatistics
        {
            CurrentMemoryBytes = 128,
            MaxMemoryBytes = 256
        };
        Assert.Equal(50.0, stats.MemoryUtilizationPercent, Tolerance);
    }

    [Fact]
    public void PoolStatistics_MemoryUtilizationPercent_ZeroMax_ReturnsZero()
    {
        var stats = new PoolStatistics
        {
            CurrentMemoryBytes = 100,
            MaxMemoryBytes = 0
        };
        Assert.Equal(0.0, stats.MemoryUtilizationPercent, Tolerance);
    }

    #endregion
}
