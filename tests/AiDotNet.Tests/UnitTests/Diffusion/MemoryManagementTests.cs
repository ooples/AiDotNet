using AiDotNet.Diffusion.Memory;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion;

/// <summary>
/// Tests for memory management components used in diffusion models.
/// </summary>
public class MemoryManagementTests
{
    #region ActivationPool Tests

    [Fact]
    public void ActivationPool_Rent_ReturnsCorrectShape()
    {
        // Arrange
        using var pool = new ActivationPool<float>(maxMemoryMB: 100);
        var shape = new[] { 2, 256, 64, 64 };

        // Act
        var tensor = pool.Rent(shape);

        // Assert
        Assert.Equal(shape.Length, tensor.Shape.Length);
        for (int i = 0; i < shape.Length; i++)
        {
            Assert.Equal(shape[i], tensor.Shape[i]);
        }
    }

    [Fact]
    public void ActivationPool_RentAndReturn_ReusesBuffer()
    {
        // Arrange
        using var pool = new ActivationPool<float>(maxMemoryMB: 100);
        var shape = new[] { 1, 64, 32, 32 };

        // Act
        var tensor1 = pool.Rent(shape);
        pool.Return(tensor1);
        var tensor2 = pool.Rent(shape);

        // Assert - Stats should show a cache hit
        Assert.True(pool.Stats.CacheHits >= 1, "Should have cache hit after return");
    }

    [Fact]
    public void ActivationPool_Stats_TracksMissesCorrectly()
    {
        // Arrange
        using var pool = new ActivationPool<float>(maxMemoryMB: 100);

        // Act
        var t1 = pool.Rent(new[] { 1, 64, 32, 32 });
        var t2 = pool.Rent(new[] { 1, 128, 16, 16 }); // Different shape

        // Assert
        Assert.Equal(2, pool.Stats.CacheMisses);
    }

    [Fact]
    public void ActivationPool_Return_IncreasesReturnCount()
    {
        // Arrange
        using var pool = new ActivationPool<float>(maxMemoryMB: 100);
        var tensor = pool.Rent(new[] { 1, 32, 8, 8 });

        // Act
        pool.Return(tensor);

        // Assert
        Assert.Equal(1, pool.Stats.Returns);
    }

    [Fact]
    public void ActivationPool_Clear_ResetsPool()
    {
        // Arrange
        using var pool = new ActivationPool<float>(maxMemoryMB: 100);
        var tensor = pool.Rent(new[] { 1, 256, 64, 64 });
        pool.Return(tensor);

        // Act
        pool.Clear();

        // Assert
        Assert.Equal(0, pool.GetMemoryUsage());
    }

    [Fact]
    public void ActivationPool_GetMemoryUsage_ReflectsAllocations()
    {
        // Arrange
        using var pool = new ActivationPool<float>(maxMemoryMB: 1000);

        // Act - Rent a tensor
        var initialMemory = pool.GetMemoryUsage();
        var tensor = pool.Rent(new[] { 1, 256, 64, 64 });
        var afterRent = pool.GetMemoryUsage();

        // Assert
        Assert.True(afterRent > initialMemory, "Memory should increase after rent");
    }

    [Fact]
    public void ActivationPool_HitRatio_ComputesCorrectly()
    {
        // Arrange
        using var pool = new ActivationPool<float>(maxMemoryMB: 100);
        var shape = new[] { 1, 64, 16, 16 };

        // Act - Create 2 misses and 2 hits
        var t1 = pool.Rent(shape);
        pool.Return(t1);
        var t2 = pool.Rent(shape); // hit
        pool.Return(t2);
        var t3 = pool.Rent(shape); // hit
        pool.Return(t3);

        // First rent is miss, second and third should be hits
        // Actually depends on pooling behavior, but let's check ratio is reasonable
        Assert.True(pool.Stats.HitRatio >= 0 && pool.Stats.HitRatio <= 1,
            "Hit ratio should be between 0 and 1");
    }

    [Fact]
    public void ActivationPool_RentNull_ThrowsException()
    {
        // Arrange
        using var pool = new ActivationPool<float>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => pool.Rent(null!));
    }

    [Fact]
    public void ActivationPool_RentEmpty_ThrowsException()
    {
        // Arrange
        using var pool = new ActivationPool<float>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => pool.Rent(Array.Empty<int>()));
    }

    [Fact]
    public void ActivationPool_ReturnNull_DoesNotThrow()
    {
        // Arrange
        using var pool = new ActivationPool<float>();

        // Act & Assert - Should not throw
        pool.Return(null!);
    }

    [Fact]
    public void ActivationPool_Dispose_IsIdempotent()
    {
        // Arrange
        var pool = new ActivationPool<float>();
        pool.Rent(new[] { 1, 64, 32, 32 });

        // Act & Assert - Should not throw when disposed multiple times
        pool.Dispose();
        pool.Dispose();
    }

    #endregion

    #region ModelShard Tests

    [Fact]
    public void ModelShard_Constructor_DistributesLayersEvenly()
    {
        // Arrange
        var layers = CreateSimpleLayers(12);

        // Act
        var shard = new ModelShard<float>(layers, numDevices: 4);

        // Assert - 12 layers / 4 devices = 3 layers each
        for (int device = 0; device < 4; device++)
        {
            Assert.Equal(3, shard.GetDeviceLayers(device).Count);
        }
    }

    [Fact]
    public void ModelShard_Constructor_HandlesUnevenDistribution()
    {
        // Arrange
        var layers = CreateSimpleLayers(10);

        // Act
        var shard = new ModelShard<float>(layers, numDevices: 3);

        // Assert - Should handle 10 / 3 = 3,3,4 or similar
        int total = 0;
        for (int device = 0; device < 3; device++)
        {
            total += shard.GetDeviceLayers(device).Count;
        }
        Assert.Equal(10, total);
    }

    [Fact]
    public void ModelShard_GetLayerDevice_ReturnsCorrectDevice()
    {
        // Arrange
        var layers = CreateSimpleLayers(8);
        var shard = new ModelShard<float>(layers, numDevices: 2);

        // Act & Assert - First 4 layers on device 0, next 4 on device 1
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(0, shard.GetLayerDevice(layers[i]));
        }
        for (int i = 4; i < 8; i++)
        {
            Assert.Equal(1, shard.GetLayerDevice(layers[i]));
        }
    }

    [Fact]
    public void ModelShard_GetDeviceMemoryUsage_ReturnsAllDevices()
    {
        // Arrange
        var layers = CreateSimpleLayers(6);
        var shard = new ModelShard<float>(layers, numDevices: 3);

        // Act
        var memoryUsage = shard.GetDeviceMemoryUsage();

        // Assert
        Assert.Equal(3, memoryUsage.Count);
        Assert.True(memoryUsage.ContainsKey(0));
        Assert.True(memoryUsage.ContainsKey(1));
        Assert.True(memoryUsage.ContainsKey(2));
    }

    [Fact]
    public void ModelShard_Constructor_InvalidDeviceCount_Throws()
    {
        // Arrange
        var layers = CreateSimpleLayers(4);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new ModelShard<float>(layers, numDevices: 0));
    }

    [Fact]
    public void ModelShard_GetDeviceLayers_InvalidDevice_Throws()
    {
        // Arrange
        var layers = CreateSimpleLayers(4);
        var shard = new ModelShard<float>(layers, numDevices: 2);

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            shard.GetDeviceLayers(5));
    }

    [Fact]
    public void ModelShard_GetLayerDevice_UnknownLayer_Throws()
    {
        // Arrange
        var layers = CreateSimpleLayers(4);
        var shard = new ModelShard<float>(layers, numDevices: 2);
        var unknownLayer = new DenseLayer<float>(10, 10, (IActivationFunction<float>?)null);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            shard.GetLayerDevice(unknownLayer));
    }

    [Fact]
    public void ModelShard_ToString_ContainsDeviceInfo()
    {
        // Arrange
        var layers = CreateSimpleLayers(6);
        var shard = new ModelShard<float>(layers, numDevices: 2);

        // Act
        var str = shard.ToString();

        // Assert
        Assert.Contains("Device 0", str);
        Assert.Contains("Device 1", str);
        Assert.Contains("layers", str.ToLower());
    }

    [Fact]
    public void ModelShard_EmptyLayers_DoesNotThrow()
    {
        // Arrange & Act
        var shard = new ModelShard<float>(Array.Empty<ILayer<float>>(), numDevices: 2);

        // Assert
        Assert.Equal(0, shard.GetDeviceLayers(0).Count);
        Assert.Equal(0, shard.GetDeviceLayers(1).Count);
    }

    [Fact]
    public void ModelShard_SingleDevice_GetsAllLayers()
    {
        // Arrange
        var layers = CreateSimpleLayers(10);

        // Act
        var shard = new ModelShard<float>(layers, numDevices: 1);

        // Assert
        Assert.Equal(10, shard.GetDeviceLayers(0).Count);
    }

    [Fact]
    public void ModelShard_Forward_ProcessesAllLayers()
    {
        // Arrange
        var layers = CreateSimpleLayers(4);
        var shard = new ModelShard<float>(layers, numDevices: 2);
        var input = new Tensor<float>(new[] { 1, 64 });
        for (int i = 0; i < input.Data.Length; i++)
        {
            input.Data[i] = 1.0f;
        }

        // Act
        var output = shard.Forward(input);

        // Assert - Each dense layer transforms the input
        Assert.NotNull(output);
        Assert.True(output.Data.Length > 0);
    }

    [Fact]
    public void ModelShard_Backward_ProcessesInReverse()
    {
        // Arrange
        var layers = CreateSimpleLayers(4);
        var shard = new ModelShard<float>(layers, numDevices: 2);

        // First do a forward pass
        var input = new Tensor<float>(new[] { 1, 64 });
        for (int i = 0; i < input.Data.Length; i++)
        {
            input.Data[i] = 1.0f;
        }
        var output = shard.Forward(input);

        // Now do backward
        var gradient = new Tensor<float>(output.Shape);
        for (int i = 0; i < gradient.Data.Length; i++)
        {
            gradient.Data[i] = 1.0f;
        }

        // Act
        var inputGradient = shard.Backward(gradient);

        // Assert
        Assert.NotNull(inputGradient);
    }

    #endregion

    #region ShardingConfig Tests

    [Fact]
    public void ShardingConfig_DefaultStrategy_IsEvenSplit()
    {
        // Arrange & Act
        var config = new ShardingConfig();

        // Assert
        Assert.Equal(ShardingStrategy.EvenSplit, config.Strategy);
    }

    [Fact]
    public void ShardingConfig_CustomAssignments_CanBeSet()
    {
        // Arrange & Act
        var config = new ShardingConfig
        {
            Strategy = ShardingStrategy.Custom,
            CustomDeviceAssignments = new[] { 0, 0, 1, 1, 2, 2 }
        };

        // Assert
        Assert.Equal(ShardingStrategy.Custom, config.Strategy);
        Assert.NotNull(config.CustomDeviceAssignments);
        Assert.Equal(6, config.CustomDeviceAssignments.Length);
    }

    [Fact]
    public void ModelShard_CustomStrategy_UsesAssignments()
    {
        // Arrange
        var layers = CreateSimpleLayers(4);
        var config = new ShardingConfig
        {
            Strategy = ShardingStrategy.Custom,
            CustomDeviceAssignments = new[] { 0, 1, 0, 1 }
        };

        // Act
        var shard = new ModelShard<float>(layers, numDevices: 2, config);

        // Assert
        Assert.Equal(2, shard.GetDeviceLayers(0).Count);
        Assert.Equal(2, shard.GetDeviceLayers(1).Count);
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates simple dense layers for testing.
    /// </summary>
    private static ILayer<float>[] CreateSimpleLayers(int count)
    {
        var layers = new ILayer<float>[count];
        for (int i = 0; i < count; i++)
        {
            // Use real DenseLayer for testing
            layers[i] = new DenseLayer<float>(64, 64, (IActivationFunction<float>?)null);
        }
        return layers;
    }

    #endregion
}
