using AiDotNet.Gpu;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Gpu;

/// <summary>
/// Tests for ExecutionContext CPU/GPU placement decisions.
/// </summary>
public class ExecutionContextTests : IDisposable
{
    private readonly IlgpuBackend<float>? _backend;
    private readonly bool _gpuAvailable;

    public ExecutionContextTests()
    {
        try
        {
            _backend = new IlgpuBackend<float>();
            _backend.Initialize();
            _gpuAvailable = _backend.IsAvailable;
        }
        catch
        {
            _gpuAvailable = false;
        }
    }

    public void Dispose()
    {
        _backend?.Dispose();
    }

    [Fact]
    public void Constructor_WithoutBackend_DisablesGpu()
    {
        // Arrange & Act
        using var context = new ExecutionContext();

        // Assert
        Assert.False(context.UseGpu);
        Assert.Null(context.GpuBackend);
    }

    [Fact]
    public void Constructor_WithBackend_EnablesGpuIfAvailable()
    {
        if (!_gpuAvailable)
        {
            return; // Skip if GPU not available
        }

        // Arrange & Act
        using var context = new ExecutionContext(_backend);

        // Assert
        Assert.True(context.UseGpu);
        Assert.NotNull(context.GpuBackend);
    }

    [Fact]
    public void AutomaticPlacement_SmallTensor_ReturnsFalse()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.AutomaticPlacement,
            GpuThreshold = 100_000
        };

        var smallTensor = new Tensor<float>(new[] { 100, 100 }); // 10,000 elements

        // Act
        var shouldUseGpu = context.ShouldUseGpu(smallTensor);

        // Assert
        Assert.False(shouldUseGpu);
    }

    [Fact]
    public void AutomaticPlacement_LargeTensor_ReturnsTrue()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.AutomaticPlacement,
            GpuThreshold = 100_000
        };

        var largeTensor = new Tensor<float>(new[] { 1000, 1000 }); // 1,000,000 elements

        // Act
        var shouldUseGpu = context.ShouldUseGpu(largeTensor);

        // Assert
        Assert.True(shouldUseGpu);
    }

    [Fact]
    public void AutomaticPlacement_ExactThreshold_ReturnsTrue()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.AutomaticPlacement,
            GpuThreshold = 10_000
        };

        var tensor = new Tensor<float>(new[] { 100, 100 }); // Exactly 10,000 elements

        // Act
        var shouldUseGpu = context.ShouldUseGpu(tensor);

        // Assert
        Assert.True(shouldUseGpu); // >= threshold
    }

    [Fact]
    public void ForceGpu_AlwaysReturnsTrue()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.ForceGpu
        };

        var tinyTensor = new Tensor<float>(new[] { 2, 2 }); // Just 4 elements
        var hugeTensor = new Tensor<float>(new[] { 1000, 1000 });

        // Act & Assert
        Assert.True(context.ShouldUseGpu(tinyTensor));
        Assert.True(context.ShouldUseGpu(hugeTensor));
    }

    [Fact]
    public void ForceCpu_AlwaysReturnsFalse()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.ForceCpu
        };

        var tinyTensor = new Tensor<float>(new[] { 2, 2 });
        var hugeTensor = new Tensor<float>(new[] { 1000, 1000 });

        // Act & Assert
        Assert.False(context.ShouldUseGpu(tinyTensor));
        Assert.False(context.ShouldUseGpu(hugeTensor));
    }

    [Fact]
    public void MinimizeTransfers_ReturnsFalseByDefault()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.MinimizeTransfers
        };

        var tensor = new Tensor<float>(new[] { 1000, 1000 });

        // Act
        var shouldUseGpu = context.ShouldUseGpu(tensor);

        // Assert
        // Data is on CPU, so should stay on CPU to minimize transfers
        Assert.False(shouldUseGpu);
    }

    [Fact]
    public void CostBased_SmallTensor_ReturnsFalse()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.CostBased,
            GpuComputeSpeedup = 10.0,
            TransferBandwidthGBps = 12.0
        };

        // Very small tensor - transfer cost dominates
        var smallTensor = new Tensor<float>(new[] { 10, 10 }); // 100 elements

        // Act
        var shouldUseGpu = context.ShouldUseGpu(smallTensor);

        // Assert
        Assert.False(shouldUseGpu);
    }

    [Fact]
    public void CostBased_LargeTensor_ReturnsTrue()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.CostBased,
            GpuComputeSpeedup = 10.0,
            TransferBandwidthGBps = 12.0
        };

        // Large tensor - compute cost dominates
        var largeTensor = new Tensor<float>(new[] { 2000, 2000 }); // 4,000,000 elements

        // Act
        var shouldUseGpu = context.ShouldUseGpu(largeTensor);

        // Assert
        Assert.True(shouldUseGpu);
    }

    [Fact]
    public void Execute_UnaryOperation_WorksCorrectly()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.ForceGpu
        };

        var input = new Tensor<float>(new[] { 3, 3 });
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = i + 1.0f; // 1, 2, 3, ..., 9
        }

        // Act
        var result = context.Execute(input, gpu => _backend!.ReLU(gpu));

        // Assert
        Assert.NotNull(result);
        Assert.Equal(input.Shape, result.Shape);
        // ReLU doesn't change positive values
        for (int i = 0; i < result.Length; i++)
        {
            Assert.Equal(input[i], result[i]);
        }
    }

    [Fact]
    public void Execute_BinaryOperation_WorksCorrectly()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.ForceGpu
        };

        var tensor1 = new Tensor<float>(new[] { 3, 3 });
        var tensor2 = new Tensor<float>(new[] { 3, 3 });

        for (int i = 0; i < tensor1.Length; i++)
        {
            tensor1[i] = i + 1.0f;
            tensor2[i] = (i + 1.0f) * 2.0f;
        }

        // Act
        var result = context.Execute(tensor1, tensor2, (a, b) => _backend!.Add(a, b));

        // Assert
        Assert.NotNull(result);
        Assert.Equal(tensor1.Shape, result.Shape);
        for (int i = 0; i < result.Length; i++)
        {
            Assert.Equal(tensor1[i] + tensor2[i], result[i], precision: 4);
        }
    }

    [Fact]
    public void Execute_ThrowsWhenShouldUseCpu()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.ForceCpu
        };

        var tensor = new Tensor<float>(new[] { 3, 3 });

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            context.Execute(tensor, gpu => _backend!.ReLU(gpu)));
    }

    [Fact]
    public void Statistics_TrackGpuOperations()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.ForceGpu
        };

        var tensor = new Tensor<float>(new[] { 3, 3 });
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = i + 1.0f;
        }

        // Act
        context.Execute(tensor, gpu => _backend!.ReLU(gpu));
        context.Execute(tensor, gpu => _backend!.Sigmoid(gpu));

        // Assert
        Assert.Equal(2, context.Statistics.GpuOperations);
        Assert.Equal(0, context.Statistics.CpuOperations);
        Assert.Equal(2, context.Statistics.TotalOperations);
        Assert.Equal(100.0, context.Statistics.GpuPercentage);
    }

    [Fact]
    public void Statistics_CanBeReset()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.ForceGpu
        };

        var tensor = new Tensor<float>(new[] { 3, 3 });
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = i + 1.0f;
        }

        context.Execute(tensor, gpu => _backend!.ReLU(gpu));

        // Act
        context.ResetStatistics();

        // Assert
        Assert.Equal(0, context.Statistics.GpuOperations);
        Assert.Equal(0, context.Statistics.CpuOperations);
        Assert.Equal(0, context.Statistics.TotalOperations);
    }

    [Fact]
    public void Statistics_ToString_FormatsCorrectly()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.ForceGpu
        };

        var tensor = new Tensor<float>(new[] { 3, 3 });
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = i + 1.0f;
        }

        // Act
        context.Execute(tensor, gpu => _backend!.ReLU(gpu));
        var statsString = context.Statistics.ToString();

        // Assert
        Assert.Contains("GPU: 1", statsString);
        Assert.Contains("CPU: 0", statsString);
        Assert.Contains("Total: 1", statsString);
        Assert.Contains("GPU%: 100", statsString);
    }

    [Fact]
    public void GpuDisabled_AlwaysReturnsFalse()
    {
        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            UseGpu = false, // Explicitly disable GPU
            Strategy = ExecutionContext.PlacementStrategy.ForceGpu
        };

        var tensor = new Tensor<float>(new[] { 1000, 1000 });

        // Act
        var shouldUseGpu = context.ShouldUseGpu(tensor);

        // Assert
        Assert.False(shouldUseGpu);
    }

    [Fact]
    public void CustomThreshold_Works()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.AutomaticPlacement,
            GpuThreshold = 50_000 // Custom threshold
        };

        var mediumTensor = new Tensor<float>(new[] { 200, 200 }); // 40,000 elements
        var largeTensor = new Tensor<float>(new[] { 250, 250 }); // 62,500 elements

        // Act & Assert
        Assert.False(context.ShouldUseGpu(mediumTensor)); // Below threshold
        Assert.True(context.ShouldUseGpu(largeTensor)); // Above threshold
    }
}
