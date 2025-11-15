using AiDotNet.Enums;
using AiDotNet.Gpu;
using AiDotNet.LinearAlgebra;
using AiDotNet.Extensions;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Gpu;

/// <summary>
/// Tests for GPU backend functionality.
/// </summary>
public class GpuBackendTests : IDisposable
{
    private readonly IlgpuBackend<float> _backend;
    private readonly bool _gpuAvailable;

    public GpuBackendTests()
    {
        _backend = new IlgpuBackend<float>(GpuDeviceType.Default);

        try
        {
            _backend.Initialize();
            _gpuAvailable = _backend.IsAvailable;
        }
        catch (Exception)
        {
            _gpuAvailable = false;
        }
    }

    [Fact]
    public void Backend_CanInitialize()
    {
        // Arrange & Act
        using var backend = new IlgpuBackend<float>(GpuDeviceType.Default);
        backend.Initialize();

        // Assert
        Assert.True(backend.IsAvailable);
        Assert.NotNull(backend.DeviceName);
        Assert.True(backend.TotalMemory > 0);
    }

    [Fact]
    public void Backend_ReportsDeviceType()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Assert
        Assert.True(
            _backend.DeviceType == GpuDeviceType.CUDA ||
            _backend.DeviceType == GpuDeviceType.OpenCL ||
            _backend.DeviceType == GpuDeviceType.CPU);
    }

    [Fact]
    public void Allocate_CreatesGpuTensor()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var shape = new[] { 10, 20 };

        // Act
        using var gpuTensor = _backend.Allocate(shape);

        // Assert
        Assert.NotNull(gpuTensor);
        Assert.Equal(shape, gpuTensor.Shape);
        Assert.Equal(200, gpuTensor.Length);
        Assert.Equal(TensorLocation.GPU, gpuTensor.Location);
    }

    [Fact]
    public void ToGpu_TransfersCpuTensorToGpu()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var cpuTensor = new Tensor<float>(new[] { 5, 4 });
        for (int i = 0; i < cpuTensor.Length; i++)
        {
            cpuTensor[i] = i * 2.0f;
        }

        // Act
        using var gpuTensor = _backend.ToGpu(cpuTensor);

        // Assert
        Assert.NotNull(gpuTensor);
        Assert.Equal(cpuTensor.Shape, gpuTensor.Shape);
        Assert.Equal(TensorLocation.GPU, gpuTensor.Location);
    }

    [Fact]
    public void ToCpu_TransfersGpuTensorToCpu()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var originalTensor = new Tensor<float>(new[] { 3, 3 });
        for (int i = 0; i < originalTensor.Length; i++)
        {
            originalTensor[i] = i + 1.0f;
        }

        // Act
        using var gpuTensor = _backend.ToGpu(originalTensor);
        var resultTensor = _backend.ToCpu(gpuTensor);

        // Assert
        Assert.Equal(originalTensor.Shape, resultTensor.Shape);

        for (int i = 0; i < originalTensor.Length; i++)
        {
            Assert.Equal(originalTensor[i], resultTensor[i], precision: 5);
        }
    }

    [Fact]
    public void Add_PerformsElementWiseAddition()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var a = new Tensor<float>(new[] { 4 });
        var b = new Tensor<float>(new[] { 4 });

        for (int i = 0; i < 4; i++)
        {
            a[new[] { i }] = i + 1.0f;  // [1, 2, 3, 4]
            b[new[] { i }] = i * 2.0f;  // [0, 2, 4, 6]
        }

        // Act
        using var gpuA = _backend.ToGpu(a);
        using var gpuB = _backend.ToGpu(b);
        using var gpuResult = _backend.Add(gpuA, gpuB);
        var result = _backend.ToCpu(gpuResult);

        // Assert
        Assert.Equal(1.0f, result[new[] { 0 }], precision: 5);  // 1 + 0
        Assert.Equal(4.0f, result[new[] { 1 }], precision: 5);  // 2 + 2
        Assert.Equal(7.0f, result[new[] { 2 }], precision: 5);  // 3 + 4
        Assert.Equal(10.0f, result[new[] { 3 }], precision: 5); // 4 + 6
    }

    [Fact]
    public void Multiply_PerformsElementWiseMultiplication()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var a = new Tensor<float>(new[] { 3 });
        var b = new Tensor<float>(new[] { 3 });

        for (int i = 0; i < 3; i++)
        {
            a[new[] { i }] = i + 1.0f;  // [1, 2, 3]
            b[new[] { i }] = 2.0f;      // [2, 2, 2]
        }

        // Act
        using var gpuA = _backend.ToGpu(a);
        using var gpuB = _backend.ToGpu(b);
        using var gpuResult = _backend.Multiply(gpuA, gpuB);
        var result = _backend.ToCpu(gpuResult);

        // Assert
        Assert.Equal(2.0f, result[new[] { 0 }], precision: 5);  // 1 * 2
        Assert.Equal(4.0f, result[new[] { 1 }], precision: 5);  // 2 * 2
        Assert.Equal(6.0f, result[new[] { 2 }], precision: 5);  // 3 * 2
    }

    [Fact]
    public void ReLU_AppliesCorrectly()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var input = new Tensor<float>(new[] { 5 });
        input[new[] { 0 }] = -2.0f;
        input[new[] { 1 }] = -1.0f;
        input[new[] { 2 }] = 0.0f;
        input[new[] { 3 }] = 1.0f;
        input[new[] { 4 }] = 2.0f;

        // Act
        using var gpuInput = _backend.ToGpu(input);
        using var gpuResult = _backend.ReLU(gpuInput);
        var result = _backend.ToCpu(gpuResult);

        // Assert
        Assert.Equal(0.0f, result[new[] { 0 }], precision: 5);  // max(-2, 0) = 0
        Assert.Equal(0.0f, result[new[] { 1 }], precision: 5);  // max(-1, 0) = 0
        Assert.Equal(0.0f, result[new[] { 2 }], precision: 5);  // max(0, 0) = 0
        Assert.Equal(1.0f, result[new[] { 3 }], precision: 5);  // max(1, 0) = 1
        Assert.Equal(2.0f, result[new[] { 4 }], precision: 5);  // max(2, 0) = 2
    }

    [Fact]
    public void TensorExtension_ToGpu_Works()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var tensor = new Tensor<float>(new[] { 3, 3 });
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = i;
        }

        // Act
        using var gpuTensor = tensor.ToGpu(_backend);

        // Assert
        Assert.NotNull(gpuTensor);
        Assert.Equal(TensorLocation.GPU, gpuTensor.Location);
        Assert.Equal(tensor.Shape, gpuTensor.Shape);
    }

    [Fact]
    public void TensorExtension_WithGpu_ExecutesOperation()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var tensor = new Tensor<float>(new[] { 4 });
        tensor[new[] { 0 }] = -1.0f;
        tensor[new[] { 1 }] = 0.0f;
        tensor[new[] { 2 }] = 1.0f;
        tensor[new[] { 3 }] = 2.0f;

        // Act
        var result = tensor.WithGpu(_backend, gpu => _backend.ReLU(gpu));

        // Assert
        Assert.Equal(0.0f, result[new[] { 0 }], precision: 5);
        Assert.Equal(0.0f, result[new[] { 1 }], precision: 5);
        Assert.Equal(1.0f, result[new[] { 2 }], precision: 5);
        Assert.Equal(2.0f, result[new[] { 3 }], precision: 5);
    }

    [Fact]
    public void MatrixExtension_ToGpu_Works()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var matrix = new Matrix<float>(3, 4);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                matrix[i, j] = i * 4 + j;
            }
        }

        // Act
        using var gpuTensor = matrix.ToGpu(_backend);

        // Assert
        Assert.NotNull(gpuTensor);
        Assert.Equal(2, gpuTensor.Rank);
        Assert.Equal(3, gpuTensor.Shape[0]);
        Assert.Equal(4, gpuTensor.Shape[1]);
    }

    [Fact]
    public void VectorExtension_ToGpu_Works()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var vector = new Vector<float>(5);
        for (int i = 0; i < 5; i++)
        {
            vector[i] = i * 2.0f;
        }

        // Act
        using var gpuTensor = vector.ToGpu(_backend);

        // Assert
        Assert.NotNull(gpuTensor);
        Assert.Equal(1, gpuTensor.Rank);
        Assert.Equal(5, gpuTensor.Shape[0]);
    }

    public void Dispose()
    {
        _backend?.Dispose();
    }
}
