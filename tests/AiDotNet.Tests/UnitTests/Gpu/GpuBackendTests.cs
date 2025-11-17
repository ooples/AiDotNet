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

    [Fact]
    public void MatMul_Small_PerformsCorrectly()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange: 2x3 * 3x2 = 2x2
        var a = new Tensor<float>(new[] { 2, 3 });
        // A = [[1, 2, 3],
        //      [4, 5, 6]]
        a[new[] { 0, 0 }] = 1; a[new[] { 0, 1 }] = 2; a[new[] { 0, 2 }] = 3;
        a[new[] { 1, 0 }] = 4; a[new[] { 1, 1 }] = 5; a[new[] { 1, 2 }] = 6;

        var b = new Tensor<float>(new[] { 3, 2 });
        // B = [[7, 8],
        //      [9, 10],
        //      [11, 12]]
        b[new[] { 0, 0 }] = 7;  b[new[] { 0, 1 }] = 8;
        b[new[] { 1, 0 }] = 9;  b[new[] { 1, 1 }] = 10;
        b[new[] { 2, 0 }] = 11; b[new[] { 2, 1 }] = 12;

        // Expected result:
        // C = [[1*7+2*9+3*11,  1*8+2*10+3*12],
        //      [4*7+5*9+6*11,  4*8+5*10+6*12]]
        //   = [[58, 64],
        //      [139, 154]]

        // Act
        using var gpuA = _backend.ToGpu(a);
        using var gpuB = _backend.ToGpu(b);
        using var gpuResult = _backend.MatMul(gpuA, gpuB);
        var result = _backend.ToCpu(gpuResult);

        // Assert
        Assert.Equal(new[] { 2, 2 }, result.Shape);
        Assert.Equal(58f, result[new[] { 0, 0 }], precision: 4);
        Assert.Equal(64f, result[new[] { 0, 1 }], precision: 4);
        Assert.Equal(139f, result[new[] { 1, 0 }], precision: 4);
        Assert.Equal(154f, result[new[] { 1, 1 }], precision: 4);
    }

    [Fact]
    public void MatMul_Large_UsesOptimizedKernel()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange: Large matrices to trigger tiled kernel
        var size = 256;
        var a = new Tensor<float>(new[] { size, size });
        var b = new Tensor<float>(new[] { size, size });

        // Fill with simple values for verification
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                a[new[] { i, j }] = 1.0f;
                b[new[] { i, j }] = 1.0f;
            }
        }

        // Expected: Each element should be size (sum of 1.0 * 1.0, size times)

        // Act
        using var gpuA = _backend.ToGpu(a);
        using var gpuB = _backend.ToGpu(b);
        using var gpuResult = _backend.MatMul(gpuA, gpuB);
        var result = _backend.ToCpu(gpuResult);

        // Assert
        Assert.Equal(new[] { size, size }, result.Shape);

        // Check a few elements
        Assert.Equal((float)size, result[new[] { 0, 0 }], precision: 2);
        Assert.Equal((float)size, result[new[] { size / 2, size / 2 }], precision: 2);
        Assert.Equal((float)size, result[new[] { size - 1, size - 1 }], precision: 2);
    }

    [Fact]
    public void MatMul_IdentityMatrix_ReturnsOriginal()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange: Multiply by identity matrix should return original
        var a = new Tensor<float>(new[] { 3, 3 });
        var identity = new Tensor<float>(new[] { 3, 3 });

        // A = [[1, 2, 3],
        //      [4, 5, 6],
        //      [7, 8, 9]]
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                a[new[] { i, j }] = i * 3 + j + 1;
                identity[new[] { i, j }] = (i == j) ? 1.0f : 0.0f;
            }
        }

        // Act
        using var gpuA = _backend.ToGpu(a);
        using var gpuId = _backend.ToGpu(identity);
        using var gpuResult = _backend.MatMul(gpuA, gpuId);
        var result = _backend.ToCpu(gpuResult);

        // Assert: Result should equal A
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(a[new[] { i, j }], result[new[] { i, j }], precision: 5);
            }
        }
    }

    [Fact]
    public void Transpose_WorksCorrectly()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var input = new Tensor<float>(new[] { 2, 3 });
        // Input = [[1, 2, 3],
        //          [4, 5, 6]]
        input[new[] { 0, 0 }] = 1; input[new[] { 0, 1 }] = 2; input[new[] { 0, 2 }] = 3;
        input[new[] { 1, 0 }] = 4; input[new[] { 1, 1 }] = 5; input[new[] { 1, 2 }] = 6;

        // Expected transpose = [[1, 4],
        //                       [2, 5],
        //                       [3, 6]]

        // Act
        using var gpuInput = _backend.ToGpu(input);
        using var gpuResult = _backend.Transpose(gpuInput);
        var result = _backend.ToCpu(gpuResult);

        // Assert
        Assert.Equal(new[] { 3, 2 }, result.Shape);
        Assert.Equal(1f, result[new[] { 0, 0 }], precision: 5);
        Assert.Equal(4f, result[new[] { 0, 1 }], precision: 5);
        Assert.Equal(2f, result[new[] { 1, 0 }], precision: 5);
        Assert.Equal(5f, result[new[] { 1, 1 }], precision: 5);
        Assert.Equal(3f, result[new[] { 2, 0 }], precision: 5);
        Assert.Equal(6f, result[new[] { 2, 1 }], precision: 5);
    }

    [Fact]
    public void Sum_ComputesCorrectly()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var input = new Tensor<float>(new[] { 4 });
        input[new[] { 0 }] = 1.0f;
        input[new[] { 1 }] = 2.0f;
        input[new[] { 2 }] = 3.0f;
        input[new[] { 3 }] = 4.0f;
        // Expected sum: 1 + 2 + 3 + 4 = 10

        // Act
        using var gpuInput = _backend.ToGpu(input);
        using var gpuResult = _backend.Sum(gpuInput);
        var result = _backend.ToCpu(gpuResult);

        // Assert
        Assert.Equal(new[] { 1 }, result.Shape);
        Assert.Equal(10.0f, result[new[] { 0 }], precision: 5);
    }

    [Fact]
    public void Mean_ComputesCorrectly()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var input = new Tensor<float>(new[] { 5 });
        input[new[] { 0 }] = 2.0f;
        input[new[] { 1 }] = 4.0f;
        input[new[] { 2 }] = 6.0f;
        input[new[] { 3 }] = 8.0f;
        input[new[] { 4 }] = 10.0f;
        // Expected mean: (2+4+6+8+10) / 5 = 30 / 5 = 6

        // Act
        using var gpuInput = _backend.ToGpu(input);
        using var gpuResult = _backend.Mean(gpuInput);
        var result = _backend.ToCpu(gpuResult);

        // Assert
        Assert.Equal(new[] { 1 }, result.Shape);
        Assert.Equal(6.0f, result[new[] { 0 }], precision: 5);
    }

    [Fact]
    public void MatMul_WithMatrix_Extension_Works()
    {
        // Skip if GPU not available
        if (!_gpuAvailable) return;

        // Arrange
        var matrixA = new Matrix<float>(2, 2);
        matrixA[0, 0] = 1; matrixA[0, 1] = 2;
        matrixA[1, 0] = 3; matrixA[1, 1] = 4;

        var matrixB = new Matrix<float>(2, 2);
        matrixB[0, 0] = 5; matrixB[0, 1] = 6;
        matrixB[1, 0] = 7; matrixB[1, 1] = 8;

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

        // Act
        using var gpuA = matrixA.ToGpu(_backend);
        using var gpuB = matrixB.ToGpu(_backend);
        using var gpuResult = _backend.MatMul(gpuA, gpuB);
        var resultMatrix = gpuResult.ToMatrix(_backend);

        // Assert
        Assert.Equal(19f, resultMatrix[0, 0], precision: 4);
        Assert.Equal(22f, resultMatrix[0, 1], precision: 4);
        Assert.Equal(43f, resultMatrix[1, 0], precision: 4);
        Assert.Equal(50f, resultMatrix[1, 1], precision: 4);
    }

    public void Dispose()
    {
        _backend?.Dispose();
    }
}
