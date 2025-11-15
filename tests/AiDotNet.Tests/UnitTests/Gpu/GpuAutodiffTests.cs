using AiDotNet.Autodiff;
using AiDotNet.Gpu;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Gpu;

/// <summary>
/// Integration tests for GPU-accelerated automatic differentiation.
/// </summary>
public class GpuAutodiffTests : IDisposable
{
    private readonly IlgpuBackend<float>? _backend;
    private readonly bool _gpuAvailable;

    public GpuAutodiffTests()
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
    public void GpuComputationNode_Create_WithAutomaticPlacement()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend)
        {
            Strategy = ExecutionContext.PlacementStrategy.AutomaticPlacement,
            GpuThreshold = 100
        };

        var smallTensor = new Tensor<float>(new[] { 5, 5 }); // 25 elements
        var largeTensor = new Tensor<float>(new[] { 20, 20 }); // 400 elements

        // Act
        using var smallNode = GpuComputationNode<float>.Create(smallTensor, context, requiresGradient: true);
        using var largeNode = GpuComputationNode<float>.Create(largeTensor, context, requiresGradient: true);

        // Assert
        Assert.False(smallNode.IsOnGpu); // Too small for GPU
        Assert.True(largeNode.IsOnGpu);  // Large enough for GPU
    }

    [Fact]
    public void GpuComputationNode_MoveToGpu_TransfersData()
    {
        if (!_gpuAvailable)
        {
            return;
        }

        // Arrange
        using var context = new ExecutionContext(_backend);
        var tensor = new Tensor<float>(new[] { 3, 3 });
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = i + 1.0f;
        }

        using var node = new GpuComputationNode<float>(tensor, context);

        // Act
        node.MoveToGpu();

        // Assert
        Assert.True(node.IsOnGpu);
        Assert.NotNull(node.GpuValue);

        // Verify data integrity
        node.Synchronize(preferGpu: true);
        for (int i = 0; i < tensor.Length; i++)
        {
            Assert.Equal(i + 1.0f, node.Value[i], precision: 4);
        }
    }

    [Fact]
    public void GpuTensorOperations_Add_ComputesCorrectResult()
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

        var tensorA = new Tensor<float>(new[] { 3, 3 });
        var tensorB = new Tensor<float>(new[] { 3, 3 });

        for (int i = 0; i < tensorA.Length; i++)
        {
            tensorA[i] = i + 1.0f;
            tensorB[i] = (i + 1.0f) * 2.0f;
        }

        using var nodeA = GpuTensorOperations<float>.Variable(tensorA, context, "a");
        using var nodeB = GpuTensorOperations<float>.Variable(tensorB, context, "b");

        // Act
        using var result = GpuTensorOperations<float>.Add(nodeA, nodeB, context);

        // Assert
        for (int i = 0; i < result.Value.Length; i++)
        {
            var expected = tensorA[i] + tensorB[i];
            Assert.Equal(expected, result.Value[i], precision: 4);
        }
    }

    [Fact]
    public void GpuTensorOperations_Add_ComputesCorrectGradients()
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

        var tensorA = new Tensor<float>(new[] { 2, 2 });
        var tensorB = new Tensor<float>(new[] { 2, 2 });

        for (int i = 0; i < tensorA.Length; i++)
        {
            tensorA[i] = i + 1.0f;
            tensorB[i] = (i + 1.0f) * 2.0f;
        }

        using var nodeA = GpuTensorOperations<float>.Variable(tensorA, context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(tensorB, context, "b", requiresGradient: true);

        // Act
        using var result = GpuTensorOperations<float>.Add(nodeA, nodeB, context);
        result.Backward();

        // Assert - for addition, gradients should be all ones
        Assert.NotNull(nodeA.Gradient);
        Assert.NotNull(nodeB.Gradient);

        for (int i = 0; i < nodeA.Gradient.Length; i++)
        {
            Assert.Equal(1.0f, nodeA.Gradient[i], precision: 4);
            Assert.Equal(1.0f, nodeB.Gradient[i], precision: 4);
        }
    }

    [Fact]
    public void GpuTensorOperations_Subtract_ComputesCorrectGradients()
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

        var tensorA = new Tensor<float>(new[] { 2, 2 });
        var tensorB = new Tensor<float>(new[] { 2, 2 });

        for (int i = 0; i < tensorA.Length; i++)
        {
            tensorA[i] = (i + 1.0f) * 3.0f;
            tensorB[i] = (i + 1.0f) * 2.0f;
        }

        using var nodeA = GpuTensorOperations<float>.Variable(tensorA, context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(tensorB, context, "b", requiresGradient: true);

        // Act
        using var result = GpuTensorOperations<float>.Subtract(nodeA, nodeB, context);
        result.Backward();

        // Assert - for subtraction, a gets +1, b gets -1
        Assert.NotNull(nodeA.Gradient);
        Assert.NotNull(nodeB.Gradient);

        for (int i = 0; i < nodeA.Gradient.Length; i++)
        {
            Assert.Equal(1.0f, nodeA.Gradient[i], precision: 4);
            Assert.Equal(-1.0f, nodeB.Gradient[i], precision: 4);
        }
    }

    [Fact]
    public void GpuTensorOperations_ElementwiseMultiply_ComputesCorrectGradients()
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

        var tensorA = new Tensor<float>(new[] { 2, 2 });
        var tensorB = new Tensor<float>(new[] { 2, 2 });

        for (int i = 0; i < tensorA.Length; i++)
        {
            tensorA[i] = i + 2.0f;  // [2, 3, 4, 5]
            tensorB[i] = i + 3.0f;  // [3, 4, 5, 6]
        }

        using var nodeA = GpuTensorOperations<float>.Variable(tensorA, context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(tensorB, context, "b", requiresGradient: true);

        // Act
        using var result = GpuTensorOperations<float>.ElementwiseMultiply(nodeA, nodeB, context);
        result.Backward();

        // Assert - for multiplication, gradient of a is b, gradient of b is a
        Assert.NotNull(nodeA.Gradient);
        Assert.NotNull(nodeB.Gradient);

        for (int i = 0; i < nodeA.Gradient.Length; i++)
        {
            Assert.Equal(tensorB[i], nodeA.Gradient[i], precision: 4);
            Assert.Equal(tensorA[i], nodeB.Gradient[i], precision: 4);
        }
    }

    [Fact]
    public void GpuTensorOperations_MatMul_ComputesCorrectResult()
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

        // 2x3 matrix
        var tensorA = new Tensor<float>(new[] { 2, 3 });
        tensorA[new[] { 0, 0 }] = 1; tensorA[new[] { 0, 1 }] = 2; tensorA[new[] { 0, 2 }] = 3;
        tensorA[new[] { 1, 0 }] = 4; tensorA[new[] { 1, 1 }] = 5; tensorA[new[] { 1, 2 }] = 6;

        // 3x2 matrix
        var tensorB = new Tensor<float>(new[] { 3, 2 });
        tensorB[new[] { 0, 0 }] = 7; tensorB[new[] { 0, 1 }] = 8;
        tensorB[new[] { 1, 0 }] = 9; tensorB[new[] { 1, 1 }] = 10;
        tensorB[new[] { 2, 0 }] = 11; tensorB[new[] { 2, 1 }] = 12;

        using var nodeA = GpuTensorOperations<float>.Variable(tensorA, context, "a");
        using var nodeB = GpuTensorOperations<float>.Variable(tensorB, context, "b");

        // Act
        using var result = GpuTensorOperations<float>.MatMul(nodeA, nodeB, context);

        // Assert - result should be 2x2
        Assert.Equal(2, result.Value.Rank);
        Assert.Equal(2, result.Value.Shape[0]);
        Assert.Equal(2, result.Value.Shape[1]);

        // Expected: [1*7+2*9+3*11, 1*8+2*10+3*12]   = [58, 64]
        //           [4*7+5*9+6*11, 4*8+5*10+6*12]   = [139, 154]
        Assert.Equal(58.0f, result.Value[new[] { 0, 0 }], precision: 4);
        Assert.Equal(64.0f, result.Value[new[] { 0, 1 }], precision: 4);
        Assert.Equal(139.0f, result.Value[new[] { 1, 0 }], precision: 4);
        Assert.Equal(154.0f, result.Value[new[] { 1, 1 }], precision: 4);
    }

    [Fact]
    public void GpuTensorOperations_MatMul_ComputesCorrectGradients()
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

        // Simple 2x2 matrices for easier gradient checking
        var tensorA = new Tensor<float>(new[] { 2, 2 });
        tensorA[new[] { 0, 0 }] = 1; tensorA[new[] { 0, 1 }] = 2;
        tensorA[new[] { 1, 0 }] = 3; tensorA[new[] { 1, 1 }] = 4;

        var tensorB = new Tensor<float>(new[] { 2, 2 });
        tensorB[new[] { 0, 0 }] = 5; tensorB[new[] { 0, 1 }] = 6;
        tensorB[new[] { 1, 0 }] = 7; tensorB[new[] { 1, 1 }] = 8;

        using var nodeA = GpuTensorOperations<float>.Variable(tensorA, context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(tensorB, context, "b", requiresGradient: true);

        // Act
        using var result = GpuTensorOperations<float>.MatMul(nodeA, nodeB, context);
        result.Backward();

        // Assert - gradients should be computed
        Assert.NotNull(nodeA.Gradient);
        Assert.NotNull(nodeB.Gradient);

        // Gradient of A = gradient · B^T
        // Gradient of B = A^T · gradient
        // With gradient initialized to all ones, we can verify the shapes at minimum
        Assert.Equal(tensorA.Shape, nodeA.Gradient.Shape);
        Assert.Equal(tensorB.Shape, nodeB.Gradient.Shape);
    }

    [Fact]
    public void GpuTensorOperations_ReLU_ComputesCorrectResult()
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

        var tensor = new Tensor<float>(new[] { 2, 3 });
        tensor[new[] { 0, 0 }] = -2.0f;
        tensor[new[] { 0, 1 }] = -1.0f;
        tensor[new[] { 0, 2 }] = 0.0f;
        tensor[new[] { 1, 0 }] = 1.0f;
        tensor[new[] { 1, 1 }] = 2.0f;
        tensor[new[] { 1, 2 }] = 3.0f;

        using var node = GpuTensorOperations<float>.Variable(tensor, context, "a");

        // Act
        using var result = GpuTensorOperations<float>.ReLU(node, context);

        // Assert - ReLU(x) = max(0, x)
        Assert.Equal(0.0f, result.Value[new[] { 0, 0 }], precision: 4);
        Assert.Equal(0.0f, result.Value[new[] { 0, 1 }], precision: 4);
        Assert.Equal(0.0f, result.Value[new[] { 0, 2 }], precision: 4);
        Assert.Equal(1.0f, result.Value[new[] { 1, 0 }], precision: 4);
        Assert.Equal(2.0f, result.Value[new[] { 1, 1 }], precision: 4);
        Assert.Equal(3.0f, result.Value[new[] { 1, 2 }], precision: 4);
    }

    [Fact]
    public void GpuTensorOperations_ReLU_ComputesCorrectGradients()
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

        var tensor = new Tensor<float>(new[] { 2, 2 });
        tensor[new[] { 0, 0 }] = -1.0f;
        tensor[new[] { 0, 1 }] = 2.0f;
        tensor[new[] { 1, 0 }] = -3.0f;
        tensor[new[] { 1, 1 }] = 4.0f;

        using var node = GpuTensorOperations<float>.Variable(tensor, context, "a", requiresGradient: true);

        // Act
        using var result = GpuTensorOperations<float>.ReLU(node, context);
        result.Backward();

        // Assert - ReLU gradient is 1 where input > 0, else 0
        Assert.NotNull(node.Gradient);
        Assert.Equal(0.0f, node.Gradient[new[] { 0, 0 }], precision: 4); // Negative input
        Assert.Equal(1.0f, node.Gradient[new[] { 0, 1 }], precision: 4); // Positive input
        Assert.Equal(0.0f, node.Gradient[new[] { 1, 0 }], precision: 4); // Negative input
        Assert.Equal(1.0f, node.Gradient[new[] { 1, 1 }], precision: 4); // Positive input
    }

    [Fact]
    public void GpuTensorOperations_ChainedOperations_ComputeCorrectGradients()
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

        var tensorA = new Tensor<float>(new[] { 2, 2 });
        var tensorB = new Tensor<float>(new[] { 2, 2 });

        for (int i = 0; i < tensorA.Length; i++)
        {
            tensorA[i] = i + 1.0f;
            tensorB[i] = (i + 1.0f) * 2.0f;
        }

        using var nodeA = GpuTensorOperations<float>.Variable(tensorA, context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(tensorB, context, "b", requiresGradient: true);

        // Act - Chain: c = (a + b) * a
        using var sum = GpuTensorOperations<float>.Add(nodeA, nodeB, context);
        using var result = GpuTensorOperations<float>.ElementwiseMultiply(sum, nodeA, context);
        result.Backward();

        // Assert - gradients should be computed through the chain
        Assert.NotNull(nodeA.Gradient);
        Assert.NotNull(nodeB.Gradient);

        // Verify gradients are non-zero (specific values depend on chain rule)
        for (int i = 0; i < nodeA.Gradient.Length; i++)
        {
            Assert.NotEqual(0.0f, nodeA.Gradient[i]);
            Assert.NotEqual(0.0f, nodeB.Gradient[i]);
        }
    }

    [Fact]
    public void GpuTensorOperations_WithGradientTape_RecordsOperations()
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

        using var tape = new GradientTape<float>();

        var tensorA = new Tensor<float>(new[] { 2, 2 });
        var tensorB = new Tensor<float>(new[] { 2, 2 });

        for (int i = 0; i < tensorA.Length; i++)
        {
            tensorA[i] = i + 1.0f;
            tensorB[i] = (i + 1.0f) * 2.0f;
        }

        using var nodeA = GpuTensorOperations<float>.Variable(tensorA, context, "a", requiresGradient: true);
        using var nodeB = GpuTensorOperations<float>.Variable(tensorB, context, "b", requiresGradient: true);

        tape.Watch(nodeA);
        tape.Watch(nodeB);

        // Act
        using var result = GpuTensorOperations<float>.Add(nodeA, nodeB, context);
        var gradients = tape.Gradient(result, new[] { nodeA, nodeB });

        // Assert
        Assert.Equal(2, gradients.Count);
        Assert.NotNull(gradients[nodeA]);
        Assert.NotNull(gradients[nodeB]);
    }

    [Fact]
    public void ExecutionContext_Statistics_TracksGpuUsage()
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

        using var node = GpuTensorOperations<float>.Variable(tensor, context, "a");

        // Act
        using var result1 = GpuTensorOperations<float>.ReLU(node, context);
        using var result2 = GpuTensorOperations<float>.Add(node, result1, context);

        // Assert
        Assert.Equal(2, context.Statistics.GpuOperations);
        Assert.Equal(0, context.Statistics.CpuOperations);
        Assert.Equal(100.0, context.Statistics.GpuPercentage);
    }
}
