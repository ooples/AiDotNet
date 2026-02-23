#nullable disable
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Autodiff;

/// <summary>
/// Comprehensive integration tests for ComputationNode automatic differentiation.
/// </summary>
/// <remarks>
/// <para>
/// These tests verify the correctness of the computation graph construction,
/// backward propagation, topological sorting, and gradient accumulation.
/// </para>
/// <para><b>For Beginners:</b> These tests ensure that the building blocks of
/// automatic differentiation work correctly. Each ComputationNode represents
/// a step in a calculation, and when connected together they form a graph
/// that can automatically compute derivatives (gradients).
/// </para>
/// </remarks>
public class ComputationNodeIntegrationTests
{
    private const double Tolerance = 1e-5;
    private const float FloatTolerance = 1e-4f;

    #region Node Construction Tests

    [Fact]
    public void Constructor_WithValue_StoresValueCorrectly()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 3 });
        tensor[0] = 1.0;
        tensor[1] = 2.0;
        tensor[2] = 3.0;

        // Act
        var node = new ComputationNode<double>(tensor);

        // Assert
        Assert.Equal(3, node.Value.Length);
        Assert.Equal(1.0, node.Value[0], Tolerance);
        Assert.Equal(2.0, node.Value[1], Tolerance);
        Assert.Equal(3.0, node.Value[2], Tolerance);
    }

    [Fact]
    public void Constructor_WithRequiresGradient_SetsFlag()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 2 });

        // Act
        var nodeWithGradient = new ComputationNode<double>(tensor, requiresGradient: true);
        var nodeWithoutGradient = new ComputationNode<double>(tensor, requiresGradient: false);

        // Assert
        Assert.True(nodeWithGradient.RequiresGradient);
        Assert.False(nodeWithoutGradient.RequiresGradient);
    }

    [Fact]
    public void Constructor_WithParents_StoresParents()
    {
        // Arrange
        var parent1 = new ComputationNode<double>(new Tensor<double>(new[] { 1 }));
        var parent2 = new ComputationNode<double>(new Tensor<double>(new[] { 1 }));
        var parents = new List<ComputationNode<double>> { parent1, parent2 };

        // Act
        var child = new ComputationNode<double>(
            new Tensor<double>(new[] { 1 }),
            parents: parents);

        // Assert
        Assert.Equal(2, child.Parents.Count);
        Assert.Same(parent1, child.Parents[0]);
        Assert.Same(parent2, child.Parents[1]);
    }

    [Fact]
    public void Constructor_WithName_StoresName()
    {
        // Arrange & Act
        var node = new ComputationNode<double>(
            new Tensor<double>(new[] { 1 }),
            name: "test_node");

        // Assert
        Assert.Equal("test_node", node.Name);
    }

    [Fact]
    public void Constructor_WithoutParents_HasEmptyParentsList()
    {
        // Act
        var node = new ComputationNode<double>(new Tensor<double>(new[] { 1 }));

        // Assert
        Assert.NotNull(node.Parents);
        Assert.Empty(node.Parents);
    }

    [Fact]
    public void Constructor_InitializesGradientToNull()
    {
        // Act
        var node = new ComputationNode<double>(new Tensor<double>(new[] { 1 }));

        // Assert
        Assert.Null(node.Gradient);
    }

    #endregion

    #region Simple Backward Propagation Tests

    [Fact]
    public void Backward_SingleNode_InitializesGradientToOnes()
    {
        // Arrange - Single node representing output
        var tensor = new Tensor<double>(new[] { 3 });
        tensor[0] = 1.0;
        tensor[1] = 2.0;
        tensor[2] = 3.0;

        var node = new ComputationNode<double>(tensor, requiresGradient: true);

        // Act
        node.Backward();

        // Assert - Gradient should be all ones
        Assert.NotNull(node.Gradient);
        Assert.Equal(3, node.Gradient.Length);
        Assert.Equal(1.0, node.Gradient[0], Tolerance);
        Assert.Equal(1.0, node.Gradient[1], Tolerance);
        Assert.Equal(1.0, node.Gradient[2], Tolerance);
    }

    [Fact]
    public void Backward_LinearChain_PropagatesGradientCorrectly()
    {
        // Arrange - Chain: x -> y = 2*x -> z = y + 1
        // dz/dx = dz/dy * dy/dx = 1 * 2 = 2

        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = 3.0;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true, name: "x");

        var yTensor = new Tensor<double>(new[] { 1 });
        yTensor[0] = 6.0;  // 2 * 3
        var y = new ComputationNode<double>(
            yTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                // y = 2*x, so dy/dx = 2, gradient flows: x.grad += 2 * grad
                if (x.Gradient == null)
                {
                    x.Gradient = new Tensor<double>(x.Value.Shape);
                }
                x.Gradient[0] += 2.0 * grad[0];
            },
            name: "y");

        var zTensor = new Tensor<double>(new[] { 1 });
        zTensor[0] = 7.0;  // 6 + 1
        var z = new ComputationNode<double>(
            zTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { y },
            backwardFunction: grad =>
            {
                // z = y + 1, so dz/dy = 1, gradient flows: y.grad += 1 * grad
                if (y.Gradient == null)
                {
                    y.Gradient = new Tensor<double>(y.Value.Shape);
                }
                y.Gradient[0] += grad[0];
            },
            name: "z");

        // Act
        z.Backward();

        // Assert
        Assert.NotNull(z.Gradient);
        Assert.Equal(1.0, z.Gradient[0], Tolerance);  // dz/dz = 1

        Assert.NotNull(y.Gradient);
        Assert.Equal(1.0, y.Gradient[0], Tolerance);  // dz/dy = 1

        Assert.NotNull(x.Gradient);
        Assert.Equal(2.0, x.Gradient[0], Tolerance);  // dz/dx = 2
    }

    [Fact]
    public void Backward_BinaryOperation_ComputesBothGradients()
    {
        // Arrange - z = x * y (elementwise)
        // dz/dx = y, dz/dy = x

        var xTensor = new Tensor<double>(new[] { 2 });
        xTensor[0] = 2.0;
        xTensor[1] = 3.0;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true, name: "x");

        var yTensor = new Tensor<double>(new[] { 2 });
        yTensor[0] = 4.0;
        yTensor[1] = 5.0;
        var y = new ComputationNode<double>(yTensor, requiresGradient: true, name: "y");

        var zTensor = new Tensor<double>(new[] { 2 });
        zTensor[0] = 8.0;   // 2 * 4
        zTensor[1] = 15.0;  // 3 * 5
        var z = new ComputationNode<double>(
            zTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x, y },
            backwardFunction: grad =>
            {
                // z = x * y elementwise
                // dz/dx = y, dz/dy = x
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                if (y.Gradient == null) y.Gradient = new Tensor<double>(y.Value.Shape);

                for (int i = 0; i < grad.Length; i++)
                {
                    x.Gradient[i] += y.Value[i] * grad[i];
                    y.Gradient[i] += x.Value[i] * grad[i];
                }
            },
            name: "z");

        // Act
        z.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(4.0, x.Gradient[0], Tolerance);  // dz/dx[0] = y[0] = 4
        Assert.Equal(5.0, x.Gradient[1], Tolerance);  // dz/dx[1] = y[1] = 5

        Assert.NotNull(y.Gradient);
        Assert.Equal(2.0, y.Gradient[0], Tolerance);  // dz/dy[0] = x[0] = 2
        Assert.Equal(3.0, y.Gradient[1], Tolerance);  // dz/dy[1] = x[1] = 3
    }

    #endregion

    #region Gradient Accumulation Tests

    [Fact]
    public void Backward_MultiplePaths_AccumulatesGradients()
    {
        // Arrange - Diamond pattern: x -> (y1, y2) -> z
        // z = y1 + y2, y1 = x, y2 = x
        // dz/dx = dz/dy1 * dy1/dx + dz/dy2 * dy2/dx = 1 * 1 + 1 * 1 = 2

        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = 5.0;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true, name: "x");

        var y1Tensor = new Tensor<double>(new[] { 1 });
        y1Tensor[0] = 5.0;
        var y1 = new ComputationNode<double>(
            y1Tensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                x.Gradient[0] += grad[0];  // dy1/dx = 1
            },
            name: "y1");

        var y2Tensor = new Tensor<double>(new[] { 1 });
        y2Tensor[0] = 5.0;
        var y2 = new ComputationNode<double>(
            y2Tensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                x.Gradient[0] += grad[0];  // dy2/dx = 1
            },
            name: "y2");

        var zTensor = new Tensor<double>(new[] { 1 });
        zTensor[0] = 10.0;  // 5 + 5
        var z = new ComputationNode<double>(
            zTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { y1, y2 },
            backwardFunction: grad =>
            {
                if (y1.Gradient == null) y1.Gradient = new Tensor<double>(y1.Value.Shape);
                if (y2.Gradient == null) y2.Gradient = new Tensor<double>(y2.Value.Shape);
                y1.Gradient[0] += grad[0];  // dz/dy1 = 1
                y2.Gradient[0] += grad[0];  // dz/dy2 = 1
            },
            name: "z");

        // Act
        z.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(2.0, x.Gradient[0], Tolerance);  // Gradient accumulated from both paths
    }

    [Fact]
    public void Backward_RepeatedNode_AccumulatesCorrectly()
    {
        // Arrange - z = x * x (squaring uses x twice)
        // dz/dx = 2x

        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = 3.0;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true, name: "x");

        var zTensor = new Tensor<double>(new[] { 1 });
        zTensor[0] = 9.0;  // 3 * 3
        var z = new ComputationNode<double>(
            zTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x, x },  // x appears twice
            backwardFunction: grad =>
            {
                // z = x * x, dz/dx = 2x
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                x.Gradient[0] += 2 * x.Value[0] * grad[0];
            },
            name: "z");

        // Act
        z.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(6.0, x.Gradient[0], Tolerance);  // dz/dx = 2 * 3 = 6
    }

    #endregion

    #region ZeroGradient Tests

    [Fact]
    public void ZeroGradient_ClearsGradient()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 3 });
        var node = new ComputationNode<double>(tensor, requiresGradient: true);
        node.Backward();  // This sets gradient to ones

        // Verify gradient is set
        Assert.NotNull(node.Gradient);
        Assert.Equal(1.0, node.Gradient[0], Tolerance);

        // Act
        node.ZeroGradient();

        // Assert
        Assert.NotNull(node.Gradient);
        Assert.Equal(0.0, node.Gradient[0], Tolerance);
        Assert.Equal(0.0, node.Gradient[1], Tolerance);
        Assert.Equal(0.0, node.Gradient[2], Tolerance);
    }

    [Fact]
    public void ZeroGradient_NullGradient_DoesNotThrow()
    {
        // Arrange
        var node = new ComputationNode<double>(new Tensor<double>(new[] { 2 }));
        Assert.Null(node.Gradient);

        // Act & Assert - Should not throw
        node.ZeroGradient();
        Assert.Null(node.Gradient);  // Should remain null
    }

    [Fact]
    public void ZeroGradientRecursive_ClearsAllGradients()
    {
        // Arrange - Build a small graph
        var x = new ComputationNode<double>(new Tensor<double>(new[] { 1 }), requiresGradient: true);
        var y = new ComputationNode<double>(
            new Tensor<double>(new[] { 1 }),
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                x.Gradient[0] += grad[0];
            });
        var z = new ComputationNode<double>(
            new Tensor<double>(new[] { 1 }),
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { y },
            backwardFunction: grad =>
            {
                if (y.Gradient == null) y.Gradient = new Tensor<double>(y.Value.Shape);
                y.Gradient[0] += grad[0];
            });

        // Perform backward to set gradients
        z.Backward();

        // Verify gradients are set
        Assert.NotNull(z.Gradient);
        Assert.NotNull(y.Gradient);
        Assert.NotNull(x.Gradient);

        // Act
        z.ZeroGradientRecursive();

        // Assert - All gradients should be zeroed
        Assert.Equal(0.0, z.Gradient[0], Tolerance);
        Assert.Equal(0.0, y.Gradient[0], Tolerance);
        Assert.Equal(0.0, x.Gradient[0], Tolerance);
    }

    #endregion

    #region RequiresGradient Flag Tests

    [Fact]
    public void Backward_NodeWithoutRequiresGradient_DoesNotComputeBackward()
    {
        // Arrange - y depends on x, but x doesn't require gradient
        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = 2.0;
        var x = new ComputationNode<double>(xTensor, requiresGradient: false, name: "x");

        bool backwardCalled = false;
        var yTensor = new Tensor<double>(new[] { 1 });
        yTensor[0] = 4.0;
        var y = new ComputationNode<double>(
            yTensor,
            requiresGradient: false,  // No gradient required
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad => { backwardCalled = true; },
            name: "y");

        // Act
        y.Backward();

        // Assert - BackwardFunction should not have been called
        Assert.False(backwardCalled);
    }

    [Fact]
    public void Backward_MixedRequiresGradient_PropagatesOnlyToRequired()
    {
        // Arrange - x1 requires gradient, x2 doesn't
        var x1Tensor = new Tensor<double>(new[] { 1 });
        x1Tensor[0] = 2.0;
        var x1 = new ComputationNode<double>(x1Tensor, requiresGradient: true, name: "x1");

        var x2Tensor = new Tensor<double>(new[] { 1 });
        x2Tensor[0] = 3.0;
        var x2 = new ComputationNode<double>(x2Tensor, requiresGradient: false, name: "x2");

        var yTensor = new Tensor<double>(new[] { 1 });
        yTensor[0] = 5.0;  // x1 + x2
        var y = new ComputationNode<double>(
            yTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x1, x2 },
            backwardFunction: grad =>
            {
                if (x1.Gradient == null) x1.Gradient = new Tensor<double>(x1.Value.Shape);
                x1.Gradient[0] += grad[0];
                // x2 doesn't require gradient, but we could still set it
            },
            name: "y");

        // Act
        y.Backward();

        // Assert
        Assert.NotNull(x1.Gradient);
        Assert.Equal(1.0, x1.Gradient[0], Tolerance);
        Assert.Null(x2.Gradient);  // Never received gradient
    }

    #endregion

    #region Deep Graph Tests

    [Fact]
    public void Backward_DeepGraph_HandlesWithoutStackOverflow()
    {
        // Arrange - Create a deep chain of nodes
        const int depth = 1000;

        var nodes = new List<ComputationNode<double>>();
        var firstTensor = new Tensor<double>(new[] { 1 });
        firstTensor[0] = 1.0;
        nodes.Add(new ComputationNode<double>(firstTensor, requiresGradient: true));

        for (int i = 1; i < depth; i++)
        {
            var prevNode = nodes[i - 1];
            var tensor = new Tensor<double>(new[] { 1 });
            tensor[0] = 1.0;
            var node = new ComputationNode<double>(
                tensor,
                requiresGradient: true,
                parents: new List<ComputationNode<double>> { prevNode },
                backwardFunction: grad =>
                {
                    if (prevNode.Gradient == null)
                        prevNode.Gradient = new Tensor<double>(prevNode.Value.Shape);
                    prevNode.Gradient[0] += grad[0];
                });
            nodes.Add(node);
        }

        // Act - Should not throw StackOverflowException
        nodes[^1].Backward();

        // Assert
        Assert.NotNull(nodes[0].Gradient);
        Assert.Equal(1.0, nodes[0].Gradient[0], Tolerance);
    }

    #endregion

    #region Complex Graph Topology Tests

    [Fact]
    public void Backward_DenseConnections_ComputesCorrectGradients()
    {
        // Arrange - Multiple connections between layers
        // Layer 0: x1, x2
        // Layer 1: y1 = x1 + x2, y2 = x1 - x2
        // Layer 2: z = y1 * y2

        var x1Tensor = new Tensor<double>(new[] { 1 });
        x1Tensor[0] = 3.0;
        var x1 = new ComputationNode<double>(x1Tensor, requiresGradient: true, name: "x1");

        var x2Tensor = new Tensor<double>(new[] { 1 });
        x2Tensor[0] = 2.0;
        var x2 = new ComputationNode<double>(x2Tensor, requiresGradient: true, name: "x2");

        // y1 = x1 + x2 = 5
        var y1Tensor = new Tensor<double>(new[] { 1 });
        y1Tensor[0] = 5.0;
        var y1 = new ComputationNode<double>(
            y1Tensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x1, x2 },
            backwardFunction: grad =>
            {
                if (x1.Gradient == null) x1.Gradient = new Tensor<double>(x1.Value.Shape);
                if (x2.Gradient == null) x2.Gradient = new Tensor<double>(x2.Value.Shape);
                x1.Gradient[0] += grad[0];  // dy1/dx1 = 1
                x2.Gradient[0] += grad[0];  // dy1/dx2 = 1
            },
            name: "y1");

        // y2 = x1 - x2 = 1
        var y2Tensor = new Tensor<double>(new[] { 1 });
        y2Tensor[0] = 1.0;
        var y2 = new ComputationNode<double>(
            y2Tensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x1, x2 },
            backwardFunction: grad =>
            {
                if (x1.Gradient == null) x1.Gradient = new Tensor<double>(x1.Value.Shape);
                if (x2.Gradient == null) x2.Gradient = new Tensor<double>(x2.Value.Shape);
                x1.Gradient[0] += grad[0];   // dy2/dx1 = 1
                x2.Gradient[0] += -grad[0];  // dy2/dx2 = -1
            },
            name: "y2");

        // z = y1 * y2 = 5
        var zTensor = new Tensor<double>(new[] { 1 });
        zTensor[0] = 5.0;
        var z = new ComputationNode<double>(
            zTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { y1, y2 },
            backwardFunction: grad =>
            {
                if (y1.Gradient == null) y1.Gradient = new Tensor<double>(y1.Value.Shape);
                if (y2.Gradient == null) y2.Gradient = new Tensor<double>(y2.Value.Shape);
                y1.Gradient[0] += y2.Value[0] * grad[0];  // dz/dy1 = y2
                y2.Gradient[0] += y1.Value[0] * grad[0];  // dz/dy2 = y1
            },
            name: "z");

        // Act
        z.Backward();

        // Assert
        // dz/dy1 = y2 = 1
        // dz/dy2 = y1 = 5
        Assert.Equal(1.0, y1.Gradient![0], Tolerance);
        Assert.Equal(5.0, y2.Gradient![0], Tolerance);

        // dz/dx1 = dz/dy1 * dy1/dx1 + dz/dy2 * dy2/dx1 = 1*1 + 5*1 = 6
        // dz/dx2 = dz/dy1 * dy1/dx2 + dz/dy2 * dy2/dx2 = 1*1 + 5*(-1) = -4
        Assert.Equal(6.0, x1.Gradient![0], Tolerance);
        Assert.Equal(-4.0, x2.Gradient![0], Tolerance);
    }

    [Fact]
    public void Backward_ParallelBranches_ComputesCorrectGradients()
    {
        // Arrange - x -> [branch1: y1 = 2x, branch2: y2 = 3x] -> z = y1 + y2
        // dz/dx = dz/dy1 * dy1/dx + dz/dy2 * dy2/dx = 1*2 + 1*3 = 5

        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = 4.0;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true, name: "x");

        var y1Tensor = new Tensor<double>(new[] { 1 });
        y1Tensor[0] = 8.0;  // 2 * 4
        var y1 = new ComputationNode<double>(
            y1Tensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                x.Gradient[0] += 2.0 * grad[0];  // dy1/dx = 2
            },
            name: "y1");

        var y2Tensor = new Tensor<double>(new[] { 1 });
        y2Tensor[0] = 12.0;  // 3 * 4
        var y2 = new ComputationNode<double>(
            y2Tensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                x.Gradient[0] += 3.0 * grad[0];  // dy2/dx = 3
            },
            name: "y2");

        var zTensor = new Tensor<double>(new[] { 1 });
        zTensor[0] = 20.0;  // 8 + 12
        var z = new ComputationNode<double>(
            zTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { y1, y2 },
            backwardFunction: grad =>
            {
                if (y1.Gradient == null) y1.Gradient = new Tensor<double>(y1.Value.Shape);
                if (y2.Gradient == null) y2.Gradient = new Tensor<double>(y2.Value.Shape);
                y1.Gradient[0] += grad[0];
                y2.Gradient[0] += grad[0];
            },
            name: "z");

        // Act
        z.Backward();

        // Assert
        Assert.Equal(5.0, x.Gradient![0], Tolerance);  // 2 + 3 = 5
    }

    #endregion

    #region Multiple Backward Passes Tests

    [Fact]
    public void Backward_CalledMultipleTimes_ClearsGradientsBetweenPasses()
    {
        // Arrange
        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = 2.0;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true);

        var yTensor = new Tensor<double>(new[] { 1 });
        yTensor[0] = 4.0;
        var y = new ComputationNode<double>(
            yTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                x.Gradient[0] += 2.0 * grad[0];
            });

        // Act - First backward pass
        y.Backward();
        var firstGradient = x.Gradient![0];

        // Second backward pass (should clear and recompute)
        y.Backward();
        var secondGradient = x.Gradient![0];

        // Assert - Both passes should give the same result (not accumulated)
        Assert.Equal(2.0, firstGradient, Tolerance);
        Assert.Equal(2.0, secondGradient, Tolerance);  // Should NOT be 4.0
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void Backward_FloatType_WorksCorrectly()
    {
        // Arrange - Test with float type
        var xTensor = new Tensor<float>(new[] { 1 });
        xTensor[0] = 3.0f;
        var x = new ComputationNode<float>(xTensor, requiresGradient: true);

        var yTensor = new Tensor<float>(new[] { 1 });
        yTensor[0] = 9.0f;  // x^2
        var y = new ComputationNode<float>(
            yTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<float>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<float>(x.Value.Shape);
                x.Gradient[0] += 2.0f * x.Value[0] * grad[0];  // dy/dx = 2x
            });

        // Act
        y.Backward();

        // Assert
        Assert.Equal(6.0f, x.Gradient![0], FloatTolerance);  // 2 * 3 = 6
    }

    #endregion

    #region Operation Type and Params Tests

    [Fact]
    public void OperationType_CanBeSetAndRetrieved()
    {
        // Arrange & Act
        var node = new ComputationNode<double>(new Tensor<double>(new[] { 1 }));
        node.OperationType = OperationType.Add;

        // Assert
        Assert.Equal(OperationType.Add, node.OperationType);
    }

    [Fact]
    public void OperationParams_CanBeSetAndRetrieved()
    {
        // Arrange
        var node = new ComputationNode<double>(new Tensor<double>(new[] { 1 }));
        var operationParams = new Dictionary<string, object>
        {
            { "axis", 0 },
            { "keepdims", true },
            { "scale", 2.5 }
        };

        // Act
        node.OperationParams = operationParams;

        // Assert
        Assert.NotNull(node.OperationParams);
        Assert.Equal(0, node.OperationParams["axis"]);
        Assert.Equal(true, node.OperationParams["keepdims"]);
        Assert.Equal(2.5, node.OperationParams["scale"]);
    }

    #endregion

    #region Edge Cases and Robustness Tests

    [Fact]
    public void Backward_EmptyParentsList_WorksCorrectly()
    {
        // Arrange - Leaf node with no parents
        var tensor = new Tensor<double>(new[] { 2 });
        tensor[0] = 1.0;
        tensor[1] = 2.0;
        var node = new ComputationNode<double>(tensor, requiresGradient: true);

        // Act
        node.Backward();

        // Assert
        Assert.NotNull(node.Gradient);
        Assert.Equal(1.0, node.Gradient[0], Tolerance);
        Assert.Equal(1.0, node.Gradient[1], Tolerance);
    }

    [Fact]
    public void Backward_NullBackwardFunction_DoesNotThrow()
    {
        // Arrange
        var parentTensor = new Tensor<double>(new[] { 1 });
        var parent = new ComputationNode<double>(parentTensor, requiresGradient: true);

        var childTensor = new Tensor<double>(new[] { 1 });
        var child = new ComputationNode<double>(
            childTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { parent },
            backwardFunction: null);  // No backward function

        // Act & Assert - Should not throw
        child.Backward();
    }

    [Fact]
    public void Value_CanBeModifiedAfterConstruction()
    {
        // Arrange
        var tensor = new Tensor<double>(new[] { 2 });
        tensor[0] = 1.0;
        tensor[1] = 2.0;
        var node = new ComputationNode<double>(tensor);

        // Act
        var newTensor = new Tensor<double>(new[] { 3 });
        newTensor[0] = 5.0;
        newTensor[1] = 6.0;
        newTensor[2] = 7.0;
        node.Value = newTensor;

        // Assert
        Assert.Equal(3, node.Value.Length);
        Assert.Equal(5.0, node.Value[0], Tolerance);
    }

    [Fact]
    public void Gradient_CanBeSetManually()
    {
        // Arrange
        var node = new ComputationNode<double>(new Tensor<double>(new[] { 2 }), requiresGradient: true);

        // Act
        var gradient = new Tensor<double>(new[] { 2 });
        gradient[0] = 10.0;
        gradient[1] = 20.0;
        node.Gradient = gradient;

        // Assert
        Assert.NotNull(node.Gradient);
        Assert.Equal(10.0, node.Gradient[0], Tolerance);
        Assert.Equal(20.0, node.Gradient[1], Tolerance);
    }

    #endregion

    #region Tensor Shape Tests

    [Fact]
    public void Backward_2DShape_PropagatesCorrectly()
    {
        // Arrange - 2x3 tensor
        var xTensor = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < 6; i++) xTensor[i] = i + 1;  // 1, 2, 3, 4, 5, 6

        var x = new ComputationNode<double>(xTensor, requiresGradient: true);

        var yTensor = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < 6; i++) yTensor[i] = 2.0 * (i + 1);  // 2, 4, 6, 8, 10, 12

        var y = new ComputationNode<double>(
            yTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                // y = 2*x, dy/dx = 2
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                for (int i = 0; i < grad.Length; i++)
                {
                    x.Gradient[i] += 2.0 * grad[i];
                }
            });

        // Act
        y.Backward();

        // Assert - All gradients should be 2
        Assert.Equal(new[] { 2, 3 }, x.Gradient!.Shape);
        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(2.0, x.Gradient[i], Tolerance);
        }
    }

    [Fact]
    public void Backward_3DShape_PropagatesCorrectly()
    {
        // Arrange - 2x2x2 tensor
        var xTensor = new Tensor<double>(new[] { 2, 2, 2 });
        for (int i = 0; i < 8; i++) xTensor[i] = 1.0;

        var x = new ComputationNode<double>(xTensor, requiresGradient: true);

        // Act
        x.Backward();

        // Assert
        Assert.NotNull(x.Gradient);
        Assert.Equal(new[] { 2, 2, 2 }, x.Gradient.Shape);
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(1.0, x.Gradient[i], Tolerance);
        }
    }

    #endregion

    #region Chain Rule Verification Tests

    [Fact]
    public void ChainRule_Exp_ComputesCorrectGradient()
    {
        // f(x) = e^x, df/dx = e^x

        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = 2.0;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true);

        var yTensor = new Tensor<double>(new[] { 1 });
        yTensor[0] = Math.Exp(2.0);
        var y = new ComputationNode<double>(
            yTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                x.Gradient[0] += yTensor[0] * grad[0];  // d(e^x)/dx = e^x
            });

        // Act
        y.Backward();

        // Assert
        Assert.Equal(Math.Exp(2.0), x.Gradient![0], Tolerance);
    }

    [Fact]
    public void ChainRule_SinOfSquare_ComputesCorrectGradient()
    {
        // f(x) = sin(x^2), df/dx = cos(x^2) * 2x
        // x = 1.5: df/dx = cos(2.25) * 3 ≈ -0.628 * 3 ≈ -1.885

        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = 1.5;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true);

        // y = x^2 = 2.25
        var yTensor = new Tensor<double>(new[] { 1 });
        yTensor[0] = 2.25;
        var y = new ComputationNode<double>(
            yTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                x.Gradient[0] += 2.0 * x.Value[0] * grad[0];  // dy/dx = 2x
            });

        // z = sin(y)
        var zTensor = new Tensor<double>(new[] { 1 });
        zTensor[0] = Math.Sin(2.25);
        var z = new ComputationNode<double>(
            zTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { y },
            backwardFunction: grad =>
            {
                if (y.Gradient == null) y.Gradient = new Tensor<double>(y.Value.Shape);
                y.Gradient[0] += Math.Cos(yTensor[0]) * grad[0];  // dz/dy = cos(y)
            });

        // Act
        z.Backward();

        // Assert
        double expected = Math.Cos(2.25) * 2.0 * 1.5;  // cos(x^2) * 2x
        Assert.Equal(expected, x.Gradient![0], Tolerance);
    }

    [Fact]
    public void ChainRule_CompositeFunction_ComputesCorrectGradient()
    {
        // f(x) = log(1 + e^x) (softplus)
        // df/dx = e^x / (1 + e^x) = sigmoid(x)
        // x = 1.0: df/dx = e^1 / (1 + e^1) ≈ 0.731

        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = 1.0;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true);

        // y = e^x
        var expX = Math.Exp(1.0);
        var yTensor = new Tensor<double>(new[] { 1 });
        yTensor[0] = expX;
        var y = new ComputationNode<double>(
            yTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                x.Gradient[0] += expX * grad[0];  // de^x/dx = e^x
            });

        // w = 1 + y = 1 + e^x
        var onePlusExpX = 1.0 + expX;
        var wTensor = new Tensor<double>(new[] { 1 });
        wTensor[0] = onePlusExpX;
        var w = new ComputationNode<double>(
            wTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { y },
            backwardFunction: grad =>
            {
                if (y.Gradient == null) y.Gradient = new Tensor<double>(y.Value.Shape);
                y.Gradient[0] += grad[0];  // d(1+y)/dy = 1
            });

        // z = log(w)
        var zTensor = new Tensor<double>(new[] { 1 });
        zTensor[0] = Math.Log(onePlusExpX);
        var z = new ComputationNode<double>(
            zTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { w },
            backwardFunction: grad =>
            {
                if (w.Gradient == null) w.Gradient = new Tensor<double>(w.Value.Shape);
                w.Gradient[0] += (1.0 / wTensor[0]) * grad[0];  // dlog(w)/dw = 1/w
            });

        // Act
        z.Backward();

        // Assert - Should equal sigmoid(1)
        double expectedSigmoid = expX / onePlusExpX;
        Assert.Equal(expectedSigmoid, x.Gradient![0], Tolerance);
    }

    #endregion

    #region Numerical Gradient Verification

    [Fact]
    public void Backward_QuadraticFunction_MatchesNumericalGradient()
    {
        // f(x, y) = x^2 + 2xy + y^2 = (x + y)^2
        // df/dx = 2x + 2y = 2(x + y)
        // df/dy = 2x + 2y = 2(x + y)

        double xVal = 3.0;
        double yVal = 4.0;

        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = xVal;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true);

        var yTensor = new Tensor<double>(new[] { 1 });
        yTensor[0] = yVal;
        var y = new ComputationNode<double>(yTensor, requiresGradient: true);

        // f = x^2 + 2xy + y^2
        var fTensor = new Tensor<double>(new[] { 1 });
        fTensor[0] = xVal * xVal + 2 * xVal * yVal + yVal * yVal;  // 49
        var f = new ComputationNode<double>(
            fTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x, y },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                if (y.Gradient == null) y.Gradient = new Tensor<double>(y.Value.Shape);
                x.Gradient[0] += (2 * x.Value[0] + 2 * y.Value[0]) * grad[0];
                y.Gradient[0] += (2 * x.Value[0] + 2 * y.Value[0]) * grad[0];
            });

        // Act
        f.Backward();

        // Assert - Compare with numerical gradient
        double eps = 1e-5;
        Func<double, double, double> func = (a, b) => a * a + 2 * a * b + b * b;

        double numGradX = (func(xVal + eps, yVal) - func(xVal - eps, yVal)) / (2 * eps);
        double numGradY = (func(xVal, yVal + eps) - func(xVal, yVal - eps)) / (2 * eps);

        Assert.Equal(numGradX, x.Gradient![0], 1e-4);  // Numerical gradient tolerance
        Assert.Equal(numGradY, y.Gradient![0], 1e-4);
    }

    [Fact]
    public void Backward_RationalFunction_MatchesNumericalGradient()
    {
        // f(x) = x / (1 + x^2)
        // df/dx = (1 + x^2 - x * 2x) / (1 + x^2)^2 = (1 - x^2) / (1 + x^2)^2

        double xVal = 2.0;

        var xTensor = new Tensor<double>(new[] { 1 });
        xTensor[0] = xVal;
        var x = new ComputationNode<double>(xTensor, requiresGradient: true);

        double onePlusXSq = 1 + xVal * xVal;
        var fTensor = new Tensor<double>(new[] { 1 });
        fTensor[0] = xVal / onePlusXSq;

        var f = new ComputationNode<double>(
            fTensor,
            requiresGradient: true,
            parents: new List<ComputationNode<double>> { x },
            backwardFunction: grad =>
            {
                if (x.Gradient == null) x.Gradient = new Tensor<double>(x.Value.Shape);
                double xv = x.Value[0];
                double denom = 1 + xv * xv;
                double deriv = (1 - xv * xv) / (denom * denom);
                x.Gradient[0] += deriv * grad[0];
            });

        // Act
        f.Backward();

        // Assert - Compare with numerical gradient
        double eps = 1e-5;
        Func<double, double> func = a => a / (1 + a * a);
        double numGrad = (func(xVal + eps) - func(xVal - eps)) / (2 * eps);

        Assert.Equal(numGrad, x.Gradient![0], 1e-4);
    }

    #endregion
}
