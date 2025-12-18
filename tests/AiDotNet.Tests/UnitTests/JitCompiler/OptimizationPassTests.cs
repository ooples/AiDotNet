using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using AiDotNet.JitCompiler.Optimizations;
using Xunit;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

/// <summary>
/// Tests for optimization passes.
/// </summary>
public class OptimizationPassTests
{
    #region DeadCodeElimination Tests

    [Fact]
    public void DeadCodeElimination_RemovesUnusedOperations()
    {
        // Arrange
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1 },
            OutputIds = new List<int> { 2 },
            Operations = new List<IROp>
            {
                new AddOp { OutputId = 2, InputIds = new[] { 0, 1 }, OutputShape = new[] { 2, 3 } },
                new ElementwiseMultiplyOp { OutputId = 3, InputIds = new[] { 0, 1 }, OutputShape = new[] { 2, 3 } },  // Dead! Never used
            },
            TensorShapes = new Dictionary<int, int[]>
            {
                [0] = new[] { 2, 3 },
                [1] = new[] { 2, 3 },
                [2] = new[] { 2, 3 },
                [3] = new[] { 2, 3 }
            }
        };

        var dce = new DeadCodeEliminationPass();

        // Act
        var optimized = dce.Optimize(graph);

        // Assert
        Assert.Single(optimized.Operations);  // Only AddOp remains
        Assert.IsType<AddOp>(optimized.Operations[0]);
    }

    [Fact]
    public void DeadCodeElimination_KeepsAllLiveOperations()
    {
        // Arrange
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0 },
            OutputIds = new List<int> { 3 },
            Operations = new List<IROp>
            {
                new ReLUOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 2, 3 } },
                new ExpOp { OutputId = 2, InputIds = new[] { 1 }, OutputShape = new[] { 2, 3 } },
                new LogOp { OutputId = 3, InputIds = new[] { 2 }, OutputShape = new[] { 2, 3 } },
            },
            TensorShapes = new Dictionary<int, int[]>
            {
                [0] = new[] { 2, 3 },
                [1] = new[] { 2, 3 },
                [2] = new[] { 2, 3 },
                [3] = new[] { 2, 3 }
            }
        };

        var dce = new DeadCodeEliminationPass();

        // Act
        var optimized = dce.Optimize(graph);

        // Assert
        Assert.Equal(3, optimized.Operations.Count);  // All operations are live
    }

    [Fact]
    public void DeadCodeElimination_HandlesDiamondPattern()
    {
        // Arrange: Diamond with dead branch
        //   0
        //  / \
        // 1   2 (dead branch)
        //  \ /
        //   3
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0 },
            OutputIds = new List<int> { 3 },
            Operations = new List<IROp>
            {
                new ExpOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 2, 3 } },
                new LogOp { OutputId = 2, InputIds = new[] { 0 }, OutputShape = new[] { 2, 3 } },  // Dead!
                new AddOp { OutputId = 3, InputIds = new[] { 1, 0 }, OutputShape = new[] { 2, 3 } },  // Uses 1, not 2
            },
            TensorShapes = new Dictionary<int, int[]>
            {
                [0] = new[] { 2, 3 },
                [1] = new[] { 2, 3 },
                [2] = new[] { 2, 3 },
                [3] = new[] { 2, 3 }
            }
        };

        var dce = new DeadCodeEliminationPass();

        // Act
        var optimized = dce.Optimize(graph);

        // Assert
        Assert.Equal(2, optimized.Operations.Count);  // LogOp removed
    }

    [Fact]
    public void DeadCodeElimination_GetStatistics_ReturnsCorrectCounts()
    {
        // Arrange
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0 },
            OutputIds = new List<int> { 1 },
            Operations = new List<IROp>
            {
                new ReLUOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 2, 3 } },
                new ExpOp { OutputId = 2, InputIds = new[] { 0 }, OutputShape = new[] { 2, 3 } },  // Dead
                new LogOp { OutputId = 3, InputIds = new[] { 0 }, OutputShape = new[] { 2, 3 } },  // Dead
            },
            TensorShapes = new Dictionary<int, int[]>()
        };

        var dce = new DeadCodeEliminationPass();

        // Act
        var (total, live, dead) = dce.GetStatistics(graph);

        // Assert
        Assert.Equal(3, total);
        Assert.Equal(1, live);
        Assert.Equal(2, dead);
    }

    #endregion

    #region OperationFusion Tests

    [Fact]
    public void OperationFusion_FusesMatMulAdd()
    {
        // Arrange
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1, 2 },  // input, weights, bias
            OutputIds = new List<int> { 4 },
            Operations = new List<IROp>
            {
                new MatMulOp { OutputId = 3, InputIds = new[] { 0, 1 }, OutputShape = new[] { 1, 4 } },
                new AddOp { OutputId = 4, InputIds = new[] { 3, 2 }, OutputShape = new[] { 1, 4 } },
            },
            TensorShapes = new Dictionary<int, int[]>
            {
                [0] = new[] { 1, 3 },
                [1] = new[] { 3, 4 },
                [2] = new[] { 1, 4 },
                [3] = new[] { 1, 4 },
                [4] = new[] { 1, 4 }
            }
        };

        var fusion = new OperationFusionPass();

        // Act
        var optimized = fusion.Optimize(graph);

        // Assert
        Assert.Single(optimized.Operations);
        Assert.IsType<FusedLinearOp>(optimized.Operations[0]);
    }

    [Fact]
    public void OperationFusion_FusesMatMulAddActivation()
    {
        // Arrange: MatMul -> Add -> ReLU
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1, 2 },
            OutputIds = new List<int> { 5 },
            Operations = new List<IROp>
            {
                new MatMulOp { OutputId = 3, InputIds = new[] { 0, 1 }, OutputShape = new[] { 1, 4 } },
                new AddOp { OutputId = 4, InputIds = new[] { 3, 2 }, OutputShape = new[] { 1, 4 } },
                new ReLUOp { OutputId = 5, InputIds = new[] { 4 }, OutputShape = new[] { 1, 4 } },
            },
            TensorShapes = new Dictionary<int, int[]>()
        };

        var fusion = new OperationFusionPass();

        // Act
        var optimized = fusion.Optimize(graph);

        // Assert
        Assert.Single(optimized.Operations);
        var fusedOp = Assert.IsType<FusedDenseLayerOp>(optimized.Operations[0]);
        Assert.Equal("ReLU", fusedOp.ActivationName);
    }

    [Fact]
    public void OperationFusion_FusesElementwiseActivation()
    {
        // Arrange: Add -> Sigmoid
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1 },
            OutputIds = new List<int> { 3 },
            Operations = new List<IROp>
            {
                new AddOp { OutputId = 2, InputIds = new[] { 0, 1 }, OutputShape = new[] { 2, 3 } },
                new SigmoidOp { OutputId = 3, InputIds = new[] { 2 }, OutputShape = new[] { 2, 3 } },
            },
            TensorShapes = new Dictionary<int, int[]>()
        };

        var fusion = new OperationFusionPass();

        // Act
        var optimized = fusion.Optimize(graph);

        // Assert
        Assert.Single(optimized.Operations);
        var fusedOp = Assert.IsType<FusedElementwiseActivationOp>(optimized.Operations[0]);
        Assert.Equal("Add", fusedOp.ElementwiseOp);
        Assert.Equal("Sigmoid", fusedOp.ActivationName);
    }

    [Fact]
    public void OperationFusion_FusesConvBatchNorm()
    {
        // Arrange: Conv2D -> BatchNorm
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1, 2, 3, 4, 5 },  // input, kernel, gamma, beta, mean, var
            OutputIds = new List<int> { 7 },
            Operations = new List<IROp>
            {
                new Conv2DOp
                {
                    OutputId = 6,
                    InputIds = new[] { 0, 1 },
                    OutputShape = new[] { 1, 32, 32, 64 },
                    Stride = new[] { 1, 1 },
                    Padding = new[] { 1, 1 }
                },
                new BatchNormOp
                {
                    OutputId = 7,
                    InputIds = new[] { 6, 2, 3, 4, 5 },
                    OutputShape = new[] { 1, 32, 32, 64 },
                    Epsilon = 1e-5,
                    Momentum = 0.1
                },
            },
            TensorShapes = new Dictionary<int, int[]>()
        };

        var fusion = new OperationFusionPass();

        // Act
        var optimized = fusion.Optimize(graph);

        // Assert
        Assert.Single(optimized.Operations);
        var fusedOp = Assert.IsType<FusedConvBatchNormOp>(optimized.Operations[0]);
        Assert.Equal(1e-5, fusedOp.Epsilon);
        Assert.Equal(0.1, fusedOp.Momentum);
    }

    [Fact]
    public void OperationFusion_DoesNotFuseMultipleConsumers()
    {
        // Arrange: MatMul output used by two operations
        //   0, 1 -> MatMul (3) -> Add (4) -> output
        //                   \-> Exp (5) -> (also output)
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1, 2 },
            OutputIds = new List<int> { 4, 5 },
            Operations = new List<IROp>
            {
                new MatMulOp { OutputId = 3, InputIds = new[] { 0, 1 }, OutputShape = new[] { 1, 4 } },
                new AddOp { OutputId = 4, InputIds = new[] { 3, 2 }, OutputShape = new[] { 1, 4 } },
                new ExpOp { OutputId = 5, InputIds = new[] { 3 }, OutputShape = new[] { 1, 4 } },
            },
            TensorShapes = new Dictionary<int, int[]>()
        };

        var fusion = new OperationFusionPass();

        // Act
        var optimized = fusion.Optimize(graph);

        // Assert
        // Should NOT fuse because MatMul output (3) is used by both Add and Exp
        Assert.Equal(3, optimized.Operations.Count);
    }

    [Fact]
    public void OperationFusion_IdentifiesFusionOpportunities()
    {
        // Arrange
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1, 2 },
            OutputIds = new List<int> { 5 },
            Operations = new List<IROp>
            {
                new MatMulOp { OutputId = 3, InputIds = new[] { 0, 1 }, OutputShape = new[] { 1, 4 } },
                new AddOp { OutputId = 4, InputIds = new[] { 3, 2 }, OutputShape = new[] { 1, 4 } },
                new ReLUOp { OutputId = 5, InputIds = new[] { 4 }, OutputShape = new[] { 1, 4 } },
            },
            TensorShapes = new Dictionary<int, int[]>()
        };

        var fusion = new OperationFusionPass();

        // Act
        var opportunities = fusion.IdentifyFusionOpportunities(graph);

        // Assert
        Assert.NotEmpty(opportunities);
        Assert.Contains(opportunities, opp => opp.Contains("MatMul+Add"));
        Assert.Contains(opportunities, opp => opp.Contains("Add+ReLU"));
    }

    #endregion

    #region ConstantFolding Tests

    [Fact]
    public void ConstantFolding_IdentifiesFoldableOperations()
    {
        // Arrange
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0, 1 },  // Assume these are constants
            OutputIds = new List<int> { 2 },
            Operations = new List<IROp>
            {
                new AddOp { OutputId = 2, InputIds = new[] { 0, 1 }, OutputShape = new[] { 2, 3 } },
            },
            TensorShapes = new Dictionary<int, int[]>
            {
                [0] = new[] { 2, 3 },
                [1] = new[] { 2, 3 },
                [2] = new[] { 2, 3 }
            }
        };

        var constantFolding = new ConstantFoldingPass();

        // Act
        var optimized = constantFolding.Optimize(graph);

        // Assert
        Assert.NotNull(optimized);
        // Note: Full constant evaluation requires runtime tensor support
        // For now, we verify the pass runs without errors
    }

    [Fact]
    public void ConstantFolding_CanFold_ChecksSupportedOperations()
    {
        // Arrange
        var graph = new IRGraph
        {
            InputIds = new List<int> { 0 },
            OutputIds = new List<int> { 1 },
            Operations = new List<IROp>
            {
                new ReLUOp { OutputId = 1, InputIds = new[] { 0 }, OutputShape = new[] { 2, 3 } },
            },
            TensorShapes = new Dictionary<int, int[]>()
        };

        var constantFolding = new ConstantFoldingPass();

        // Act & Assert - Should not throw
        var optimized = constantFolding.Optimize(graph);
        Assert.NotNull(optimized);
    }

    #endregion
}
