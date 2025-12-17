using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.JitCompiler;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using Xunit;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

/// <summary>
/// Tests for the IRBuilder class.
/// </summary>
public class IRBuilderTests
{
    [Fact]
    public void Build_SimpleAddOperation_CreatesCorrectIR()
    {
        // Arrange
        var input1 = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };
        var input2 = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };
        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input1, input2 })
        {
            OperationType = OperationType.Add
        };

        var builder = new IRBuilder();
        var inputs = new List<ComputationNode<float>> { input1, input2 };

        // Act
        var irGraph = builder.Build(result, inputs);

        // Assert
        Assert.NotNull(irGraph);
        Assert.Equal(2, irGraph.InputIds.Count);
        Assert.Single(irGraph.OutputIds);
        Assert.Single(irGraph.Operations);
        Assert.IsType<AddOp>(irGraph.Operations[0]);
    }

    [Fact]
    public void Build_LinearLayer_CreatesCorrectSequence()
    {
        // Arrange: result = Add(MatMul(input, weights), bias)
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 1, 3 }))
        {
            OperationType = OperationType.Input
        };
        var weights = new ComputationNode<float>(new Tensor<float>(new[] { 3, 4 }))
        {
            OperationType = OperationType.Input
        };
        var bias = new ComputationNode<float>(new Tensor<float>(new[] { 1, 4 }))
        {
            OperationType = OperationType.Input
        };

        var matmul = new ComputationNode<float>(
            new Tensor<float>(new[] { 1, 4 }),
            parents: new List<ComputationNode<float>> { input, weights })
        {
            OperationType = OperationType.MatMul
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 1, 4 }),
            parents: new List<ComputationNode<float>> { matmul, bias })
        {
            OperationType = OperationType.Add
        };

        var builder = new IRBuilder();
        var inputs = new List<ComputationNode<float>> { input, weights, bias };

        // Act
        var irGraph = builder.Build(result, inputs);

        // Assert
        Assert.NotNull(irGraph);
        Assert.Equal(3, irGraph.InputIds.Count);
        Assert.Single(irGraph.OutputIds);
        Assert.Equal(2, irGraph.Operations.Count);
        Assert.IsType<MatMulOp>(irGraph.Operations[0]);
        Assert.IsType<AddOp>(irGraph.Operations[1]);
    }

    [Fact]
    public void Build_MultipleOutputs_TracksAllOutputs()
    {
        // Arrange
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        var exp = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Exp
        };

        var log = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Log
        };

        var builder = new IRBuilder();

        // Act - build two separate graphs (simulating multi-output scenario)
        var irGraph1 = builder.Build(exp, new List<ComputationNode<float>> { input });
        builder = new IRBuilder(); // Reset for second build
        var irGraph2 = builder.Build(log, new List<ComputationNode<float>> { input });

        // Assert
        Assert.NotNull(irGraph1);
        Assert.NotNull(irGraph2);
        Assert.Single(irGraph1.Operations);
        Assert.Single(irGraph2.Operations);
        Assert.IsType<ExpOp>(irGraph1.Operations[0]);
        Assert.IsType<LogOp>(irGraph2.Operations[0]);
    }

    [Fact]
    public void Build_WithOperationParams_StoresParamsCorrectly()
    {
        // Arrange
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        var power = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Power,
            OperationParams = new Dictionary<string, object>
            {
                ["Exponent"] = 2.0
            }
        };

        var builder = new IRBuilder();

        // Act
        var irGraph = builder.Build(power, new List<ComputationNode<float>> { input });

        // Assert
        Assert.NotNull(irGraph);
        Assert.Single(irGraph.Operations);
        var powerOp = Assert.IsType<PowerOp>(irGraph.Operations[0]);
        Assert.Equal(2.0, powerOp.Exponent);
    }

    [Fact]
    public void Build_DAG_HandlesSharedNodes()
    {
        // Arrange: Diamond pattern - two paths from input to output
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        var exp = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Exp
        };

        var log = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Log
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { exp, log })
        {
            OperationType = OperationType.Add
        };

        var builder = new IRBuilder();

        // Act
        var irGraph = builder.Build(result, new List<ComputationNode<float>> { input });

        // Assert
        Assert.NotNull(irGraph);
        Assert.Single(irGraph.InputIds);
        Assert.Single(irGraph.OutputIds);
        Assert.Equal(3, irGraph.Operations.Count);  // Exp, Log, Add
    }

    [Fact]
    public void Build_WithoutOperationType_ThrowsException()
    {
        // Arrange
        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        var invalidNode = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            // OperationType not set!
        };

        var builder = new IRBuilder();

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() =>
            builder.Build(invalidNode, new List<ComputationNode<float>> { input }));
    }

    [Fact]
    public void Build_ComplexNetwork_CorrectTopologicalOrder()
    {
        // Arrange: input -> relu -> exp -> add <- log
        //                                    ^
        //                                    |
        //                             input -+

        var input = new ComputationNode<float>(new Tensor<float>(new[] { 2, 3 }))
        {
            OperationType = OperationType.Input
        };

        var relu = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.ReLU
        };

        var exp = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { relu })
        {
            OperationType = OperationType.Exp
        };

        var log = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { input })
        {
            OperationType = OperationType.Log
        };

        var result = new ComputationNode<float>(
            new Tensor<float>(new[] { 2, 3 }),
            parents: new List<ComputationNode<float>> { exp, log })
        {
            OperationType = OperationType.Add
        };

        var builder = new IRBuilder();

        // Act
        var irGraph = builder.Build(result, new List<ComputationNode<float>> { input });

        // Assert
        Assert.NotNull(irGraph);
        Assert.Equal(4, irGraph.Operations.Count);

        // Verify operations are in valid topological order
        // ReLU and Log can be in any order (both depend only on input)
        // Exp must come after ReLU
        // Add must come last
        var ops = irGraph.Operations;
        int reluIdx = ops.FindIndex(op => op is ReLUOp);
        int expIdx = ops.FindIndex(op => op is ExpOp);
        int logIdx = ops.FindIndex(op => op is LogOp);
        int addIdx = ops.FindIndex(op => op is AddOp);

        Assert.True(reluIdx >= 0 && expIdx > reluIdx);  // Exp after ReLU
        Assert.True(logIdx >= 0);
        Assert.True(addIdx == ops.Count - 1);  // Add is last
    }
}
