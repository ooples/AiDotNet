using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;
using AiDotNet.InferenceOptimization.Passes;
using Xunit;

namespace AiDotNet.Tests.InferenceOptimization;

public class OptimizationPassTests
{
    [Fact]
    public void ConvBatchNormFusionPass_ShouldFuseConvAndBatchNorm()
    {
        // Arrange
        var graph = new ComputationGraph<double>();

        var input = new ComputationNode<double> { OperationType = OperationType.Input, Name = "input" };
        var conv = new ComputationNode<double> { OperationType = OperationType.Convolution, Name = "conv" };
        var bn = new ComputationNode<double> { OperationType = OperationType.BatchNormalization, Name = "bn" };
        var output = new ComputationNode<double> { OperationType = OperationType.Output, Name = "output" };

        conv.AddInput(input);
        bn.AddInput(conv);
        output.AddInput(bn);

        graph.AddNode(input);
        graph.AddNode(conv);
        graph.AddNode(bn);
        graph.AddNode(output);

        var pass = new ConvBatchNormFusionPass<double>();

        // Act
        var modified = pass.Apply(graph);

        // Assert
        Assert.True(modified);
        Assert.Contains(graph.Nodes, n => n.OperationType == OperationType.FusedConvBatchNorm);
        Assert.DoesNotContain(graph.Nodes, n => n.Name == "conv");
        Assert.DoesNotContain(graph.Nodes, n => n.Name == "bn");
    }

    [Fact]
    public void ConvBatchNormReLUFusionPass_ShouldFuseThreeOperations()
    {
        // Arrange
        var graph = new ComputationGraph<double>();

        var input = new ComputationNode<double> { OperationType = OperationType.Input, Name = "input" };
        var conv = new ComputationNode<double> { OperationType = OperationType.Convolution, Name = "conv" };
        var bn = new ComputationNode<double> { OperationType = OperationType.BatchNormalization, Name = "bn" };
        var relu = new ComputationNode<double> { OperationType = OperationType.ReLU, Name = "relu" };
        var output = new ComputationNode<double> { OperationType = OperationType.Output, Name = "output" };

        conv.AddInput(input);
        bn.AddInput(conv);
        relu.AddInput(bn);
        output.AddInput(relu);

        graph.AddNode(input);
        graph.AddNode(conv);
        graph.AddNode(bn);
        graph.AddNode(relu);
        graph.AddNode(output);

        var pass = new ConvBatchNormReLUFusionPass<double>();

        // Act
        var modified = pass.Apply(graph);

        // Assert
        Assert.True(modified);
        Assert.Contains(graph.Nodes, n => n.OperationType == OperationType.FusedConvBatchNormReLU);
    }

    [Fact]
    public void DeadCodeEliminationPass_ShouldRemoveUnusedNodes()
    {
        // Arrange
        var graph = new ComputationGraph<double>();

        var input = new ComputationNode<double> { OperationType = OperationType.Input, Name = "input" };
        var relu1 = new ComputationNode<double> { OperationType = OperationType.ReLU, Name = "relu1" };
        var relu2 = new ComputationNode<double> { OperationType = OperationType.ReLU, Name = "relu2" }; // Dead code
        var output = new ComputationNode<double> { OperationType = OperationType.Output, Name = "output" };

        relu1.AddInput(input);
        relu2.AddInput(input); // Not connected to output
        output.AddInput(relu1);

        graph.AddNode(input);
        graph.AddNode(relu1);
        graph.AddNode(relu2);
        graph.AddNode(output);

        var pass = new DeadCodeEliminationPass<double>();

        // Act
        var modified = pass.Apply(graph);

        // Assert
        Assert.True(modified);
        Assert.DoesNotContain(graph.Nodes, n => n.Name == "relu2");
        Assert.Contains(graph.Nodes, n => n.Name == "relu1");
    }

    [Fact]
    public void ConstantFoldingPass_ShouldFoldConstants()
    {
        // Arrange
        var graph = new ComputationGraph<double>();

        var const1 = new ComputationNode<double>
        {
            OperationType = OperationType.Constant,
            Name = "const1",
            ConstantValue = null // Would be actual tensor
        };

        var const2 = new ComputationNode<double>
        {
            OperationType = OperationType.Constant,
            Name = "const2",
            ConstantValue = null // Would be actual tensor
        };

        var add = new ComputationNode<double>
        {
            OperationType = OperationType.Add,
            Name = "add"
        };

        add.AddInput(const1);
        add.AddInput(const2);

        graph.AddNode(const1);
        graph.AddNode(const2);
        graph.AddNode(add);

        var pass = new ConstantFoldingPass<double>();

        // Act
        var canApply = pass.CanApply(graph);

        // Assert
        Assert.True(canApply);
        // Note: Actual folding would require tensor implementation
    }

    [Fact]
    public void InPlaceOptimizationPass_ShouldMarkEligibleOperations()
    {
        // Arrange
        var graph = new ComputationGraph<double>();

        var input = new ComputationNode<double> { OperationType = OperationType.Input, Name = "input" };
        var relu = new ComputationNode<double> { OperationType = OperationType.ReLU, Name = "relu" };
        var output = new ComputationNode<double> { OperationType = OperationType.Output, Name = "output" };

        relu.AddInput(input);
        output.AddInput(relu);

        graph.AddNode(input);
        graph.AddNode(relu);
        graph.AddNode(output);

        var pass = new InPlaceOptimizationPass<double>();

        // Act
        var modified = pass.Apply(graph);

        // Assert
        Assert.True(modified);
        Assert.True(relu.CanOperateInPlace);
    }

    [Fact]
    public void AlgebraicSimplificationPass_ShouldSimplifyIdentities()
    {
        // Arrange
        var graph = new ComputationGraph<double>();

        var input = new ComputationNode<double> { OperationType = OperationType.Input, Name = "input" };

        var zero = new ComputationNode<double>
        {
            OperationType = OperationType.Constant,
            Name = "zero",
            Metadata = new Dictionary<string, object> { ["IsZero"] = true }
        };

        var add = new ComputationNode<double> { OperationType = OperationType.Add, Name = "add" };
        var output = new ComputationNode<double> { OperationType = OperationType.Output, Name = "output" };

        add.AddInput(input);
        add.AddInput(zero);
        output.AddInput(add);

        graph.AddNode(input);
        graph.AddNode(zero);
        graph.AddNode(add);
        graph.AddNode(output);

        var pass = new AlgebraicSimplificationPass<double>();

        // Act
        var modified = pass.Apply(graph);

        // Assert
        Assert.True(modified);
        // x + 0 should be simplified to x
        Assert.DoesNotContain(graph.Nodes, n => n.Name == "add");
    }

    [Fact]
    public void GraphOptimizer_ShouldApplyMultiplePasses()
    {
        // Arrange
        var graph = new ComputationGraph<double>();

        var input = new ComputationNode<double> { OperationType = OperationType.Input, Name = "input" };
        var conv = new ComputationNode<double> { OperationType = OperationType.Convolution, Name = "conv" };
        var bn = new ComputationNode<double> { OperationType = OperationType.BatchNormalization, Name = "bn" };
        var relu = new ComputationNode<double> { OperationType = OperationType.ReLU, Name = "relu" };
        var output = new ComputationNode<double> { OperationType = OperationType.Output, Name = "output" };

        conv.AddInput(input);
        bn.AddInput(conv);
        relu.AddInput(bn);
        output.AddInput(relu);

        graph.AddNode(input);
        graph.AddNode(conv);
        graph.AddNode(bn);
        graph.AddNode(relu);
        graph.AddNode(output);

        var options = OptimizationOptions.FromLevel(OptimizationLevel.Standard);
        var optimizer = new GraphOptimizer<double>(options);

        // Act
        var optimizedGraph = optimizer.Optimize(graph);

        // Assert
        Assert.NotNull(optimizedGraph);
        // Should have fewer nodes due to fusion
        Assert.True(optimizedGraph.Nodes.Count < graph.Nodes.Count);
    }

    [Fact]
    public void OptimizationOptions_ShouldConfigureCorrectly()
    {
        // Arrange & Act
        var basicOptions = OptimizationOptions.FromLevel(OptimizationLevel.Basic);
        var standardOptions = OptimizationOptions.FromLevel(OptimizationLevel.Standard);
        var aggressiveOptions = OptimizationOptions.FromLevel(OptimizationLevel.Aggressive);

        // Assert
        Assert.False(basicOptions.EnableOperatorFusion);
        Assert.True(standardOptions.EnableOperatorFusion);
        Assert.True(aggressiveOptions.EnableMemoryReuse);
    }
}
