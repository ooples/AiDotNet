using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;
using Xunit;

namespace AiDotNet.Tests.InferenceOptimization;

public class ComputationGraphTests
{
    [Fact]
    public void AddNode_ShouldAddNodeToGraph()
    {
        // Arrange
        var graph = new ComputationGraph<double>();
        var node = new ComputationNode<double>
        {
            OperationType = OperationType.ReLU,
            Name = "relu1"
        };

        // Act
        graph.AddNode(node);

        // Assert
        Assert.Contains(node, graph.Nodes);
        Assert.Single(graph.Nodes);
    }

    [Fact]
    public void RemoveNode_ShouldRemoveNodeFromGraph()
    {
        // Arrange
        var graph = new ComputationGraph<double>();
        var node = new ComputationNode<double>
        {
            OperationType = OperationType.ReLU,
            Name = "relu1"
        };
        graph.AddNode(node);

        // Act
        graph.RemoveNode(node);

        // Assert
        Assert.DoesNotContain(node, graph.Nodes);
        Assert.Empty(graph.Nodes);
    }

    [Fact]
    public void FindNodeById_ShouldReturnCorrectNode()
    {
        // Arrange
        var graph = new ComputationGraph<double>();
        var node = new ComputationNode<double>
        {
            Id = "test-id",
            OperationType = OperationType.ReLU,
            Name = "relu1"
        };
        graph.AddNode(node);

        // Act
        var found = graph.FindNodeById("test-id");

        // Assert
        Assert.NotNull(found);
        Assert.Equal(node, found);
    }

    [Fact]
    public void GetTopologicalOrder_ShouldReturnValidOrder()
    {
        // Arrange
        var graph = new ComputationGraph<double>();

        var input = new ComputationNode<double> { OperationType = OperationType.Input, Name = "input" };
        var conv = new ComputationNode<double> { OperationType = OperationType.Convolution, Name = "conv" };
        var relu = new ComputationNode<double> { OperationType = OperationType.ReLU, Name = "relu" };
        var output = new ComputationNode<double> { OperationType = OperationType.Output, Name = "output" };

        conv.AddInput(input);
        relu.AddInput(conv);
        output.AddInput(relu);

        graph.AddNode(input);
        graph.AddNode(conv);
        graph.AddNode(relu);
        graph.AddNode(output);

        // Act
        var order = graph.GetTopologicalOrder();

        // Assert
        Assert.Equal(4, order.Count);
        Assert.True(order.IndexOf(input) < order.IndexOf(conv));
        Assert.True(order.IndexOf(conv) < order.IndexOf(relu));
        Assert.True(order.IndexOf(relu) < order.IndexOf(output));
    }

    [Fact]
    public void Validate_ShouldReturnTrueForValidGraph()
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

        // Act
        var isValid = graph.Validate();

        // Assert
        Assert.True(isValid);
    }

    [Fact]
    public void Clone_ShouldCreateDeepCopy()
    {
        // Arrange
        var graph = new ComputationGraph<double>();

        var input = new ComputationNode<double> { OperationType = OperationType.Input, Name = "input" };
        var relu = new ComputationNode<double> { OperationType = OperationType.ReLU, Name = "relu" };

        relu.AddInput(input);

        graph.AddNode(input);
        graph.AddNode(relu);

        // Act
        var clonedResult = graph.Clone();
        var cloned = clonedResult as ComputationGraph<double>;

        // Assert
        Assert.NotNull(cloned);
        if (cloned != null)
        {
            Assert.Equal(graph.Nodes.Count, cloned.Nodes.Count);
            Assert.NotSame(graph.Nodes[0], cloned.Nodes[0]);
        }
    }

    [Fact]
    public void GetStatistics_ShouldReturnCorrectCounts()
    {
        // Arrange
        var graph = new ComputationGraph<double>();

        var input = new ComputationNode<double> { OperationType = OperationType.Input, Name = "input" };
        var relu1 = new ComputationNode<double> { OperationType = OperationType.ReLU, Name = "relu1" };
        var relu2 = new ComputationNode<double> { OperationType = OperationType.ReLU, Name = "relu2" };
        var output = new ComputationNode<double> { OperationType = OperationType.Output, Name = "output" };

        graph.AddNode(input);
        graph.AddNode(relu1);
        graph.AddNode(relu2);
        graph.AddNode(output);

        // Act
        var stats = graph.GetStatistics();

        // Assert
        Assert.Equal(4, stats.TotalNodes);
        Assert.Equal(1, stats.InputNodes);
        Assert.Equal(1, stats.OutputNodes);
        Assert.Equal(2, stats.OperationTypeCounts[OperationType.ReLU]);
    }
}
