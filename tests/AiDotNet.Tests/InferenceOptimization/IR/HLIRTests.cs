using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.IR.Common;
using AiDotNet.InferenceOptimization.IR.HighLevel;
using Xunit;

namespace AiDotNet.Tests.InferenceOptimization.IR;

/// <summary>
/// Tests for High-Level IR classes.
/// </summary>
public class HLIRTests
{
    #region HLIRNode Tests

    [Fact]
    public void HLIRNode_DefaultValues_AreCorrect()
    {
        var node = new HLIRNode<double>();

        Assert.Equal(0, node.Id);
        Assert.Equal(string.Empty, node.Name);
        Assert.NotNull(node.Inputs);
        Assert.Empty(node.Inputs);
        Assert.NotNull(node.Outputs);
        Assert.Empty(node.Outputs);
        Assert.NotNull(node.OutputType);
        Assert.True(node.CanEliminate);
        Assert.False(node.IsFused);
        Assert.False(node.IsMarkedForDeletion);
    }

    [Fact]
    public void HLIRNode_AddInput_CreatesBidirectionalConnection()
    {
        var inputNode = new HLIRNode<double> { Id = 1, Name = "input" };
        var outputNode = new HLIRNode<double> { Id = 2, Name = "output" };

        outputNode.AddInput(inputNode);

        Assert.Contains(inputNode, outputNode.Inputs);
        Assert.Contains(outputNode, inputNode.Outputs);
    }

    [Fact]
    public void HLIRNode_AddInput_DoesNotDuplicate()
    {
        var inputNode = new HLIRNode<double> { Id = 1 };
        var outputNode = new HLIRNode<double> { Id = 2 };

        outputNode.AddInput(inputNode);
        outputNode.AddInput(inputNode);

        Assert.Single(outputNode.Inputs);
        Assert.Single(inputNode.Outputs);
    }

    [Fact]
    public void HLIRNode_RemoveInput_RemovesBidirectionalConnection()
    {
        var inputNode = new HLIRNode<double> { Id = 1 };
        var outputNode = new HLIRNode<double> { Id = 2 };

        outputNode.AddInput(inputNode);
        outputNode.RemoveInput(inputNode);

        Assert.DoesNotContain(inputNode, outputNode.Inputs);
        Assert.DoesNotContain(outputNode, inputNode.Outputs);
    }

    [Fact]
    public void HLIRNode_ReplaceInput_UpdatesConnections()
    {
        var oldInput = new HLIRNode<double> { Id = 1 };
        var newInput = new HLIRNode<double> { Id = 2 };
        var node = new HLIRNode<double> { Id = 3 };

        node.AddInput(oldInput);
        node.ReplaceInput(oldInput, newInput);

        Assert.DoesNotContain(oldInput, node.Inputs);
        Assert.Contains(newInput, node.Inputs);
        Assert.DoesNotContain(node, oldInput.Outputs);
        Assert.Contains(node, newInput.Outputs);
    }

    [Fact]
    public void HLIRNode_HasConsumers_ReturnsCorrectValue()
    {
        var node = new HLIRNode<double>();
        Assert.False(node.HasConsumers);

        var consumer = new HLIRNode<double>();
        consumer.AddInput(node);

        Assert.True(node.HasConsumers);
        Assert.Equal(1, node.ConsumerCount);
    }

    [Fact]
    public void HLIRNode_Validate_ReturnsTrueForValidNode()
    {
        var input = new HLIRNode<double> { Id = 1 };
        var output = new HLIRNode<double> { Id = 2, OutputType = new TensorType() };
        output.AddInput(input);

        Assert.True(output.Validate());
    }

    [Fact]
    public void HLIRNode_Validate_ReturnsFalseForInvalidId()
    {
        var node = new HLIRNode<double> { Id = -1 };
        Assert.False(node.Validate());
    }

    [Fact]
    public void HLIRNode_Clone_CreatesIndependentCopy()
    {
        var original = new HLIRNode<double>
        {
            Id = 1,
            Name = "test",
            Operation = OperationType.Add,
            CanEliminate = true,
            OutputType = new TensorType { Shape = new[] { 2, 3 } }
        };

        var clone = original.Clone();

        Assert.Equal(-1, clone.Id); // Clone gets new ID
        Assert.Contains("_clone", clone.Name);
        Assert.Equal(original.Operation, clone.Operation);
        Assert.Equal(original.CanEliminate, clone.CanEliminate);
    }

    [Fact]
    public void HLIRNode_AddProvenance_TracksHistory()
    {
        var node = new HLIRNode<double>();

        node.AddProvenance("Created");
        node.AddProvenance("Modified");

        Assert.Equal(2, node.Provenance.Count);
        Assert.Contains("Created", node.Provenance[0]);
        Assert.Contains("Modified", node.Provenance[1]);
    }

    [Fact]
    public void HLIRNode_ToString_ReturnsFormattedString()
    {
        var node = new HLIRNode<double>
        {
            Id = 1,
            Name = "relu1",
            Operation = OperationType.ReLU
        };

        var str = node.ToString();
        Assert.Contains("n1", str);
        Assert.Contains("relu1", str);
        Assert.Contains("ReLU", str);
    }

    #endregion

    #region HLIRGraph Tests

    [Fact]
    public void HLIRGraph_AddNode_AssignsId()
    {
        var graph = new HLIRGraph<double>();
        var node = new HLIRNode<double> { Id = -1, Name = "test" };

        graph.AddNode(node);

        Assert.True(node.Id >= 0);
        Assert.Equal(1, graph.NodeCount);
    }

    [Fact]
    public void HLIRGraph_AddNode_ThrowsOnDuplicateId()
    {
        var graph = new HLIRGraph<double>();
        var node1 = new HLIRNode<double> { Id = 5 };
        var node2 = new HLIRNode<double> { Id = 5 };

        graph.AddNode(node1);

        Assert.Throws<InvalidOperationException>(() => graph.AddNode(node2));
    }

    [Fact]
    public void HLIRGraph_CreateNode_CreatesAndAddsNode()
    {
        var graph = new HLIRGraph<double>();
        var input = graph.CreateNode(OperationType.Input, "input");
        var relu = graph.CreateNode(OperationType.ReLU, "relu", input);

        Assert.Equal(2, graph.NodeCount);
        Assert.Contains(input, relu.Inputs);
    }

    [Fact]
    public void HLIRGraph_RemoveNode_RemovesNodeAndConnections()
    {
        var graph = new HLIRGraph<double>();
        var input = graph.CreateNode(OperationType.Input, "input");
        var middle = graph.CreateNode(OperationType.ReLU, "middle", input);
        var output = graph.CreateNode(OperationType.Output, "output", middle);

        graph.RemoveNode(middle);

        Assert.Equal(2, graph.NodeCount);
        Assert.DoesNotContain(middle, output.Inputs);
        Assert.DoesNotContain(output, input.Outputs);
    }

    [Fact]
    public void HLIRGraph_FindNode_ReturnsCorrectNode()
    {
        var graph = new HLIRGraph<double>();
        var node = graph.CreateNode(OperationType.Input, "test");

        Assert.Same(node, graph.FindNode(node.Id));
        Assert.Null(graph.FindNode(999));
    }

    [Fact]
    public void HLIRGraph_FindNodesByName_ReturnsMatchingNodes()
    {
        var graph = new HLIRGraph<double>();
        graph.CreateNode(OperationType.ReLU, "relu1");
        graph.CreateNode(OperationType.ReLU, "relu2");
        graph.CreateNode(OperationType.Add, "add1");

        var reluNodes = graph.FindNodesByName("relu").ToList();

        Assert.Equal(2, reluNodes.Count);
    }

    [Fact]
    public void HLIRGraph_FindNodesByOperation_ReturnsMatchingNodes()
    {
        var graph = new HLIRGraph<double>();
        graph.CreateNode(OperationType.ReLU, "relu1");
        graph.CreateNode(OperationType.ReLU, "relu2");
        graph.CreateNode(OperationType.Add, "add1");

        var reluNodes = graph.FindNodesByOperation(OperationType.ReLU).ToList();

        Assert.Equal(2, reluNodes.Count);
    }

    [Fact]
    public void HLIRGraph_GetTopologicalOrder_ReturnsCorrectOrder()
    {
        var graph = new HLIRGraph<double>();
        var input = graph.CreateNode(OperationType.Input, "input");
        var relu = graph.CreateNode(OperationType.ReLU, "relu", input);
        var output = graph.CreateNode(OperationType.Output, "output", relu);

        graph.InputNodes.Add(input);
        graph.OutputNodes.Add(output);

        var order = graph.GetTopologicalOrder();

        Assert.Equal(3, order.Count);
        Assert.True(order.IndexOf(input) < order.IndexOf(relu));
        Assert.True(order.IndexOf(relu) < order.IndexOf(output));
    }

    [Fact]
    public void HLIRGraph_GetTopologicalOrder_DetectsCycle()
    {
        var graph = new HLIRGraph<double>();
        var node1 = graph.CreateNode(OperationType.Add, "node1");
        var node2 = graph.CreateNode(OperationType.Add, "node2", node1);

        // Create cycle
        node1.AddInput(node2);

        Assert.Throws<InvalidOperationException>(() => graph.GetTopologicalOrder());
    }

    [Fact]
    public void HLIRGraph_Validate_ReturnsTrueForValidGraph()
    {
        var graph = new HLIRGraph<double>();
        var input = graph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 2, 3 } };
        var output = graph.CreateNode(OperationType.Output, "output", input);
        output.OutputType = new TensorType { Shape = new[] { 2, 3 } };

        graph.InputNodes.Add(input);
        graph.OutputNodes.Add(output);

        var result = graph.Validate();

        Assert.True(result.IsValid);
        Assert.Empty(result.Errors);
    }

    [Fact]
    public void HLIRGraph_Clone_CreatesIndependentCopy()
    {
        var graph = new HLIRGraph<double> { Name = "original" };
        var input = graph.CreateNode(OperationType.Input, "input");
        var output = graph.CreateNode(OperationType.Output, "output", input);
        graph.InputNodes.Add(input);
        graph.OutputNodes.Add(output);

        var clone = graph.Clone();

        Assert.Equal(graph.NodeCount, clone.NodeCount);
        Assert.Contains("_clone", clone.Name);

        // Verify cloned nodes are different objects
        var originalNodes = graph.Nodes.ToList();
        var clonedNodes = clone.Nodes.ToList();
        Assert.NotSame(originalNodes[0], clonedNodes[0]);
    }

    [Fact]
    public void HLIRGraph_GetStatistics_ReturnsCorrectStats()
    {
        var graph = new HLIRGraph<double>();
        var input = graph.CreateNode(OperationType.Input, "input");
        var relu1 = graph.CreateNode(OperationType.ReLU, "relu1", input);
        var relu2 = graph.CreateNode(OperationType.ReLU, "relu2", relu1);
        var output = graph.CreateNode(OperationType.Output, "output", relu2);

        graph.InputNodes.Add(input);
        graph.OutputNodes.Add(output);

        var stats = graph.GetStatistics();

        Assert.Equal(4, stats.TotalNodes);
        Assert.Equal(1, stats.InputNodes);
        Assert.Equal(1, stats.OutputNodes);
        Assert.Equal(2, stats.NodesByOperation[OperationType.ReLU]);
    }

    [Fact]
    public void HLIRGraph_FindPatterns_DetectsSequentialPattern()
    {
        var graph = new HLIRGraph<double>();
        var input = graph.CreateNode(OperationType.Input, "input");
        var conv = graph.CreateNode(OperationType.Conv2D, "conv", input);
        var bn = graph.CreateNode(OperationType.BatchNorm, "bn", conv);
        var relu = graph.CreateNode(OperationType.ReLU, "relu", bn);

        var patterns = graph.FindPatterns(OperationType.Conv2D, OperationType.BatchNorm, OperationType.ReLU);

        Assert.Single(patterns);
        Assert.Equal(3, patterns[0].Count);
    }

    [Fact]
    public void HLIRGraph_ReplaceNode_UpdatesAllConnections()
    {
        var graph = new HLIRGraph<double>();
        var input = graph.CreateNode(OperationType.Input, "input");
        var oldNode = graph.CreateNode(OperationType.ReLU, "old", input);
        var output = graph.CreateNode(OperationType.Output, "output", oldNode);

        var newNode = new HLIRNode<double>
        {
            Id = -1,
            Name = "new",
            Operation = OperationType.GELU,
            OutputType = new TensorType()
        };

        graph.ReplaceNode(oldNode, newNode);

        Assert.Contains(input, newNode.Inputs);
        Assert.Contains(newNode, output.Inputs);
        Assert.DoesNotContain(oldNode, output.Inputs);
    }

    [Fact]
    public void HLIRGraph_CompactNodeIds_ReassignsIdsSequentially()
    {
        var graph = new HLIRGraph<double>();
        var node1 = new HLIRNode<double> { Id = 100 };
        var node2 = new HLIRNode<double> { Id = 200 };
        var node3 = new HLIRNode<double> { Id = 300 };

        graph.AddNode(node1);
        graph.AddNode(node2);
        node2.AddInput(node1);
        graph.AddNode(node3);
        node3.AddInput(node2);

        graph.InputNodes.Add(node1);
        graph.OutputNodes.Add(node3);

        graph.CompactNodeIds();

        var ids = graph.Nodes.Select(n => n.Id).ToList();
        Assert.Contains(0, ids);
        Assert.Contains(1, ids);
        Assert.Contains(2, ids);
    }

    #endregion

    #region OperationCost Tests

    [Fact]
    public void OperationCost_ArithmeticIntensity_CalculatesCorrectly()
    {
        var cost = new OperationCost
        {
            FLOPs = 1000,
            MemoryRead = 50,
            MemoryWrite = 50
        };

        Assert.Equal(10.0, cost.ArithmeticIntensity);
    }

    [Fact]
    public void OperationCost_IsMemoryBound_DetectsCorrectly()
    {
        var memBound = new OperationCost { FLOPs = 100, MemoryRead = 100, MemoryWrite = 100 };
        var computeBound = new OperationCost { FLOPs = 10000, MemoryRead = 100, MemoryWrite = 100 };

        Assert.True(memBound.IsMemoryBound);
        Assert.False(computeBound.IsMemoryBound);
    }

    #endregion

    #region OptimizationHints Tests

    [Fact]
    public void OptimizationHints_DefaultValues_AreCorrect()
    {
        var hints = new OptimizationHints();

        Assert.Equal(DeviceType.Auto, hints.PreferredDevice);
        Assert.False(hints.PrioritizeMemory);
        Assert.False(hints.PrioritizeLatency);
        Assert.True(hints.IsFusionCandidate);
        Assert.True(hints.EnableVectorization);
        Assert.True(hints.EnableParallelization);
    }

    [Fact]
    public void OptimizationHints_Clone_CreatesIndependentCopy()
    {
        var original = new OptimizationHints
        {
            PreferredDevice = DeviceType.GPU,
            TileSizes = new[] { 32, 32 },
            EnableVectorization = false
        };

        var clone = original.Clone();

        Assert.Equal(original.PreferredDevice, clone.PreferredDevice);
        Assert.Equal(original.TileSizes, clone.TileSizes);
        Assert.Equal(original.EnableVectorization, clone.EnableVectorization);

        // Modify clone and verify original unchanged
        clone.TileSizes![0] = 64;
        Assert.NotEqual(original.TileSizes[0], clone.TileSizes[0]);
    }

    #endregion
}
