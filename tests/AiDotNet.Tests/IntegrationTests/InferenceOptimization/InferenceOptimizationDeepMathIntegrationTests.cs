using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.InferenceOptimization;

/// <summary>
/// Deep integration tests for OptimizationGraph and OptimizationNode:
/// graph construction, topological ordering, cycle detection, node operations,
/// statistics, cloning, validation, and graph topology invariants.
/// </summary>
public class InferenceOptimizationDeepMathIntegrationTests
{
    // ============================
    // Helper Methods
    // ============================

    private static OptimizationNode<double> CreateNode(string name, OperationType opType = OperationType.MatMul)
    {
        return new OptimizationNode<double>
        {
            Name = name,
            OperationType = opType,
            OutputShape = new[] { 1, 10 }
        };
    }

    /// <summary>
    /// Creates a simple chain: Input -> MatMul -> ReLU -> Output
    /// </summary>
    private static OptimizationGraph<double> CreateChainGraph()
    {
        var graph = new OptimizationGraph<double>();

        var input = CreateNode("input", OperationType.Input);
        var matmul = CreateNode("matmul", OperationType.MatMul);
        var relu = CreateNode("relu", OperationType.ReLU);
        var output = CreateNode("output", OperationType.Output);

        matmul.AddInput(input);
        relu.AddInput(matmul);
        output.AddInput(relu);

        graph.AddNode(input);
        graph.AddNode(matmul);
        graph.AddNode(relu);
        graph.AddNode(output);

        return graph;
    }

    /// <summary>
    /// Creates a diamond graph: Input -> {A, B} -> Output
    /// </summary>
    private static OptimizationGraph<double> CreateDiamondGraph()
    {
        var graph = new OptimizationGraph<double>();

        var input = CreateNode("input", OperationType.Input);
        var branchA = CreateNode("branchA", OperationType.MatMul);
        var branchB = CreateNode("branchB", OperationType.Convolution);
        var merge = CreateNode("merge", OperationType.Add);
        var output = CreateNode("output", OperationType.Output);

        branchA.AddInput(input);
        branchB.AddInput(input);
        merge.AddInput(branchA);
        merge.AddInput(branchB);
        output.AddInput(merge);

        graph.AddNode(input);
        graph.AddNode(branchA);
        graph.AddNode(branchB);
        graph.AddNode(merge);
        graph.AddNode(output);

        return graph;
    }

    // ============================
    // Graph Construction Tests
    // ============================

    [Fact]
    public void EmptyGraph_HasNoNodes()
    {
        var graph = new OptimizationGraph<double>();

        Assert.Empty(graph.Nodes);
        Assert.Empty(graph.InputNodes);
        Assert.Empty(graph.OutputNodes);
    }

    [Fact]
    public void AddNode_IncreasesNodeCount()
    {
        var graph = new OptimizationGraph<double>();
        var node = CreateNode("test");

        graph.AddNode(node);

        Assert.Single(graph.Nodes);
    }

    [Fact]
    public void AddNode_InputType_TrackedAsInput()
    {
        var graph = new OptimizationGraph<double>();
        var input = CreateNode("input", OperationType.Input);

        graph.AddNode(input);

        Assert.Single(graph.InputNodes);
        Assert.Contains(input, graph.InputNodes);
    }

    [Fact]
    public void AddNode_OutputType_TrackedAsOutput()
    {
        var graph = new OptimizationGraph<double>();
        var output = CreateNode("output", OperationType.Output);

        graph.AddNode(output);

        Assert.Single(graph.OutputNodes);
        Assert.Contains(output, graph.OutputNodes);
    }

    [Fact]
    public void AddNode_DuplicateId_NotAdded()
    {
        var graph = new OptimizationGraph<double>();
        var node = CreateNode("test");

        graph.AddNode(node);
        graph.AddNode(node); // Same node (same Id)

        Assert.Single(graph.Nodes);
    }

    [Fact]
    public void AddNode_NullNode_Throws()
    {
        var graph = new OptimizationGraph<double>();
        Assert.Throws<ArgumentNullException>(() => graph.AddNode(null!));
    }

    // ============================
    // Chain Graph Tests
    // ============================

    [Fact]
    public void ChainGraph_NodeCount()
    {
        var graph = CreateChainGraph();

        Assert.Equal(4, graph.Nodes.Count);
        Assert.Single(graph.InputNodes);
        Assert.Single(graph.OutputNodes);
    }

    [Fact]
    public void ChainGraph_TopologicalOrder_InputFirst()
    {
        var graph = CreateChainGraph();
        var order = graph.GetTopologicalOrder();

        Assert.Equal(4, order.Count);
        // Input should come before matmul, matmul before relu, relu before output
        var inputIdx = order.FindIndex(n => n.Name == "input");
        var matmulIdx = order.FindIndex(n => n.Name == "matmul");
        var reluIdx = order.FindIndex(n => n.Name == "relu");
        var outputIdx = order.FindIndex(n => n.Name == "output");

        Assert.True(inputIdx < matmulIdx);
        Assert.True(matmulIdx < reluIdx);
        Assert.True(reluIdx < outputIdx);
    }

    [Fact]
    public void ChainGraph_Validates()
    {
        var graph = CreateChainGraph();
        Assert.True(graph.Validate());
    }

    // ============================
    // Diamond Graph Tests
    // ============================

    [Fact]
    public void DiamondGraph_NodeCount()
    {
        var graph = CreateDiamondGraph();

        Assert.Equal(5, graph.Nodes.Count);
        Assert.Single(graph.InputNodes);
        Assert.Single(graph.OutputNodes);
    }

    [Fact]
    public void DiamondGraph_TopologicalOrder_Valid()
    {
        var graph = CreateDiamondGraph();
        var order = graph.GetTopologicalOrder();

        Assert.Equal(5, order.Count);

        var inputIdx = order.FindIndex(n => n.Name == "input");
        var branchAIdx = order.FindIndex(n => n.Name == "branchA");
        var branchBIdx = order.FindIndex(n => n.Name == "branchB");
        var mergeIdx = order.FindIndex(n => n.Name == "merge");
        var outputIdx = order.FindIndex(n => n.Name == "output");

        // Input must come before both branches
        Assert.True(inputIdx < branchAIdx);
        Assert.True(inputIdx < branchBIdx);

        // Both branches must come before merge
        Assert.True(branchAIdx < mergeIdx);
        Assert.True(branchBIdx < mergeIdx);

        // Merge must come before output
        Assert.True(mergeIdx < outputIdx);
    }

    [Fact]
    public void DiamondGraph_Validates()
    {
        var graph = CreateDiamondGraph();
        Assert.True(graph.Validate());
    }

    // ============================
    // Node Connection Tests
    // ============================

    [Fact]
    public void AddInput_CreatesEdgeBothWays()
    {
        var a = CreateNode("a");
        var b = CreateNode("b");

        b.AddInput(a);

        Assert.Contains(a, b.Inputs);
        Assert.Contains(b, a.Outputs);
    }

    [Fact]
    public void AddInput_DuplicateIgnored()
    {
        var a = CreateNode("a");
        var b = CreateNode("b");

        b.AddInput(a);
        b.AddInput(a); // duplicate

        Assert.Single(b.Inputs);
        Assert.Single(a.Outputs);
    }

    [Fact]
    public void RemoveInput_CleansUpBothWays()
    {
        var a = CreateNode("a");
        var b = CreateNode("b");

        b.AddInput(a);
        b.RemoveInput(a);

        Assert.Empty(b.Inputs);
        Assert.Empty(a.Outputs);
    }

    [Fact]
    public void ReplaceInput_SwapsEdge()
    {
        var a = CreateNode("a");
        var b = CreateNode("b");
        var c = CreateNode("c");

        c.AddInput(a);
        c.ReplaceInput(a, b);

        Assert.Contains(b, c.Inputs);
        Assert.DoesNotContain(a, c.Inputs);
        Assert.Contains(c, b.Outputs);
        Assert.DoesNotContain(c, a.Outputs);
    }

    [Fact]
    public void AddInput_NullNode_Throws()
    {
        var node = CreateNode("test");
        Assert.Throws<ArgumentNullException>(() => node.AddInput(null!));
    }

    // ============================
    // Node Query Tests
    // ============================

    [Fact]
    public void HasConsumers_WithOutputs_True()
    {
        var a = CreateNode("a");
        var b = CreateNode("b");

        b.AddInput(a);

        Assert.True(a.HasConsumers());
    }

    [Fact]
    public void HasConsumers_WithoutOutputs_False()
    {
        var a = CreateNode("a");
        Assert.False(a.HasConsumers());
    }

    [Fact]
    public void ConsumerCount_HandVerified()
    {
        var a = CreateNode("a");
        var b = CreateNode("b");
        var c = CreateNode("c");

        b.AddInput(a);
        c.AddInput(a);

        Assert.Equal(2, a.ConsumerCount());
    }

    // ============================
    // FindNode Tests
    // ============================

    [Fact]
    public void FindNodeById_ExistingNode_Found()
    {
        var graph = new OptimizationGraph<double>();
        var node = CreateNode("test");

        graph.AddNode(node);

        var found = graph.FindNodeById(node.Id);
        Assert.NotNull(found);
        Assert.Equal("test", found.Name);
    }

    [Fact]
    public void FindNodeById_NonExisting_ReturnsNull()
    {
        var graph = new OptimizationGraph<double>();
        var found = graph.FindNodeById("nonexistent");
        Assert.Null(found);
    }

    [Fact]
    public void FindNodeById_NullId_Throws()
    {
        var graph = new OptimizationGraph<double>();
        Assert.Throws<ArgumentNullException>(() => graph.FindNodeById(null!));
    }

    [Fact]
    public void FindNodesByName_ExistingName_Found()
    {
        var graph = new OptimizationGraph<double>();
        var node1 = CreateNode("conv1", OperationType.Convolution);
        var node2 = CreateNode("conv1", OperationType.Convolution);

        graph.AddNode(node1);
        graph.AddNode(node2);

        var found = graph.FindNodesByName("conv1");
        Assert.Equal(2, found.Count);
    }

    [Fact]
    public void FindNodesByName_NonExisting_Empty()
    {
        var graph = new OptimizationGraph<double>();
        var found = graph.FindNodesByName("nonexistent");
        Assert.Empty(found);
    }

    // ============================
    // Remove Node Tests
    // ============================

    [Fact]
    public void RemoveNode_DecreasesCount()
    {
        var graph = new OptimizationGraph<double>();
        var node = CreateNode("test");

        graph.AddNode(node);
        Assert.Single(graph.Nodes);

        graph.RemoveNode(node);
        Assert.Empty(graph.Nodes);
    }

    [Fact]
    public void RemoveNode_CleansUpConnections()
    {
        var graph = new OptimizationGraph<double>();
        var a = CreateNode("a", OperationType.Input);
        var b = CreateNode("b", OperationType.MatMul);
        var c = CreateNode("c", OperationType.Output);

        b.AddInput(a);
        c.AddInput(b);
        graph.AddNode(a);
        graph.AddNode(b);
        graph.AddNode(c);

        // Remove middle node
        graph.RemoveNode(b);

        Assert.DoesNotContain(b, a.Outputs);
        Assert.DoesNotContain(b, c.Inputs);
    }

    // ============================
    // Graph Statistics Tests
    // ============================

    [Fact]
    public void Statistics_ChainGraph_HandVerified()
    {
        var graph = CreateChainGraph();
        var stats = graph.GetStatistics();

        Assert.Equal(4, stats.TotalNodes);
        Assert.Equal(1, stats.InputNodes);
        Assert.Equal(1, stats.OutputNodes);
        Assert.Equal(0, stats.FusedNodes);
    }

    [Fact]
    public void Statistics_DiamondGraph_OperationCounts()
    {
        var graph = CreateDiamondGraph();
        var stats = graph.GetStatistics();

        Assert.Equal(5, stats.TotalNodes);
        Assert.True(stats.OperationTypeCounts.ContainsKey(OperationType.Input));
        Assert.True(stats.OperationTypeCounts.ContainsKey(OperationType.MatMul));
        Assert.True(stats.OperationTypeCounts.ContainsKey(OperationType.Convolution));
        Assert.True(stats.OperationTypeCounts.ContainsKey(OperationType.Add));
        Assert.True(stats.OperationTypeCounts.ContainsKey(OperationType.Output));

        Assert.Equal(1, stats.OperationTypeCounts[OperationType.Input]);
        Assert.Equal(1, stats.OperationTypeCounts[OperationType.MatMul]);
        Assert.Equal(1, stats.OperationTypeCounts[OperationType.Convolution]);
        Assert.Equal(1, stats.OperationTypeCounts[OperationType.Add]);
        Assert.Equal(1, stats.OperationTypeCounts[OperationType.Output]);
    }

    [Fact]
    public void Statistics_WithFusedNode()
    {
        var graph = new OptimizationGraph<double>();
        var node = CreateNode("fused", OperationType.MatMul);
        node.IsFused = true;

        graph.AddNode(node);

        var stats = graph.GetStatistics();
        Assert.Equal(1, stats.FusedNodes);
    }

    // ============================
    // Node Clone Tests
    // ============================

    [Fact]
    public void Clone_CopiesProperties()
    {
        var original = CreateNode("conv1", OperationType.Convolution);
        original.OutputShape = new[] { 32, 28, 28 };
        original.Parameters["weight"] = "test";
        original.CanEliminate = false;

        var clone = original.Clone();

        Assert.NotEqual(original.Id, clone.Id); // New Id
        Assert.Equal("conv1_clone", clone.Name);
        Assert.Equal(OperationType.Convolution, clone.OperationType);
        Assert.Equal(new[] { 32, 28, 28 }, clone.OutputShape);
        Assert.False(clone.CanEliminate);
    }

    [Fact]
    public void Clone_NoSharedConnections()
    {
        var a = CreateNode("a");
        var b = CreateNode("b");
        b.AddInput(a);

        var clone = b.Clone();

        // Clone should have no connections
        Assert.Empty(clone.Inputs);
        Assert.Empty(clone.Outputs);
    }

    // ============================
    // Graph Clone Tests
    // ============================

    [Fact]
    public void GraphClone_PreservesStructure()
    {
        var original = CreateChainGraph();
        var clone = (OptimizationGraph<double>)original.Clone();

        Assert.Equal(original.Nodes.Count, clone.Nodes.Count);
        Assert.Equal(original.InputNodes.Count, clone.InputNodes.Count);
        Assert.Equal(original.OutputNodes.Count, clone.OutputNodes.Count);
    }

    [Fact]
    public void GraphClone_TopologicalOrderPreserved()
    {
        var original = CreateChainGraph();
        var clone = (OptimizationGraph<double>)original.Clone();

        var cloneOrder = clone.GetTopologicalOrder();
        Assert.Equal(4, cloneOrder.Count);
    }

    [Fact]
    public void GraphClone_IsIndependent()
    {
        var original = CreateChainGraph();
        var clone = (OptimizationGraph<double>)original.Clone();

        // Modifying original should not affect clone
        var newNode = CreateNode("extra");
        original.AddNode(newNode);

        Assert.Equal(5, original.Nodes.Count);
        Assert.Equal(4, clone.Nodes.Count);
    }

    // ============================
    // Validation Tests
    // ============================

    [Fact]
    public void Validate_EmptyGraph_Valid()
    {
        var graph = new OptimizationGraph<double>();
        Assert.True(graph.Validate());
    }

    [Fact]
    public void Validate_DisconnectedNode_Invalid()
    {
        var graph = new OptimizationGraph<double>();
        var input = CreateNode("input", OperationType.Input);
        var disconnected = CreateNode("disconnected", OperationType.MatMul);

        graph.AddNode(input);
        graph.AddNode(disconnected);

        // disconnected is not reachable from input
        Assert.False(graph.Validate());
    }

    [Fact]
    public void Validate_ConstantNodes_AllowedDisconnected()
    {
        var graph = new OptimizationGraph<double>();
        var input = CreateNode("input", OperationType.Input);
        var constant = CreateNode("const", OperationType.Constant);

        graph.AddNode(input);
        graph.AddNode(constant);

        // Constants are allowed to be disconnected
        Assert.True(graph.Validate());
    }

    // ============================
    // Node Default Properties Tests
    // ============================

    [Fact]
    public void NewNode_DefaultProperties()
    {
        var node = new OptimizationNode<double>();

        Assert.NotNull(node.Id);
        Assert.NotEmpty(node.Id);
        Assert.Equal(string.Empty, node.Name);
        Assert.Empty(node.Inputs);
        Assert.Empty(node.Outputs);
        Assert.Empty(node.OutputShape);
        Assert.Empty(node.Parameters);
        Assert.Empty(node.Metadata);
        Assert.True(node.CanEliminate);
        Assert.False(node.CanOperateInPlace);
        Assert.False(node.IsMarkedForDeletion);
        Assert.False(node.IsFused);
        Assert.Null(node.ConstantValue);
        Assert.Null(node.OriginalLayer);
        Assert.Null(node.FusedFrom);
    }

    // ============================
    // Graph Topology Invariants
    // ============================

    [Fact]
    public void TopologicalOrder_EveryNodePrecedesItsConsumers()
    {
        var graph = CreateDiamondGraph();
        var order = graph.GetTopologicalOrder();

        var positionMap = new Dictionary<string, int>();
        for (int i = 0; i < order.Count; i++)
            positionMap[order[i].Id] = i;

        // For every edge A -> B: position(A) < position(B)
        foreach (var node in graph.Nodes)
        {
            foreach (var output in node.Outputs)
            {
                Assert.True(positionMap[node.Id] < positionMap[output.Id],
                    $"Node {node.Name} should come before {output.Name} in topological order");
            }
        }
    }

    [Fact]
    public void TopologicalOrder_AllNodesIncluded()
    {
        var graph = CreateChainGraph();
        var order = graph.GetTopologicalOrder();

        Assert.Equal(graph.Nodes.Count, order.Count);

        var orderIds = new HashSet<string>(order.Select(n => n.Id));
        foreach (var node in graph.Nodes)
            Assert.Contains(node.Id, orderIds);
    }

    [Fact]
    public void Edges_AreBidirectional()
    {
        // For every node, if B is in A.Outputs, then A should be in B.Inputs
        var graph = CreateDiamondGraph();

        foreach (var node in graph.Nodes)
        {
            foreach (var output in node.Outputs)
                Assert.Contains(node, output.Inputs);

            foreach (var input in node.Inputs)
                Assert.Contains(node, input.Outputs);
        }
    }

    // ============================
    // ToString Tests
    // ============================

    [Fact]
    public void Node_ToString_ContainsInfo()
    {
        var node = CreateNode("conv1", OperationType.Convolution);
        node.OutputShape = new[] { 32, 28, 28 };

        var str = node.ToString();
        Assert.Contains("conv1", str);
        Assert.Contains("Convolution", str);
    }

    [Fact]
    public void Graph_ToString_ContainsInfo()
    {
        var graph = CreateChainGraph();
        var str = graph.ToString();

        Assert.Contains("4", str); // 4 nodes
    }
}
