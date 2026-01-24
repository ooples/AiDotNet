using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.Core;
using AiDotNet.InferenceOptimization.IR.Common;
using AiDotNet.InferenceOptimization.Passes;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.InferenceOptimization;

/// <summary>
/// Integration tests for the InferenceOptimization module.
/// Tests cover optimization graphs, nodes, passes, IR types, and various optimization strategies.
/// </summary>
public class InferenceOptimizationIntegrationTests
{
    #region OptimizationNode Tests

    [Fact]
    public void OptimizationNode_Constructor_SetsDefaults()
    {
        var node = new OptimizationNode<double>();

        Assert.NotNull(node.Id);
        Assert.Empty(node.Name);
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
        Assert.Null(node.FusedFrom);
    }

    [Fact]
    public void OptimizationNode_AddInput_EstablishesConnection()
    {
        var node1 = new OptimizationNode<double> { Name = "input" };
        var node2 = new OptimizationNode<double> { Name = "output" };

        node2.AddInput(node1);

        Assert.Single(node2.Inputs);
        Assert.Contains(node1, node2.Inputs);
        Assert.Single(node1.Outputs);
        Assert.Contains(node2, node1.Outputs);
    }

    [Fact]
    public void OptimizationNode_AddInput_DoesNotDuplicate()
    {
        var node1 = new OptimizationNode<double>();
        var node2 = new OptimizationNode<double>();

        node2.AddInput(node1);
        node2.AddInput(node1); // Add same node twice

        Assert.Single(node2.Inputs);
        Assert.Single(node1.Outputs);
    }

    [Fact]
    public void OptimizationNode_RemoveInput_RemovesConnection()
    {
        var node1 = new OptimizationNode<double>();
        var node2 = new OptimizationNode<double>();

        node2.AddInput(node1);
        node2.RemoveInput(node1);

        Assert.Empty(node2.Inputs);
        Assert.Empty(node1.Outputs);
    }

    [Fact]
    public void OptimizationNode_ReplaceInput_UpdatesConnection()
    {
        var oldInput = new OptimizationNode<double> { Name = "old" };
        var newInput = new OptimizationNode<double> { Name = "new" };
        var node = new OptimizationNode<double> { Name = "consumer" };

        node.AddInput(oldInput);
        node.ReplaceInput(oldInput, newInput);

        Assert.Single(node.Inputs);
        Assert.Contains(newInput, node.Inputs);
        Assert.DoesNotContain(oldInput, node.Inputs);
        Assert.Empty(oldInput.Outputs);
        Assert.Single(newInput.Outputs);
    }

    [Fact]
    public void OptimizationNode_HasConsumers_ReturnsTrueWhenOutputsExist()
    {
        var node1 = new OptimizationNode<double>();
        var node2 = new OptimizationNode<double>();

        Assert.False(node1.HasConsumers());

        node2.AddInput(node1);

        Assert.True(node1.HasConsumers());
    }

    [Fact]
    public void OptimizationNode_ConsumerCount_ReturnsCorrectCount()
    {
        var producer = new OptimizationNode<double>();
        var consumer1 = new OptimizationNode<double>();
        var consumer2 = new OptimizationNode<double>();

        Assert.Equal(0, producer.ConsumerCount());

        consumer1.AddInput(producer);
        Assert.Equal(1, producer.ConsumerCount());

        consumer2.AddInput(producer);
        Assert.Equal(2, producer.ConsumerCount());
    }

    [Fact]
    public void OptimizationNode_Clone_CreatesDeepCopy()
    {
        var original = new OptimizationNode<double>
        {
            Name = "test_node",
            OperationType = OperationType.Add,
            OutputShape = new[] { 1, 2, 3 },
            CanEliminate = false,
            IsFused = true
        };
        original.Parameters["weight"] = 0.5;
        original.Metadata["stride"] = 1;

        var clone = original.Clone();

        Assert.NotEqual(original.Id, clone.Id);
        Assert.Equal("test_node_clone", clone.Name);
        Assert.Equal(OperationType.Add, clone.OperationType);
        Assert.Equal(original.OutputShape, clone.OutputShape);
        Assert.Equal(original.CanEliminate, clone.CanEliminate);
        Assert.Equal(original.IsFused, clone.IsFused);
        Assert.Equal(original.Parameters["weight"], clone.Parameters["weight"]);
        Assert.Equal(original.Metadata["stride"], clone.Metadata["stride"]);
    }

    [Fact]
    public void OptimizationNode_ToString_ReturnsFormattedString()
    {
        var node = new OptimizationNode<double>
        {
            Name = "conv1",
            OperationType = OperationType.Convolution2D,
            OutputShape = new[] { 1, 64, 32, 32 }
        };

        var str = node.ToString();

        Assert.Contains("conv1", str);
        Assert.Contains("Convolution2D", str);
        Assert.Contains("[1, 64, 32, 32]", str);
    }

    #endregion

    #region OptimizationGraph Tests

    [Fact]
    public void OptimizationGraph_Constructor_InitializesEmptyCollections()
    {
        var graph = new OptimizationGraph<double>();

        Assert.Empty(graph.Nodes);
        Assert.Empty(graph.InputNodes);
        Assert.Empty(graph.OutputNodes);
    }

    [Fact]
    public void OptimizationGraph_AddNode_AddsToCollection()
    {
        var graph = new OptimizationGraph<double>();
        var node = new OptimizationNode<double> { Name = "node1" };

        graph.AddNode(node);

        Assert.Single(graph.Nodes);
        Assert.Contains(node, graph.Nodes);
    }

    [Fact]
    public void OptimizationGraph_AddNode_TracksInputNodes()
    {
        var graph = new OptimizationGraph<double>();
        var inputNode = new OptimizationNode<double>
        {
            Name = "input",
            OperationType = OperationType.Input
        };

        graph.AddNode(inputNode);

        Assert.Single(graph.InputNodes);
        Assert.Contains(inputNode, graph.InputNodes);
    }

    [Fact]
    public void OptimizationGraph_AddNode_TracksOutputNodes()
    {
        var graph = new OptimizationGraph<double>();
        var outputNode = new OptimizationNode<double>
        {
            Name = "output",
            OperationType = OperationType.Output
        };

        graph.AddNode(outputNode);

        Assert.Single(graph.OutputNodes);
        Assert.Contains(outputNode, graph.OutputNodes);
    }

    [Fact]
    public void OptimizationGraph_AddNode_ThrowsOnNull()
    {
        var graph = new OptimizationGraph<double>();

        Assert.Throws<ArgumentNullException>(() => graph.AddNode(null!));
    }

    [Fact]
    public void OptimizationGraph_AddNode_DoesNotAddDuplicate()
    {
        var graph = new OptimizationGraph<double>();
        var node = new OptimizationNode<double>();

        graph.AddNode(node);
        graph.AddNode(node); // Add same node twice

        Assert.Single(graph.Nodes);
    }

    [Fact]
    public void OptimizationGraph_RemoveNode_RemovesFromCollection()
    {
        var graph = new OptimizationGraph<double>();
        var node = new OptimizationNode<double>();

        graph.AddNode(node);
        graph.RemoveNode(node);

        Assert.Empty(graph.Nodes);
    }

    [Fact]
    public void OptimizationGraph_RemoveNode_RemovesConnections()
    {
        var graph = new OptimizationGraph<double>();
        var node1 = new OptimizationNode<double>();
        var node2 = new OptimizationNode<double>();
        var node3 = new OptimizationNode<double>();

        graph.AddNode(node1);
        graph.AddNode(node2);
        graph.AddNode(node3);

        node2.AddInput(node1);
        node3.AddInput(node2);

        graph.RemoveNode(node2);

        Assert.Empty(node1.Outputs);
        Assert.Empty(node3.Inputs);
    }

    [Fact]
    public void OptimizationGraph_FindNodeById_ReturnsCorrectNode()
    {
        var graph = new OptimizationGraph<double>();
        var node = new OptimizationNode<double> { Name = "target" };

        graph.AddNode(node);

        var found = graph.FindNodeById(node.Id);

        Assert.NotNull(found);
        Assert.Equal(node, found);
    }

    [Fact]
    public void OptimizationGraph_FindNodeById_ReturnsNullForMissing()
    {
        var graph = new OptimizationGraph<double>();

        var found = graph.FindNodeById("nonexistent");

        Assert.Null(found);
    }

    [Fact]
    public void OptimizationGraph_FindNodesByName_ReturnsMatchingNodes()
    {
        var graph = new OptimizationGraph<double>();
        var node1 = new OptimizationNode<double> { Name = "conv" };
        var node2 = new OptimizationNode<double> { Name = "conv" };
        var node3 = new OptimizationNode<double> { Name = "relu" };

        graph.AddNode(node1);
        graph.AddNode(node2);
        graph.AddNode(node3);

        var found = graph.FindNodesByName("conv");

        Assert.Equal(2, found.Count);
    }

    [Fact]
    public void OptimizationGraph_GetTopologicalOrder_ReturnsCorrectOrder()
    {
        var graph = new OptimizationGraph<double>();
        var input = new OptimizationNode<double> { Name = "input", OperationType = OperationType.Input };
        var middle = new OptimizationNode<double> { Name = "middle", OperationType = OperationType.ReLU };
        var output = new OptimizationNode<double> { Name = "output", OperationType = OperationType.Output };

        graph.AddNode(input);
        graph.AddNode(middle);
        graph.AddNode(output);

        middle.AddInput(input);
        output.AddInput(middle);

        var order = graph.GetTopologicalOrder();

        Assert.Equal(3, order.Count);
        Assert.True(order.IndexOf(input) < order.IndexOf(middle));
        Assert.True(order.IndexOf(middle) < order.IndexOf(output));
    }

    [Fact]
    public void OptimizationGraph_GetTopologicalOrder_ThrowsOnCycle()
    {
        var graph = new OptimizationGraph<double>();
        var node1 = new OptimizationNode<double> { Name = "node1" };
        var node2 = new OptimizationNode<double> { Name = "node2" };

        graph.AddNode(node1);
        graph.AddNode(node2);

        // Create a cycle
        node1.Inputs.Add(node2);
        node2.Inputs.Add(node1);
        node1.Outputs.Add(node2);
        node2.Outputs.Add(node1);

        Assert.Throws<InvalidOperationException>(() => graph.GetTopologicalOrder());
    }

    [Fact]
    public void OptimizationGraph_Validate_ReturnsTrueForValidGraph()
    {
        var graph = new OptimizationGraph<double>();
        var input = new OptimizationNode<double> { Name = "input", OperationType = OperationType.Input };
        var output = new OptimizationNode<double> { Name = "output", OperationType = OperationType.Output };

        graph.AddNode(input);
        graph.AddNode(output);

        output.AddInput(input);

        Assert.True(graph.Validate());
    }

    [Fact]
    public void OptimizationGraph_Validate_ReturnsFalseForCyclicGraph()
    {
        var graph = new OptimizationGraph<double>();
        var node1 = new OptimizationNode<double>();
        var node2 = new OptimizationNode<double>();

        graph.AddNode(node1);
        graph.AddNode(node2);

        // Create cycle
        node1.Inputs.Add(node2);
        node2.Inputs.Add(node1);
        node1.Outputs.Add(node2);
        node2.Outputs.Add(node1);

        Assert.False(graph.Validate());
    }

    [Fact]
    public void OptimizationGraph_Clone_CreatesDeepCopy()
    {
        var graph = new OptimizationGraph<double>();
        var input = new OptimizationNode<double> { Name = "input", OperationType = OperationType.Input };
        var output = new OptimizationNode<double> { Name = "output", OperationType = OperationType.Output };

        graph.AddNode(input);
        graph.AddNode(output);
        output.AddInput(input);

        var clone = graph.Clone();

        Assert.Equal(2, clone.Nodes.Count);
        Assert.Single(clone.InputNodes);
        Assert.Single(clone.OutputNodes);
    }

    [Fact]
    public void OptimizationGraph_GetStatistics_ReturnsCorrectStats()
    {
        var graph = new OptimizationGraph<double>();
        var input = new OptimizationNode<double> { OperationType = OperationType.Input };
        var relu1 = new OptimizationNode<double> { OperationType = OperationType.ReLU };
        var relu2 = new OptimizationNode<double> { OperationType = OperationType.ReLU, IsFused = true };
        var output = new OptimizationNode<double> { OperationType = OperationType.Output };

        graph.AddNode(input);
        graph.AddNode(relu1);
        graph.AddNode(relu2);
        graph.AddNode(output);

        var stats = graph.GetStatistics();

        Assert.Equal(4, stats.TotalNodes);
        Assert.Equal(1, stats.InputNodes);
        Assert.Equal(1, stats.OutputNodes);
        Assert.Equal(1, stats.FusedNodes);
        Assert.True(stats.OperationTypeCounts.ContainsKey(OperationType.ReLU));
        Assert.Equal(2, stats.OperationTypeCounts[OperationType.ReLU]);
    }

    [Fact]
    public void OptimizationGraph_ToString_ReturnsFormattedString()
    {
        var graph = new OptimizationGraph<double>();
        var input = new OptimizationNode<double> { OperationType = OperationType.Input };
        var output = new OptimizationNode<double> { OperationType = OperationType.Output };

        graph.AddNode(input);
        graph.AddNode(output);

        var str = graph.ToString();

        Assert.Contains("2 nodes", str);
        Assert.Contains("1 inputs", str);
        Assert.Contains("1 outputs", str);
    }

    #endregion

    #region GraphStatistics Tests

    [Fact]
    public void GraphStatistics_ToString_ReturnsFormattedString()
    {
        var stats = new GraphStatistics
        {
            TotalNodes = 10,
            InputNodes = 2,
            OutputNodes = 1,
            FusedNodes = 3,
            OperationTypeCounts = new Dictionary<OperationType, int>
            {
                { OperationType.ReLU, 5 },
                { OperationType.Add, 3 }
            }
        };

        var str = stats.ToString();

        Assert.Contains("10", str);
        Assert.Contains("ReLU", str);
    }

    #endregion

    #region OptimizationLevel Enum Tests

    [Fact]
    public void OptimizationLevel_HasExpectedValues()
    {
        var levels = (OptimizationLevel[])Enum.GetValues(typeof(OptimizationLevel));

        Assert.Contains(OptimizationLevel.None, levels);
        Assert.Contains(OptimizationLevel.Basic, levels);
        Assert.Contains(OptimizationLevel.Standard, levels);
        Assert.Contains(OptimizationLevel.Aggressive, levels);
        Assert.Contains(OptimizationLevel.Maximum, levels);
    }

    [Fact]
    public void OptimizationLevel_ValuesAreOrdered()
    {
        Assert.True((int)OptimizationLevel.None < (int)OptimizationLevel.Basic);
        Assert.True((int)OptimizationLevel.Basic < (int)OptimizationLevel.Standard);
        Assert.True((int)OptimizationLevel.Standard < (int)OptimizationLevel.Aggressive);
        Assert.True((int)OptimizationLevel.Aggressive < (int)OptimizationLevel.Maximum);
    }

    #endregion

    #region OptimizationOptions Tests

    [Fact]
    public void OptimizationOptions_Constructor_SetsDefaults()
    {
        var options = new OptimizationOptions();

        Assert.Equal(OptimizationLevel.Standard, options.Level);
        Assert.Equal("NCHW", options.TargetLayout);
        Assert.Equal(10, options.MaxIterations);
        Assert.True(options.EnableOperatorFusion);
        Assert.True(options.EnableConstantFolding);
        Assert.True(options.EnableDeadCodeElimination);
        Assert.True(options.EnableCSE);
        Assert.True(options.EnableLayoutOptimization);
        Assert.True(options.EnableInPlaceOptimization);
        Assert.True(options.EnableMemoryReuse);
        Assert.True(options.EnableAlgebraicSimplification);
        Assert.True(options.EnableStrengthReduction);
        Assert.False(options.PrintStatistics);
        Assert.False(options.ValidateAfterEachPass);
    }

    [Fact]
    public void OptimizationOptions_FromLevel_None_DisablesAll()
    {
        var options = OptimizationOptions.FromLevel(OptimizationLevel.None);

        Assert.False(options.EnableOperatorFusion);
        Assert.False(options.EnableConstantFolding);
        Assert.False(options.EnableDeadCodeElimination);
        Assert.False(options.EnableCSE);
        Assert.False(options.EnableLayoutOptimization);
        Assert.False(options.EnableInPlaceOptimization);
        Assert.False(options.EnableMemoryReuse);
        Assert.False(options.EnableAlgebraicSimplification);
        Assert.False(options.EnableStrengthReduction);
    }

    [Fact]
    public void OptimizationOptions_FromLevel_Basic_EnablesBasicOnly()
    {
        var options = OptimizationOptions.FromLevel(OptimizationLevel.Basic);

        Assert.True(options.EnableDeadCodeElimination);
        Assert.True(options.EnableConstantFolding);
        Assert.False(options.EnableOperatorFusion);
        Assert.False(options.EnableCSE);
    }

    [Fact]
    public void OptimizationOptions_FromLevel_Standard_EnablesStandard()
    {
        var options = OptimizationOptions.FromLevel(OptimizationLevel.Standard);

        Assert.True(options.EnableOperatorFusion);
        Assert.True(options.EnableConstantFolding);
        Assert.True(options.EnableDeadCodeElimination);
        Assert.True(options.EnableAlgebraicSimplification);
    }

    [Fact]
    public void OptimizationOptions_FromLevel_Aggressive_EnablesMore()
    {
        var options = OptimizationOptions.FromLevel(OptimizationLevel.Aggressive);

        Assert.True(options.EnableOperatorFusion);
        Assert.True(options.EnableConstantFolding);
        Assert.True(options.EnableDeadCodeElimination);
        Assert.True(options.EnableCSE);
        Assert.True(options.EnableAlgebraicSimplification);
        Assert.True(options.EnableStrengthReduction);
        Assert.True(options.EnableInPlaceOptimization);
        Assert.True(options.EnableMemoryReuse);
    }

    [Fact]
    public void OptimizationOptions_FromLevel_Maximum_EnablesAll()
    {
        var options = OptimizationOptions.FromLevel(OptimizationLevel.Maximum);

        Assert.True(options.EnableOperatorFusion);
        Assert.True(options.EnableConstantFolding);
        Assert.True(options.EnableDeadCodeElimination);
        Assert.True(options.EnableCSE);
        Assert.True(options.EnableLayoutOptimization);
        Assert.True(options.EnableInPlaceOptimization);
        Assert.True(options.EnableMemoryReuse);
        Assert.True(options.EnableAlgebraicSimplification);
        Assert.True(options.EnableStrengthReduction);
    }

    #endregion

    #region OptimizationPassType Enum Tests

    [Fact]
    public void OptimizationPassType_HasExpectedValues()
    {
        var passTypes = (OptimizationPassType[])Enum.GetValues(typeof(OptimizationPassType));

        // Fusion passes
        Assert.Contains(OptimizationPassType.OperatorFusion, passTypes);
        Assert.Contains(OptimizationPassType.ConvBatchNormFusion, passTypes);
        Assert.Contains(OptimizationPassType.ConvBatchNormReLUFusion, passTypes);
        Assert.Contains(OptimizationPassType.MatMulBiasFusion, passTypes);
        Assert.Contains(OptimizationPassType.ElementwiseFusion, passTypes);

        // Graph optimization passes
        Assert.Contains(OptimizationPassType.ConstantFolding, passTypes);
        Assert.Contains(OptimizationPassType.DeadCodeElimination, passTypes);
        Assert.Contains(OptimizationPassType.CommonSubexpressionElimination, passTypes);
        Assert.Contains(OptimizationPassType.LayoutOptimization, passTypes);

        // Memory passes
        Assert.Contains(OptimizationPassType.InPlaceOptimization, passTypes);
        Assert.Contains(OptimizationPassType.MemoryReuseOptimization, passTypes);

        // Computation passes
        Assert.Contains(OptimizationPassType.AlgebraicSimplification, passTypes);
        Assert.Contains(OptimizationPassType.StrengthReduction, passTypes);
    }

    #endregion

    #region DeadCodeEliminationPass Tests

    [Fact]
    public void DeadCodeEliminationPass_Properties()
    {
        var pass = new DeadCodeEliminationPass<double>();

        Assert.Equal(OptimizationPassType.DeadCodeElimination, pass.PassType);
        Assert.Equal("Dead Code Elimination", pass.Name);
    }

    [Fact]
    public void DeadCodeEliminationPass_CanApply_ReturnsFalseForEmptyGraph()
    {
        var pass = new DeadCodeEliminationPass<double>();
        var graph = new OptimizationGraph<double>();

        Assert.False(pass.CanApply(graph));
    }

    [Fact]
    public void DeadCodeEliminationPass_CanApply_ReturnsFalseWhenNoOutputs()
    {
        var pass = new DeadCodeEliminationPass<double>();
        var graph = new OptimizationGraph<double>();
        var node = new OptimizationNode<double> { OperationType = OperationType.ReLU };
        graph.AddNode(node);

        Assert.False(pass.CanApply(graph));
    }

    [Fact]
    public void DeadCodeEliminationPass_CanApply_ReturnsTrueForValidGraph()
    {
        var pass = new DeadCodeEliminationPass<double>();
        var graph = new OptimizationGraph<double>();
        var output = new OptimizationNode<double> { OperationType = OperationType.Output };
        graph.AddNode(output);

        Assert.True(pass.CanApply(graph));
    }

    [Fact]
    public void DeadCodeEliminationPass_Apply_RemovesUnreachableNodes()
    {
        var pass = new DeadCodeEliminationPass<double>();
        var graph = new OptimizationGraph<double>();

        var input = new OptimizationNode<double> { OperationType = OperationType.Input };
        var output = new OptimizationNode<double> { OperationType = OperationType.Output };
        var deadNode = new OptimizationNode<double> { OperationType = OperationType.ReLU, CanEliminate = true };

        graph.AddNode(input);
        graph.AddNode(output);
        graph.AddNode(deadNode);
        output.AddInput(input);

        bool modified = pass.Apply(graph);

        Assert.True(modified);
        Assert.DoesNotContain(deadNode, graph.Nodes);
        Assert.Equal(2, graph.Nodes.Count);
    }

    [Fact]
    public void DeadCodeEliminationPass_Apply_PreservesReachableNodes()
    {
        var pass = new DeadCodeEliminationPass<double>();
        var graph = new OptimizationGraph<double>();

        var input = new OptimizationNode<double> { OperationType = OperationType.Input };
        var middle = new OptimizationNode<double> { OperationType = OperationType.ReLU };
        var output = new OptimizationNode<double> { OperationType = OperationType.Output };

        graph.AddNode(input);
        graph.AddNode(middle);
        graph.AddNode(output);

        middle.AddInput(input);
        output.AddInput(middle);

        bool modified = pass.Apply(graph);

        Assert.False(modified);
        Assert.Equal(3, graph.Nodes.Count);
    }

    [Fact]
    public void DeadCodeEliminationPass_Apply_DoesNotRemoveNonEliminableNodes()
    {
        var pass = new DeadCodeEliminationPass<double>();
        var graph = new OptimizationGraph<double>();

        var input = new OptimizationNode<double> { OperationType = OperationType.Input };
        var output = new OptimizationNode<double> { OperationType = OperationType.Output };
        var sideEffectNode = new OptimizationNode<double>
        {
            OperationType = OperationType.Custom,
            CanEliminate = false
        };

        graph.AddNode(input);
        graph.AddNode(output);
        graph.AddNode(sideEffectNode);
        output.AddInput(input);

        bool modified = pass.Apply(graph);

        Assert.Contains(sideEffectNode, graph.Nodes);
    }

    #endregion

    #region IRDataType Tests

    [Fact]
    public void IRDataType_HasExpectedValues()
    {
        var types = (IRDataType[])Enum.GetValues(typeof(IRDataType));

        // Floating point
        Assert.Contains(IRDataType.Float16, types);
        Assert.Contains(IRDataType.Float32, types);
        Assert.Contains(IRDataType.Float64, types);
        Assert.Contains(IRDataType.BFloat16, types);

        // Integer
        Assert.Contains(IRDataType.Int8, types);
        Assert.Contains(IRDataType.Int32, types);
        Assert.Contains(IRDataType.Int64, types);
        Assert.Contains(IRDataType.UInt8, types);

        // Quantized
        Assert.Contains(IRDataType.QInt8, types);
        Assert.Contains(IRDataType.QUInt8, types);
        Assert.Contains(IRDataType.QInt4, types);

        // Other
        Assert.Contains(IRDataType.Bool, types);
        Assert.Contains(IRDataType.Complex64, types);
        Assert.Contains(IRDataType.Decimal, types);
    }

    [Fact]
    public void IRDataTypeExtensions_IsFloatingPoint()
    {
        Assert.True(IRDataType.Float16.IsFloatingPoint());
        Assert.True(IRDataType.Float32.IsFloatingPoint());
        Assert.True(IRDataType.Float64.IsFloatingPoint());
        Assert.True(IRDataType.BFloat16.IsFloatingPoint());
        Assert.False(IRDataType.Int32.IsFloatingPoint());
        Assert.False(IRDataType.QInt8.IsFloatingPoint());
    }

    [Fact]
    public void IRDataTypeExtensions_IsInteger()
    {
        Assert.True(IRDataType.Int8.IsInteger());
        Assert.True(IRDataType.Int32.IsInteger());
        Assert.True(IRDataType.UInt8.IsInteger());
        Assert.True(IRDataType.Int64.IsInteger());
        Assert.False(IRDataType.Float32.IsInteger());
        Assert.False(IRDataType.QInt8.IsInteger());
    }

    [Fact]
    public void IRDataTypeExtensions_IsQuantized()
    {
        Assert.True(IRDataType.QInt8.IsQuantized());
        Assert.True(IRDataType.QUInt8.IsQuantized());
        Assert.True(IRDataType.QInt4.IsQuantized());
        Assert.True(IRDataType.QInt2.IsQuantized());
        Assert.False(IRDataType.Float32.IsQuantized());
        Assert.False(IRDataType.Int8.IsQuantized());
    }

    [Fact]
    public void IRDataTypeExtensions_ElementSizeInBytes()
    {
        Assert.Equal(1, IRDataType.Bool.ElementSizeInBytes());
        Assert.Equal(1, IRDataType.Int8.ElementSizeInBytes());
        Assert.Equal(2, IRDataType.Float16.ElementSizeInBytes());
        Assert.Equal(4, IRDataType.Float32.ElementSizeInBytes());
        Assert.Equal(8, IRDataType.Float64.ElementSizeInBytes());
        Assert.Equal(16, IRDataType.Decimal.ElementSizeInBytes());
    }

    [Fact]
    public void IRDataTypeExtensions_FromSystemType()
    {
        Assert.Equal(IRDataType.Float32, IRDataTypeExtensions.FromSystemType(typeof(float)));
        Assert.Equal(IRDataType.Float64, IRDataTypeExtensions.FromSystemType(typeof(double)));
        Assert.Equal(IRDataType.Int32, IRDataTypeExtensions.FromSystemType(typeof(int)));
        Assert.Equal(IRDataType.Int64, IRDataTypeExtensions.FromSystemType(typeof(long)));
        Assert.Equal(IRDataType.Bool, IRDataTypeExtensions.FromSystemType(typeof(bool)));
        Assert.Equal(IRDataType.Decimal, IRDataTypeExtensions.FromSystemType(typeof(decimal)));
    }

    [Fact]
    public void IRDataTypeExtensions_ToSystemType()
    {
        Assert.Equal(typeof(float), IRDataType.Float32.ToSystemType());
        Assert.Equal(typeof(double), IRDataType.Float64.ToSystemType());
        Assert.Equal(typeof(int), IRDataType.Int32.ToSystemType());
        Assert.Equal(typeof(long), IRDataType.Int64.ToSystemType());
        Assert.Equal(typeof(bool), IRDataType.Bool.ToSystemType());
        Assert.Equal(typeof(decimal), IRDataType.Decimal.ToSystemType());
    }

    #endregion

    #region MemoryLayout Tests

    [Fact]
    public void MemoryLayout_HasExpectedValues()
    {
        var layouts = (MemoryLayout[])Enum.GetValues(typeof(MemoryLayout));

        Assert.Contains(MemoryLayout.RowMajor, layouts);
        Assert.Contains(MemoryLayout.ColumnMajor, layouts);
        Assert.Contains(MemoryLayout.NCHW, layouts);
        Assert.Contains(MemoryLayout.NHWC, layouts);
        Assert.Contains(MemoryLayout.Tiled4x4, layouts);
        Assert.Contains(MemoryLayout.Blocked, layouts);
    }

    #endregion

    #region DeviceType Tests

    [Fact]
    public void DeviceType_HasExpectedValues()
    {
        var devices = (DeviceType[])Enum.GetValues(typeof(DeviceType));

        Assert.Contains(DeviceType.CPU, devices);
        Assert.Contains(DeviceType.GPU, devices);
        Assert.Contains(DeviceType.TPU, devices);
        Assert.Contains(DeviceType.NPU, devices);
        Assert.Contains(DeviceType.FPGA, devices);
        Assert.Contains(DeviceType.Auto, devices);
        Assert.Contains(DeviceType.Any, devices);
    }

    #endregion

    #region QuantizationParams Tests

    [Fact]
    public void QuantizationParams_DefaultValues()
    {
        var qParams = new QuantizationParams();

        Assert.Equal(1.0, qParams.Scale);
        Assert.Equal(0, qParams.ZeroPoint);
        Assert.Equal(double.MinValue, qParams.Min);
        Assert.Equal(double.MaxValue, qParams.Max);
        Assert.False(qParams.PerChannel);
        Assert.Equal(-1, qParams.QuantizationAxis);
        Assert.Null(qParams.PerChannelScales);
        Assert.Null(qParams.PerChannelZeroPoints);
    }

    [Fact]
    public void QuantizationParams_CanBeConfigured()
    {
        var qParams = new QuantizationParams
        {
            Scale = 0.01,
            ZeroPoint = 128,
            Min = -1.0,
            Max = 1.0,
            PerChannel = true,
            QuantizationAxis = 0,
            PerChannelScales = new[] { 0.01, 0.02, 0.03 },
            PerChannelZeroPoints = new[] { 128, 127, 126 }
        };

        Assert.Equal(0.01, qParams.Scale);
        Assert.Equal(128, qParams.ZeroPoint);
        Assert.True(qParams.PerChannel);
        Assert.Equal(3, qParams.PerChannelScales?.Length);
    }

    #endregion

    #region TensorType Tests

    [Fact]
    public void TensorType_DefaultValues()
    {
        var tensorType = new TensorType();

        Assert.Equal(IRDataType.Float32, tensorType.DataType);
        Assert.Empty(tensorType.Shape);
        Assert.Equal(MemoryLayout.RowMajor, tensorType.Layout);
        Assert.Equal(DeviceType.Auto, tensorType.Device);
        Assert.Null(tensorType.Quantization);
        Assert.Null(tensorType.Strides);
    }

    [Fact]
    public void TensorType_HasDynamicShape_ReturnsTrueForDynamicDimensions()
    {
        var staticType = new TensorType { Shape = new[] { 1, 3, 224, 224 } };
        var dynamicType = new TensorType { Shape = new[] { -1, 3, 224, 224 } };

        Assert.False(staticType.HasDynamicShape);
        Assert.True(dynamicType.HasDynamicShape);
    }

    [Fact]
    public void TensorType_NumElements_CalculatesCorrectly()
    {
        var scalar = new TensorType { Shape = Array.Empty<int>() };
        var vector = new TensorType { Shape = new[] { 10 } };
        var matrix = new TensorType { Shape = new[] { 3, 4 } };
        var tensor = new TensorType { Shape = new[] { 2, 3, 4 } };

        Assert.Equal(1, scalar.NumElements);
        Assert.Equal(10, vector.NumElements);
        Assert.Equal(12, matrix.NumElements);
        Assert.Equal(24, tensor.NumElements);
    }

    [Fact]
    public void TensorType_NumElements_ReturnsMinusOneForDynamic()
    {
        var dynamicType = new TensorType { Shape = new[] { -1, 3, 224, 224 } };

        Assert.Equal(-1, dynamicType.NumElements);
    }

    [Fact]
    public void TensorType_ElementSize_ReturnsCorrectSize()
    {
        var float32Type = new TensorType { DataType = IRDataType.Float32 };
        var float64Type = new TensorType { DataType = IRDataType.Float64 };
        var int8Type = new TensorType { DataType = IRDataType.Int8 };

        Assert.Equal(4, float32Type.ElementSize);
        Assert.Equal(8, float64Type.ElementSize);
        Assert.Equal(1, int8Type.ElementSize);
    }

    [Fact]
    public void TensorType_TotalBytes_CalculatesCorrectly()
    {
        var tensorType = new TensorType
        {
            DataType = IRDataType.Float32,
            Shape = new[] { 1, 3, 224, 224 }
        };

        long expectedBytes = 1L * 3 * 224 * 224 * 4; // 4 bytes per float32
        Assert.Equal(expectedBytes, tensorType.TotalBytes);
    }

    [Fact]
    public void TensorType_IsBroadcastCompatible_ChecksCorrectly()
    {
        var type1 = new TensorType { Shape = new[] { 1, 3, 1 } };
        var type2 = new TensorType { Shape = new[] { 4, 3, 5 } };
        var type3 = new TensorType { Shape = new[] { 4, 2, 5 } };

        Assert.True(type1.IsBroadcastCompatible(type2));
        Assert.False(type1.IsBroadcastCompatible(type3)); // 3 != 2
    }

    [Fact]
    public void TensorType_Clone_CreatesDeepCopy()
    {
        var original = new TensorType
        {
            DataType = IRDataType.Float16,
            Shape = new[] { 1, 2, 3 },
            Layout = MemoryLayout.NHWC,
            Device = DeviceType.GPU
        };

        var clone = original.Clone();

        Assert.Equal(original.DataType, clone.DataType);
        Assert.Equal(original.Shape, clone.Shape);
        Assert.NotSame(original.Shape, clone.Shape);
        Assert.Equal(original.Layout, clone.Layout);
        Assert.Equal(original.Device, clone.Device);
    }

    [Fact]
    public void TensorType_ToString_ReturnsFormattedString()
    {
        var tensorType = new TensorType
        {
            DataType = IRDataType.Float32,
            Shape = new[] { 1, 3, 224, 224 },
            Device = DeviceType.GPU
        };

        var str = tensorType.ToString();

        Assert.Contains("Float32", str);
        Assert.Contains("GPU", str);
    }

    #endregion

    #region ConstantFoldingPass Tests

    [Fact]
    public void ConstantFoldingPass_Properties()
    {
        var pass = new ConstantFoldingPass<double>();

        Assert.Equal(OptimizationPassType.ConstantFolding, pass.PassType);
        Assert.Equal("Constant Folding", pass.Name);
    }

    [Fact]
    public void ConstantFoldingPass_CanApply_ReturnsTrueWhenConstantExists()
    {
        var pass = new ConstantFoldingPass<double>();
        var graph = new OptimizationGraph<double>();
        var constantNode = new OptimizationNode<double> { OperationType = OperationType.Constant };
        graph.AddNode(constantNode);

        Assert.True(pass.CanApply(graph));
    }

    [Fact]
    public void ConstantFoldingPass_CanApply_ReturnsFalseWhenNoConstants()
    {
        var pass = new ConstantFoldingPass<double>();
        var graph = new OptimizationGraph<double>();
        var node = new OptimizationNode<double> { OperationType = OperationType.ReLU };
        graph.AddNode(node);

        Assert.False(pass.CanApply(graph));
    }

    #endregion

    #region AlgebraicSimplificationPass Tests

    [Fact]
    public void AlgebraicSimplificationPass_Properties()
    {
        var pass = new AlgebraicSimplificationPass<double>();

        Assert.Equal(OptimizationPassType.AlgebraicSimplification, pass.PassType);
        Assert.Equal("Algebraic Simplification", pass.Name);
    }

    #endregion

    #region CommonSubexpressionEliminationPass Tests

    [Fact]
    public void CommonSubexpressionEliminationPass_Properties()
    {
        var pass = new CommonSubexpressionEliminationPass<double>();

        Assert.Equal(OptimizationPassType.CommonSubexpressionElimination, pass.PassType);
        Assert.Equal("Common Subexpression Elimination", pass.Name);
    }

    [Fact]
    public void CommonSubexpressionEliminationPass_EliminatesIdenticalAddOperations()
    {
        var pass = new CommonSubexpressionEliminationPass<double>();
        var graph = new OptimizationGraph<double>();

        var input1 = new OptimizationNode<double> { OperationType = OperationType.Input, Name = "a" };
        var input2 = new OptimizationNode<double> { OperationType = OperationType.Input, Name = "b" };
        var add1 = new OptimizationNode<double> { OperationType = OperationType.Add, Name = "add1" };
        var add2 = new OptimizationNode<double> { OperationType = OperationType.Add, Name = "add2" };
        var output = new OptimizationNode<double> { OperationType = OperationType.Output, Name = "out" };

        graph.AddNode(input1);
        graph.AddNode(input2);
        graph.AddNode(add1);
        graph.AddNode(add2);
        graph.AddNode(output);

        // Both add operations use the same inputs (a + b and a + b)
        add1.AddInput(input1);
        add1.AddInput(input2);
        add2.AddInput(input1);
        add2.AddInput(input2);
        output.AddInput(add1);

        bool modified = pass.Apply(graph);

        // Should eliminate one of the duplicate add operations
        Assert.True(modified);
    }

    /// <summary>
    /// BUG TEST: CSE should NOT eliminate non-commutative operations with reversed operands.
    /// The current implementation incorrectly sorts input IDs which would merge a-b and b-a.
    /// </summary>
    [Fact]
    public void CommonSubexpressionEliminationPass_PreservesNonCommutativeOperations()
    {
        var pass = new CommonSubexpressionEliminationPass<double>();
        var graph = new OptimizationGraph<double>();

        var inputA = new OptimizationNode<double> { OperationType = OperationType.Input, Name = "a" };
        var inputB = new OptimizationNode<double> { OperationType = OperationType.Input, Name = "b" };
        var sub1 = new OptimizationNode<double> { OperationType = OperationType.Subtract, Name = "a_minus_b" };
        var sub2 = new OptimizationNode<double> { OperationType = OperationType.Subtract, Name = "b_minus_a" };
        var add = new OptimizationNode<double> { OperationType = OperationType.Add, Name = "result" };
        var output = new OptimizationNode<double> { OperationType = OperationType.Output, Name = "out" };

        graph.AddNode(inputA);
        graph.AddNode(inputB);
        graph.AddNode(sub1);
        graph.AddNode(sub2);
        graph.AddNode(add);
        graph.AddNode(output);

        // sub1 = a - b
        sub1.AddInput(inputA);
        sub1.AddInput(inputB);

        // sub2 = b - a (DIFFERENT from a - b!)
        sub2.AddInput(inputB);
        sub2.AddInput(inputA);

        // result = (a - b) + (b - a)
        add.AddInput(sub1);
        add.AddInput(sub2);
        output.AddInput(add);

        // Count subtraction nodes before
        int subCountBefore = graph.Nodes.Count(n => n.OperationType == OperationType.Subtract);
        Assert.Equal(2, subCountBefore);

        bool modified = pass.Apply(graph);

        // Count subtraction nodes after
        int subCountAfter = graph.Nodes.Count(n => n.OperationType == OperationType.Subtract);

        // Both subtraction nodes should be preserved because a-b ≠ b-a
        Assert.Equal(2, subCountAfter);
        Assert.False(modified, "Non-commutative operations with different operand order should NOT be merged");
    }

    /// <summary>
    /// BUG TEST: CSE should NOT eliminate division operations with reversed operands.
    /// a/b ≠ b/a, so these should not be merged.
    /// </summary>
    [Fact]
    public void CommonSubexpressionEliminationPass_PreservesDivisionOperandOrder()
    {
        var pass = new CommonSubexpressionEliminationPass<double>();
        var graph = new OptimizationGraph<double>();

        var inputA = new OptimizationNode<double> { OperationType = OperationType.Input, Name = "a" };
        var inputB = new OptimizationNode<double> { OperationType = OperationType.Input, Name = "b" };
        var div1 = new OptimizationNode<double> { OperationType = OperationType.Divide, Name = "a_div_b" };
        var div2 = new OptimizationNode<double> { OperationType = OperationType.Divide, Name = "b_div_a" };
        var multiply = new OptimizationNode<double> { OperationType = OperationType.Multiply, Name = "result" };
        var output = new OptimizationNode<double> { OperationType = OperationType.Output, Name = "out" };

        graph.AddNode(inputA);
        graph.AddNode(inputB);
        graph.AddNode(div1);
        graph.AddNode(div2);
        graph.AddNode(multiply);
        graph.AddNode(output);

        // div1 = a / b
        div1.AddInput(inputA);
        div1.AddInput(inputB);

        // div2 = b / a (DIFFERENT from a / b!)
        div2.AddInput(inputB);
        div2.AddInput(inputA);

        // result = (a / b) * (b / a)
        multiply.AddInput(div1);
        multiply.AddInput(div2);
        output.AddInput(multiply);

        // Count division nodes before
        int divCountBefore = graph.Nodes.Count(n => n.OperationType == OperationType.Divide);
        Assert.Equal(2, divCountBefore);

        bool modified = pass.Apply(graph);

        // Count division nodes after
        int divCountAfter = graph.Nodes.Count(n => n.OperationType == OperationType.Divide);

        // Both division nodes should be preserved because a/b ≠ b/a
        Assert.Equal(2, divCountAfter);
        Assert.False(modified, "Division operations with different operand order should NOT be merged");
    }

    [Fact]
    public void CommonSubexpressionEliminationPass_CanApply_ReturnsTrueForGraphWithMultipleNodes()
    {
        var pass = new CommonSubexpressionEliminationPass<double>();
        var graph = new OptimizationGraph<double>();

        var node1 = new OptimizationNode<double> { OperationType = OperationType.Add };
        var node2 = new OptimizationNode<double> { OperationType = OperationType.Add };
        graph.AddNode(node1);
        graph.AddNode(node2);

        Assert.True(pass.CanApply(graph));
    }

    [Fact]
    public void CommonSubexpressionEliminationPass_CanApply_ReturnsFalseForSingleNodeGraph()
    {
        var pass = new CommonSubexpressionEliminationPass<double>();
        var graph = new OptimizationGraph<double>();

        var node = new OptimizationNode<double>();
        graph.AddNode(node);

        Assert.False(pass.CanApply(graph));
    }

    #endregion

    #region StrengthReductionPass Tests

    [Fact]
    public void StrengthReductionPass_Properties()
    {
        var pass = new StrengthReductionPass<double>();

        Assert.Equal(OptimizationPassType.StrengthReduction, pass.PassType);
        Assert.Equal("Strength Reduction", pass.Name);
    }

    #endregion

    #region InPlaceOptimizationPass Tests

    [Fact]
    public void InPlaceOptimizationPass_Properties()
    {
        var pass = new InPlaceOptimizationPass<double>();

        Assert.Equal(OptimizationPassType.InPlaceOptimization, pass.PassType);
        Assert.Equal("In-Place Operation Optimization", pass.Name);
    }

    #endregion

    #region MemoryReuseOptimizationPass Tests

    [Fact]
    public void MemoryReuseOptimizationPass_Properties()
    {
        var pass = new MemoryReuseOptimizationPass<double>();

        Assert.Equal(OptimizationPassType.MemoryReuseOptimization, pass.PassType);
        Assert.Equal("Memory Reuse Optimization", pass.Name);
    }

    #endregion

    #region LayoutOptimizationPass Tests

    [Fact]
    public void LayoutOptimizationPass_Properties()
    {
        var pass = new LayoutOptimizationPass<double>();

        Assert.Equal(OptimizationPassType.LayoutOptimization, pass.PassType);
        Assert.Equal("Layout Optimization", pass.Name);
    }

    #endregion

    #region ElementwiseFusionPass Tests

    [Fact]
    public void ElementwiseFusionPass_Properties()
    {
        var pass = new ElementwiseFusionPass<double>();

        Assert.Equal(OptimizationPassType.ElementwiseFusion, pass.PassType);
        Assert.Equal("Elementwise Operation Fusion", pass.Name);
    }

    #endregion

    #region ConvBatchNormFusionPass Tests

    [Fact]
    public void ConvBatchNormFusionPass_Properties()
    {
        var pass = new ConvBatchNormFusionPass<double>();

        Assert.Equal(OptimizationPassType.ConvBatchNormFusion, pass.PassType);
        Assert.Equal("Conv + BatchNorm Fusion", pass.Name);
    }

    #endregion

    #region ConvBatchNormReLUFusionPass Tests

    [Fact]
    public void ConvBatchNormReLUFusionPass_Properties()
    {
        var pass = new ConvBatchNormReLUFusionPass<double>();

        Assert.Equal(OptimizationPassType.ConvBatchNormReLUFusion, pass.PassType);
        Assert.Equal("Conv + BatchNorm + ReLU Fusion", pass.Name);
    }

    #endregion

    #region MatMulBiasFusionPass Tests

    [Fact]
    public void MatMulBiasFusionPass_Properties()
    {
        var pass = new MatMulBiasFusionPass<double>();

        Assert.Equal(OptimizationPassType.MatMulBiasFusion, pass.PassType);
        Assert.Equal("MatMul + Bias Fusion", pass.Name);
    }

    #endregion

    #region MatMulBiasActivationFusionPass Tests

    [Fact]
    public void MatMulBiasActivationFusionPass_Properties()
    {
        var pass = new MatMulBiasActivationFusionPass<double>();

        Assert.Equal(OptimizationPassType.MatMulBiasActivationFusion, pass.PassType);
        Assert.Equal("MatMul + Bias + Activation Fusion", pass.Name);
    }

    #endregion

    #region MultiHeadAttentionFusionPass Tests

    [Fact]
    public void MultiHeadAttentionFusionPass_Properties()
    {
        var pass = new MultiHeadAttentionFusionPass<double>();

        Assert.Equal(OptimizationPassType.AttentionFusion, pass.PassType);
        Assert.Equal("Multi-Head Attention Fusion", pass.Name);
    }

    #endregion

    #region Integration Tests - Graph Construction and Optimization

    [Fact]
    public void IntegrationTest_SimpleLinearGraph()
    {
        // Create a simple Input -> ReLU -> Output graph
        var graph = new OptimizationGraph<double>();

        var input = new OptimizationNode<double>
        {
            Name = "input",
            OperationType = OperationType.Input,
            OutputShape = new[] { 1, 784 }
        };

        var relu = new OptimizationNode<double>
        {
            Name = "relu1",
            OperationType = OperationType.ReLU,
            OutputShape = new[] { 1, 784 }
        };

        var output = new OptimizationNode<double>
        {
            Name = "output",
            OperationType = OperationType.Output,
            OutputShape = new[] { 1, 784 }
        };

        graph.AddNode(input);
        graph.AddNode(relu);
        graph.AddNode(output);

        relu.AddInput(input);
        output.AddInput(relu);

        Assert.True(graph.Validate());
        Assert.Equal(3, graph.Nodes.Count);
        Assert.Single(graph.InputNodes);
        Assert.Single(graph.OutputNodes);

        var order = graph.GetTopologicalOrder();
        Assert.Equal(input, order[0]);
        Assert.Equal(relu, order[1]);
        Assert.Equal(output, order[2]);
    }

    [Fact]
    public void IntegrationTest_ApplyMultiplePasses()
    {
        var graph = new OptimizationGraph<double>();

        var input = new OptimizationNode<double> { OperationType = OperationType.Input };
        var relu = new OptimizationNode<double> { OperationType = OperationType.ReLU };
        var output = new OptimizationNode<double> { OperationType = OperationType.Output };
        var deadNode = new OptimizationNode<double> { OperationType = OperationType.Add, CanEliminate = true };

        graph.AddNode(input);
        graph.AddNode(relu);
        graph.AddNode(output);
        graph.AddNode(deadNode);

        relu.AddInput(input);
        output.AddInput(relu);

        // Apply dead code elimination
        var dcePass = new DeadCodeEliminationPass<double>();
        bool modified = dcePass.Apply(graph);

        Assert.True(modified);
        Assert.Equal(3, graph.Nodes.Count);
        Assert.DoesNotContain(deadNode, graph.Nodes);
    }

    #endregion
}
