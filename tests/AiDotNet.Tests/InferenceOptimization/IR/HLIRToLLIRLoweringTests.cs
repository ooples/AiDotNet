using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.IR.Common;
using AiDotNet.InferenceOptimization.IR.HighLevel;
using AiDotNet.InferenceOptimization.IR.Lowering;
using AiDotNet.InferenceOptimization.IR.LowLevel;
using Xunit;

namespace AiDotNet.Tests.InferenceOptimization.IR;

/// <summary>
/// Tests for HLIR to LLIR lowering pipeline.
/// </summary>
public class HLIRToLLIRLoweringTests
{
    #region Basic Lowering Tests

    [Fact]
    public void Lower_EmptyGraph_ReturnsEmptyLLIRGraph()
    {
        var hlirGraph = new HLIRGraph<double>();
        var lowering = new HLIRToLLIRLowering<double>();

        var llirGraph = lowering.Lower(hlirGraph);

        Assert.NotNull(llirGraph);
        Assert.Empty(llirGraph.Operations);
    }

    [Fact]
    public void Lower_SingleInputNode_CreatesInputBuffer()
    {
        var hlirGraph = new HLIRGraph<double>();
        var inputNode = hlirGraph.CreateNode(OperationType.Input, "input");
        inputNode.OutputType = new TensorType { Shape = new[] { 2, 3 }, DataType = IRDataType.Float32 };
        hlirGraph.InputNodes.Add(inputNode);
        hlirGraph.OutputNodes.Add(inputNode);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        Assert.Single(llirGraph.InputIds);
        // BufferShapes is keyed by LLIR buffer IDs, not HLIR node IDs
        Assert.Contains(llirGraph.InputIds[0], llirGraph.BufferShapes.Keys);
    }

    [Fact]
    public void Lower_ReLUOperation_CreatesElementwiseOp()
    {
        var hlirGraph = new HLIRGraph<double>();
        var input = hlirGraph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 2, 3 }, DataType = IRDataType.Float32 };

        var relu = hlirGraph.CreateNode(OperationType.ReLU, "relu", input);
        relu.OutputType = new TensorType { Shape = new[] { 2, 3 }, DataType = IRDataType.Float32 };

        hlirGraph.InputNodes.Add(input);
        hlirGraph.OutputNodes.Add(relu);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        Assert.Contains(llirGraph.Operations, op => op is ElementwiseOp);
        var elementwiseOp = llirGraph.Operations.OfType<ElementwiseOp>().First();
        Assert.Equal(ElementwiseOpType.ReLU, elementwiseOp.ElementwiseType);
    }

    [Fact]
    public void Lower_AddOperation_CreatesElementwiseOp()
    {
        var hlirGraph = new HLIRGraph<double>();
        var input1 = hlirGraph.CreateNode(OperationType.Input, "input1");
        input1.OutputType = new TensorType { Shape = new[] { 2, 3 }, DataType = IRDataType.Float32 };

        var input2 = hlirGraph.CreateNode(OperationType.Input, "input2");
        input2.OutputType = new TensorType { Shape = new[] { 2, 3 }, DataType = IRDataType.Float32 };

        var add = hlirGraph.CreateNode(OperationType.Add, "add", input1, input2);
        add.OutputType = new TensorType { Shape = new[] { 2, 3 }, DataType = IRDataType.Float32 };

        hlirGraph.InputNodes.Add(input1);
        hlirGraph.InputNodes.Add(input2);
        hlirGraph.OutputNodes.Add(add);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        var addOp = llirGraph.Operations.OfType<ElementwiseOp>().FirstOrDefault(op => op.ElementwiseType == ElementwiseOpType.Add);
        Assert.NotNull(addOp);
        Assert.Equal(2, addOp.InputIds.Length);
    }

    [Fact]
    public void Lower_MatMulOperation_CreatesMatMulOp()
    {
        var hlirGraph = new HLIRGraph<double>();
        var input1 = hlirGraph.CreateNode(OperationType.Input, "input1");
        input1.OutputType = new TensorType { Shape = new[] { 128, 256 }, DataType = IRDataType.Float32 };

        var input2 = hlirGraph.CreateNode(OperationType.Input, "input2");
        input2.OutputType = new TensorType { Shape = new[] { 256, 512 }, DataType = IRDataType.Float32 };

        var matmul = hlirGraph.CreateNode(OperationType.MatMul, "matmul", input1, input2);
        matmul.OutputType = new TensorType { Shape = new[] { 128, 512 }, DataType = IRDataType.Float32 };

        hlirGraph.InputNodes.Add(input1);
        hlirGraph.InputNodes.Add(input2);
        hlirGraph.OutputNodes.Add(matmul);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        var matmulOp = llirGraph.Operations.OfType<MatMulOp>().FirstOrDefault();
        Assert.NotNull(matmulOp);
        Assert.Equal(128, matmulOp.M);
        Assert.Equal(512, matmulOp.N);
        Assert.Equal(256, matmulOp.K);
    }

    [Fact]
    public void Lower_Conv2DOperation_CreatesConv2DOp()
    {
        var hlirGraph = new HLIRGraph<double>();
        var input = hlirGraph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 1, 64, 56, 56 }, DataType = IRDataType.Float32 };

        // Create kernel/weight input node
        var kernel = hlirGraph.CreateNode(OperationType.Constant, "kernel");
        kernel.OutputType = new TensorType { Shape = new[] { 128, 64, 3, 3 }, DataType = IRDataType.Float32 }; // OIHW format

        var conv = hlirGraph.CreateNode(OperationType.Conv2D, "conv", input, kernel);
        conv.OutputType = new TensorType { Shape = new[] { 1, 128, 56, 56 }, DataType = IRDataType.Float32 };
        // Add input types for proper lowering
        conv.InputTypes.Add(new TensorType { Shape = new[] { 1, 64, 56, 56 }, DataType = IRDataType.Float32 });
        conv.InputTypes.Add(new TensorType { Shape = new[] { 128, 64, 3, 3 }, DataType = IRDataType.Float32 });
        conv.Attributes["kernel_size"] = new[] { 3, 3 };
        conv.Attributes["stride"] = new[] { 1, 1 };
        conv.Attributes["padding"] = new[] { 1, 1 };
        conv.Attributes["input_channels"] = 64;
        conv.Attributes["output_channels"] = 128;

        hlirGraph.InputNodes.Add(input);
        hlirGraph.OutputNodes.Add(conv);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        var convOp = llirGraph.Operations.OfType<Conv2DOp>().FirstOrDefault();
        Assert.NotNull(convOp);
        Assert.Equal(1, convOp.BatchSize);
    }

    [Fact]
    public void Lower_ReshapeOperation_CreatesMemoryOp()
    {
        var hlirGraph = new HLIRGraph<double>();
        var input = hlirGraph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 2, 3, 4 }, DataType = IRDataType.Float32 };

        var reshape = hlirGraph.CreateNode(OperationType.Reshape, "reshape", input);
        reshape.OutputType = new TensorType { Shape = new[] { 6, 4 }, DataType = IRDataType.Float32 };

        hlirGraph.InputNodes.Add(input);
        hlirGraph.OutputNodes.Add(reshape);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        var memOp = llirGraph.Operations.OfType<MemoryOp>().FirstOrDefault();
        Assert.NotNull(memOp);
        Assert.Equal(MemoryOpType.Reshape, memOp.MemoryOpType);
    }

    [Fact]
    public void Lower_TransposeOperation_CreatesMemoryOp()
    {
        var hlirGraph = new HLIRGraph<double>();
        var input = hlirGraph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 2, 3 }, DataType = IRDataType.Float32 };

        var transpose = hlirGraph.CreateNode(OperationType.Transpose, "transpose", input);
        transpose.OutputType = new TensorType { Shape = new[] { 3, 2 }, DataType = IRDataType.Float32 };

        hlirGraph.InputNodes.Add(input);
        hlirGraph.OutputNodes.Add(transpose);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        var memOp = llirGraph.Operations.OfType<MemoryOp>().FirstOrDefault();
        Assert.NotNull(memOp);
        Assert.Equal(MemoryOpType.Transpose, memOp.MemoryOpType);
    }

    [Fact]
    public void Lower_ConstantNode_CreatesConstantOp()
    {
        var hlirGraph = new HLIRGraph<double>();
        var constant = hlirGraph.CreateNode(OperationType.Constant, "constant");
        constant.OutputType = new TensorType { Shape = new[] { 2, 3 }, DataType = IRDataType.Float32 };

        hlirGraph.OutputNodes.Add(constant);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        var constOp = llirGraph.Operations.OfType<ConstantOp>().FirstOrDefault();
        Assert.NotNull(constOp);
    }

    #endregion

    #region Activation Function Tests

    [Theory]
    [InlineData(OperationType.ReLU, ElementwiseOpType.ReLU)]
    [InlineData(OperationType.Sigmoid, ElementwiseOpType.Sigmoid)]
    [InlineData(OperationType.Tanh, ElementwiseOpType.Tanh)]
    [InlineData(OperationType.GELU, ElementwiseOpType.GELU)]
    public void Lower_ActivationFunction_CreatesCorrectElementwiseOp(OperationType hlirOp, ElementwiseOpType expectedLlirOp)
    {
        var hlirGraph = new HLIRGraph<double>();
        var input = hlirGraph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 2, 3 }, DataType = IRDataType.Float32 };

        var activation = hlirGraph.CreateNode(hlirOp, "activation", input);
        activation.OutputType = new TensorType { Shape = new[] { 2, 3 }, DataType = IRDataType.Float32 };

        hlirGraph.InputNodes.Add(input);
        hlirGraph.OutputNodes.Add(activation);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        var elementwiseOp = llirGraph.Operations.OfType<ElementwiseOp>().FirstOrDefault();
        Assert.NotNull(elementwiseOp);
        Assert.Equal(expectedLlirOp, elementwiseOp.ElementwiseType);
    }

    #endregion

    #region Fused Operation Tests

    [Fact]
    public void Lower_FusedNode_CreatesFusedOp()
    {
        var hlirGraph = new HLIRGraph<double>();
        var input = hlirGraph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 1, 64, 56, 56 }, DataType = IRDataType.Float32 };

        // Create original nodes that will be marked as fused
        var conv = new HLIRNode<double>
        {
            Id = 10,
            Name = "conv",
            Operation = OperationType.Conv2D,
            OutputType = new TensorType { Shape = new[] { 1, 128, 56, 56 }, DataType = IRDataType.Float32 },
            InputTypes = new List<TensorType>
            {
                new TensorType { Shape = new[] { 1, 64, 56, 56 }, DataType = IRDataType.Float32 },
                new TensorType { Shape = new[] { 128, 64, 3, 3 }, DataType = IRDataType.Float32 }
            }
        };
        var relu = new HLIRNode<double>
        {
            Id = 11,
            Name = "relu",
            Operation = OperationType.ReLU,
            OutputType = new TensorType { Shape = new[] { 1, 128, 56, 56 }, DataType = IRDataType.Float32 }
        };

        // Create fused node
        var fused = hlirGraph.CreateNode(OperationType.Add, "fused", input);
        fused.OutputType = new TensorType { Shape = new[] { 1, 128, 56, 56 }, DataType = IRDataType.Float32 };
        fused.IsFused = true;
        fused.FusedFrom = new List<HLIRNode<double>> { conv, relu };

        hlirGraph.InputNodes.Add(input);
        hlirGraph.OutputNodes.Add(fused);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        var fusedOp = llirGraph.Operations.OfType<FusedOp>().FirstOrDefault();
        Assert.NotNull(fusedOp);
        Assert.Contains("Conv2D", fusedOp.FusionPattern);
        Assert.Contains("ReLU", fusedOp.FusionPattern);
    }

    #endregion

    #region Complex Graph Tests

    [Fact]
    public void Lower_LinearSequence_PreservesOrder()
    {
        var hlirGraph = new HLIRGraph<double>();

        var input = hlirGraph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 10 }, DataType = IRDataType.Float32 };

        var relu = hlirGraph.CreateNode(OperationType.ReLU, "relu", input);
        relu.OutputType = new TensorType { Shape = new[] { 10 }, DataType = IRDataType.Float32 };

        var sigmoid = hlirGraph.CreateNode(OperationType.Sigmoid, "sigmoid", relu);
        sigmoid.OutputType = new TensorType { Shape = new[] { 10 }, DataType = IRDataType.Float32 };

        hlirGraph.InputNodes.Add(input);
        hlirGraph.OutputNodes.Add(sigmoid);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        // Should have at least 2 elementwise operations (ReLU and Sigmoid)
        var elementwiseOps = llirGraph.Operations.OfType<ElementwiseOp>().ToList();
        Assert.True(elementwiseOps.Count >= 2);
    }

    [Fact]
    public void Lower_BranchingGraph_HandlesMultipleOutputs()
    {
        var hlirGraph = new HLIRGraph<double>();

        var input = hlirGraph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 10 }, DataType = IRDataType.Float32 };

        var relu = hlirGraph.CreateNode(OperationType.ReLU, "relu", input);
        relu.OutputType = new TensorType { Shape = new[] { 10 }, DataType = IRDataType.Float32 };

        var sigmoid = hlirGraph.CreateNode(OperationType.Sigmoid, "sigmoid", input);
        sigmoid.OutputType = new TensorType { Shape = new[] { 10 }, DataType = IRDataType.Float32 };

        hlirGraph.InputNodes.Add(input);
        hlirGraph.OutputNodes.Add(relu);
        hlirGraph.OutputNodes.Add(sigmoid);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        Assert.Equal(2, llirGraph.OutputIds.Count);
    }

    #endregion

    #region Data Type Preservation Tests

    [Theory]
    [InlineData(IRDataType.Float32)]
    [InlineData(IRDataType.Float64)]
    [InlineData(IRDataType.Float16)]
    public void Lower_PreservesDataType(IRDataType dataType)
    {
        var hlirGraph = new HLIRGraph<double>();
        var input = hlirGraph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 10 }, DataType = dataType };

        var relu = hlirGraph.CreateNode(OperationType.ReLU, "relu", input);
        relu.OutputType = new TensorType { Shape = new[] { 10 }, DataType = dataType };

        hlirGraph.InputNodes.Add(input);
        hlirGraph.OutputNodes.Add(relu);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        var elementwiseOp = llirGraph.Operations.OfType<ElementwiseOp>().FirstOrDefault();
        Assert.NotNull(elementwiseOp);
        Assert.Equal(dataType, elementwiseOp.OutputDataType);
    }

    #endregion

    #region Provenance Tracking Tests

    [Fact]
    public void Lower_PreservesSourceNodeId()
    {
        var hlirGraph = new HLIRGraph<double>();
        var input = hlirGraph.CreateNode(OperationType.Input, "input");
        input.OutputType = new TensorType { Shape = new[] { 10 }, DataType = IRDataType.Float32 };

        var relu = hlirGraph.CreateNode(OperationType.ReLU, "relu", input);
        relu.OutputType = new TensorType { Shape = new[] { 10 }, DataType = IRDataType.Float32 };

        hlirGraph.InputNodes.Add(input);
        hlirGraph.OutputNodes.Add(relu);

        var lowering = new HLIRToLLIRLowering<double>();
        var llirGraph = lowering.Lower(hlirGraph);

        var elementwiseOp = llirGraph.Operations.OfType<ElementwiseOp>().FirstOrDefault();
        Assert.NotNull(elementwiseOp);
        Assert.Equal(relu.Id, elementwiseOp.SourceHLIRNodeId);
    }

    #endregion
}
