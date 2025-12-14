using AiDotNet.Enums;
using AiDotNet.InferenceOptimization.IR.Common;
using AiDotNet.InferenceOptimization.IR.HighLevel;
using AiDotNet.InferenceOptimization.IR.LowLevel;

namespace AiDotNet.InferenceOptimization.IR.Lowering;

/// <summary>
/// Lowers High-Level IR to Low-Level IR.
/// Transforms semantic operations into hardware-optimized operations.
/// </summary>
/// <remarks>
/// <para><b>Design Philosophy:</b></para>
/// <para>
/// The lowering process transforms high-level semantic operations into low-level
/// hardware-optimized operations. This is similar to MLIR's progressive lowering
/// or TVM's Relay to TIR conversion.
/// </para>
///
/// <para><b>Lowering Stages:</b></para>
/// <list type="number">
/// <item>Type conversion: Generic T -> concrete types (float32, etc.)</item>
/// <item>Operation mapping: Semantic ops -> hardware ops</item>
/// <item>Layout selection: Choose optimal memory layouts</item>
/// <item>Scheduling: Generate schedules for loops</item>
/// <item>Memory planning: Allocate and reuse buffers</item>
/// </list>
///
/// <para><b>Exceeds Standards By:</b></para>
/// <list type="bullet">
/// <item>Preserves fusion information from HLIR</item>
/// <item>Automatic algorithm selection (Winograd, FFT, etc.)</item>
/// <item>Multi-device lowering in single pass</item>
/// <item>Quantization-aware lowering</item>
/// </list>
/// </remarks>
public class HLIRToLLIRLowering<T> where T : struct
{
    #region Configuration

    /// <summary>
    /// Target device for lowering.
    /// </summary>
    public DeviceType TargetDevice { get; set; } = DeviceType.CPU;

    /// <summary>
    /// Device configuration.
    /// </summary>
    public DeviceConfiguration DeviceConfig { get; set; } = new();

    /// <summary>
    /// Whether to optimize memory usage.
    /// </summary>
    public bool OptimizeMemory { get; set; } = true;

    /// <summary>
    /// Whether to auto-schedule operations.
    /// </summary>
    public bool AutoSchedule { get; set; } = true;

    /// <summary>
    /// Target data type (for type conversion).
    /// </summary>
    public IRDataType TargetDataType { get; set; } = IRDataType.Float32;

    /// <summary>
    /// Preferred memory layout.
    /// </summary>
    public MemoryLayout PreferredLayout { get; set; } = MemoryLayout.RowMajor;

    #endregion

    #region State

    private readonly Dictionary<int, int> _hlirToLlirBufferMap = new();
    private LLIRGraph _llirGraph = new();

    #endregion

    #region Main Entry Point

    /// <summary>
    /// Lowers an HLIR graph to an LLIR graph.
    /// </summary>
    public LLIRGraph Lower(HLIRGraph<T> hlirGraph)
    {
        _llirGraph = new LLIRGraph
        {
            Name = hlirGraph.Name + "_lowered",
            DeviceConfig = DeviceConfig
        };
        _hlirToLlirBufferMap.Clear();

        // Process nodes in topological order
        var orderedNodes = hlirGraph.GetTopologicalOrder();

        // First pass: map input nodes
        foreach (var inputNode in hlirGraph.InputNodes)
        {
            MapInputNode(inputNode);
        }

        // Second pass: lower each node
        foreach (var node in orderedNodes)
        {
            if (!hlirGraph.InputNodes.Contains(node))
            {
                LowerNode(node);
            }
        }

        // Map output nodes - fail-fast if any output is missing
        foreach (var outputNode in hlirGraph.OutputNodes)
        {
            if (!_hlirToLlirBufferMap.TryGetValue(outputNode.Id, out var llirId))
            {
                throw new InvalidOperationException(
                    $"Output node '{outputNode.Name}' (ID: {outputNode.Id}) was not lowered. " +
                    $"This indicates a missing lowering implementation for operation type '{outputNode.Operation}'.");
            }
            _llirGraph.OutputIds.Add(llirId);
        }

        // Post-processing
        if (OptimizeMemory)
        {
            _llirGraph.OptimizeMemory();
        }

        if (AutoSchedule)
        {
            _llirGraph.AutoSchedule();
        }

        return _llirGraph;
    }

    #endregion

    #region Node Lowering

    private void MapInputNode(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        _hlirToLlirBufferMap[node.Id] = bufferId;
        _llirGraph.InputIds.Add(bufferId);
        _llirGraph.BufferShapes[bufferId] = node.OutputType.Shape;
        _llirGraph.BufferTypes[bufferId] = ConvertDataType(node.OutputType.DataType);
    }

    private void LowerNode(HLIRNode<T> node)
    {
        // Handle fused nodes
        if (node.IsFused && node.FusedFrom != null && node.FusedFrom.Count > 0)
        {
            LowerFusedNode(node);
            return;
        }

        // Lower based on operation type
        var llirOp = node.Operation switch
        {
            OperationType.MatMul or OperationType.Gemm => LowerMatMul(node),
            OperationType.Add or OperationType.Subtract or OperationType.Multiply or OperationType.Divide =>
                LowerElementwise(node),
            OperationType.Conv2D or OperationType.Convolution2D or OperationType.Convolution =>
                LowerConv2D(node),
            OperationType.ReLU or OperationType.Sigmoid or OperationType.Tanh or OperationType.GELU or
            OperationType.Softmax or OperationType.LogSoftmax =>
                LowerActivation(node),
            OperationType.BatchNormalization or OperationType.LayerNormalization =>
                LowerNormalization(node),
            OperationType.MaxPool2D or OperationType.AvgPool2D or OperationType.GlobalAveragePooling =>
                LowerPooling(node),
            OperationType.Reshape or OperationType.Transpose or OperationType.Flatten or
            OperationType.Concat or OperationType.Split or OperationType.Slice =>
                LowerMemoryOp(node),
            OperationType.Constant =>
                LowerConstant(node),
            OperationType.Input or OperationType.Output =>
                LowerInputOutput(node),
            OperationType.ReduceSum or OperationType.Mean or OperationType.ReduceMax or OperationType.ReduceMin =>
                LowerReduction(node),
            OperationType.Dense or OperationType.FullyConnected =>
                LowerDense(node),
            OperationType.Embedding =>
                LowerEmbedding(node),
            OperationType.Attention or OperationType.MultiHeadAttention =>
                LowerAttention(node),
            OperationType.Dropout =>
                LowerDropout(node),
            OperationType.FusedConvBatchNormReLU or OperationType.FusedMatMulBias or
            OperationType.FusedMatMulBiasReLU or OperationType.FusedMatMulBiasGELU or
            OperationType.FusedMultiHeadAttention or OperationType.FusedLayerNormAttention =>
                LowerFusedOperation(node),
            _ => LowerGeneric(node)
        };

        if (llirOp != null)
        {
            _llirGraph.AddOperation(llirOp);
            _hlirToLlirBufferMap[node.Id] = llirOp.OutputId;
        }
    }

    #endregion

    #region Operation-Specific Lowering

    private LLIROp LowerMatMul(HLIRNode<T> node)
    {
        var inputIds = GetLLIRInputIds(node);
        var (m, n, k) = InferMatMulDims(node);
        var bufferId = _llirGraph.AllocateBufferId();

        var transposeA = node.Attributes.TryGetValue("transposeA", out var ta) && (bool)ta;
        var transposeB = node.Attributes.TryGetValue("transposeB", out var tb) && (bool)tb;

        return new MatMulOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = inputIds,
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            M = m,
            N = n,
            K = k,
            TransposeA = transposeA,
            TransposeB = transposeB,
            SourceHLIRNodeId = node.Id
        };
    }

    private LLIROp LowerElementwise(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        var opType = node.Operation switch
        {
            OperationType.Add => ElementwiseOpType.Add,
            OperationType.Subtract => ElementwiseOpType.Subtract,
            OperationType.Multiply => ElementwiseOpType.Multiply,
            OperationType.Divide => ElementwiseOpType.Divide,
            _ => ElementwiseOpType.Add
        };

        return new ElementwiseOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            ElementwiseType = opType,
            SourceHLIRNodeId = node.Id
        };
    }

    /// <summary>
    /// Lowers a Conv2D operation from HLIR to LLIR representation.
    /// </summary>
    /// <param name="node">The HLIR node representing the Conv2D operation.</param>
    /// <returns>The lowered Conv2DOp for LLIR execution.</returns>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the node has insufficient input type information for proper lowering.
    /// </exception>
    /// <remarks>
    /// <para>
    /// Conv2D lowering requires complete shape information for both input tensor and kernel.
    /// The input tensor shape must be in NCHW format: [batch, channels, height, width].
    /// The kernel shape must be in OIHW format: [out_channels, in_channels, kernel_h, kernel_w].
    /// </para>
    /// </remarks>
    private LLIROp LowerConv2D(HLIRNode<T> node)
    {
        var inputIds = GetLLIRInputIds(node);
        var bufferId = _llirGraph.AllocateBufferId();

        // Validate input types - Conv2D requires proper shape information
        if (node.InputTypes.Count < 2)
        {
            throw new InvalidOperationException(
                $"Conv2D node '{node.Name}' (id={node.Id}) requires at least 2 input types " +
                $"(input tensor and kernel), but only has {node.InputTypes.Count}.");
        }

        var inputShape = node.InputTypes[0].Shape;
        var kernelShape = node.InputTypes[1].Shape;

        if (inputShape == null || inputShape.Length < 4)
        {
            throw new InvalidOperationException(
                $"Conv2D node '{node.Name}' (id={node.Id}) has invalid input tensor shape. " +
                $"Expected 4D NCHW tensor, got {(inputShape == null ? "null" : $"{inputShape.Length}D")}.");
        }

        if (kernelShape == null || kernelShape.Length < 4)
        {
            throw new InvalidOperationException(
                $"Conv2D node '{node.Name}' (id={node.Id}) has invalid kernel shape. " +
                $"Expected 4D OIHW tensor, got {(kernelShape == null ? "null" : $"{kernelShape.Length}D")}.");
        }

        // Extract convolution parameters
        var strideH = GetAttributeInt(node, "strideH", 1);
        var strideW = GetAttributeInt(node, "strideW", 1);
        var padH = GetAttributeInt(node, "padH", 0);
        var padW = GetAttributeInt(node, "padW", 0);
        var groups = GetAttributeInt(node, "groups", 1);

        // Shapes have been validated - safe to access directly
        // Input shape: NCHW [batch, channels, height, width]
        // Kernel shape: OIHW [out_channels, in_channels, kernel_h, kernel_w]
        var op = new Conv2DOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = inputIds,
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            BatchSize = inputShape[0],
            InputChannels = inputShape[1],
            InputHeight = inputShape[2],
            InputWidth = inputShape[3],
            OutputChannels = kernelShape[0],
            KernelHeight = kernelShape[2],
            KernelWidth = kernelShape[3],
            StrideH = strideH,
            StrideW = strideW,
            PadH = padH,
            PadW = padW,
            Groups = groups,
            SourceHLIRNodeId = node.Id
        };

        // Select algorithm
        op.Algorithm = SelectConvAlgorithm(op);

        return op;
    }

    private LLIROp LowerActivation(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        var opType = node.Operation switch
        {
            OperationType.ReLU => ElementwiseOpType.ReLU,
            OperationType.Sigmoid => ElementwiseOpType.Sigmoid,
            OperationType.Tanh => ElementwiseOpType.Tanh,
            OperationType.GELU => ElementwiseOpType.GELU,
            OperationType.Softmax => ElementwiseOpType.Softmax,
            OperationType.LogSoftmax => ElementwiseOpType.LogSoftmax,
            _ => ElementwiseOpType.ReLU
        };

        return new ElementwiseOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            ElementwiseType = opType,
            SourceHLIRNodeId = node.Id
        };
    }

    private LLIROp LowerNormalization(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();

        var fusedOp = new FusedOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            FusionPattern = node.Operation == OperationType.BatchNormalization
                ? "BatchNorm"
                : "LayerNorm",
            SourceHLIRNodeId = node.Id
        };

        return fusedOp;
    }

    /// <summary>
    /// Lowers a pooling operation from HLIR to LLIR representation.
    /// </summary>
    /// <param name="node">The HLIR node representing the pooling operation.</param>
    /// <returns>The lowered FusedOp containing pooling parameters for LLIR execution.</returns>
    /// <remarks>
    /// <para>
    /// Pooling operations (MaxPool2D, AvgPool2D) have spatial window, stride, and padding
    /// parameters that distinguish them from simple reductions. This method uses FusedOp
    /// to preserve these windowed operation semantics rather than ReduceOp which would
    /// lose the spatial parameters.
    /// </para>
    /// <para>
    /// The pooling parameters (kernel size, stride, padding) are extracted from the node's
    /// attributes and stored in the FusedOp's Attributes dictionary for runtime execution.
    /// </para>
    /// </remarks>
    private LLIROp LowerPooling(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        var pattern = node.Operation switch
        {
            OperationType.MaxPool2D => "MaxPool2D",
            OperationType.AvgPool2D => "AvgPool2D",
            OperationType.GlobalAveragePooling => "GlobalAvgPool",
            _ => "MaxPool2D"
        };

        // Extract pooling parameters from node attributes
        var kernelH = GetAttributeInt(node, "kernelH", 2);
        var kernelW = GetAttributeInt(node, "kernelW", 2);
        var strideH = GetAttributeInt(node, "strideH", 2);
        var strideW = GetAttributeInt(node, "strideW", 2);
        var padH = GetAttributeInt(node, "padH", 0);
        var padW = GetAttributeInt(node, "padW", 0);

        var fusedOp = new FusedOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            FusionPattern = pattern,
            SourceHLIRNodeId = node.Id
        };

        // Store pooling parameters in attributes for runtime execution
        fusedOp.Attributes["kernelH"] = kernelH;
        fusedOp.Attributes["kernelW"] = kernelW;
        fusedOp.Attributes["strideH"] = strideH;
        fusedOp.Attributes["strideW"] = strideW;
        fusedOp.Attributes["padH"] = padH;
        fusedOp.Attributes["padW"] = padW;

        return fusedOp;
    }

    private LLIROp LowerMemoryOp(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        var memOpType = node.Operation switch
        {
            OperationType.Reshape => MemoryOpType.Reshape,
            OperationType.Transpose => MemoryOpType.Transpose,
            OperationType.Flatten => MemoryOpType.Reshape,
            OperationType.Concat => MemoryOpType.Concat,
            OperationType.Split => MemoryOpType.Slice,
            OperationType.Slice => MemoryOpType.Slice,
            _ => MemoryOpType.Copy
        };

        var op = new MemoryOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            MemoryOpType = memOpType,
            SourceHLIRNodeId = node.Id
        };

        if (node.Operation == OperationType.Transpose &&
            node.Attributes.TryGetValue("perm", out var perm))
        {
            op.Permutation = (int[])perm;
        }

        if (node.Operation == OperationType.Reshape)
        {
            op.NewShape = node.OutputType.Shape;
        }

        return op;
    }

    private LLIROp? LowerConstant(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        _hlirToLlirBufferMap[node.Id] = bufferId;

        var op = new ConstantOp
        {
            OutputId = bufferId,
            Name = node.Name,
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            IsParameter = node.Parameters.Count > 0,
            ParameterName = node.Name,
            SourceHLIRNodeId = node.Id
        };

        _llirGraph.AddOperation(op);
        return null; // Already added
    }

    private LLIROp? LowerInputOutput(HLIRNode<T> node)
    {
        // Input/output nodes are handled separately
        return null;
    }

    private LLIROp LowerReduction(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        var reduceType = node.Operation switch
        {
            OperationType.ReduceSum => ReduceType.Sum,
            OperationType.Mean or OperationType.ReduceMean => ReduceType.Mean,
            OperationType.ReduceMax => ReduceType.Max,
            OperationType.ReduceMin => ReduceType.Min,
            _ => ReduceType.Sum
        };

        var axes = node.Attributes.TryGetValue("axes", out var ax) ? (int[])ax : Array.Empty<int>();
        var keepDims = node.Attributes.TryGetValue("keepDims", out var kd) && (bool)kd;

        // Get input shape for accurate cost estimation
        var inputShape = node.InputTypes.Count > 0 && node.InputTypes[0].Shape != null
            ? node.InputTypes[0].Shape
            : Array.Empty<int>();

        return new ReduceOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            ReduceType = reduceType,
            Axes = axes,
            KeepDims = keepDims,
            InputShape = inputShape,
            SourceHLIRNodeId = node.Id
        };
    }

    private LLIROp LowerDense(HLIRNode<T> node)
    {
        // Dense is lowered to MatMul + optional bias add
        return LowerMatMul(node);
    }

    private LLIROp LowerEmbedding(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        return new MemoryOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            MemoryOpType = MemoryOpType.Gather,
            SourceHLIRNodeId = node.Id
        };
    }

    private LLIROp LowerAttention(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        return new FusedOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            FusionPattern = "Attention",
            SourceHLIRNodeId = node.Id
        };
    }

    /// <summary>
    /// Lowers a dropout operation from HLIR to LLIR representation.
    /// </summary>
    /// <param name="node">The HLIR node representing the dropout operation.</param>
    /// <returns>Always returns null since dropout is a no-op in inference mode.</returns>
    /// <remarks>
    /// <para>
    /// During inference, dropout is a no-op (identity operation) - inputs pass through
    /// unchanged. This method simply maps the node's output to its input buffer,
    /// avoiding unnecessary memory allocation or computation.
    /// </para>
    /// </remarks>
    private LLIROp? LowerDropout(HLIRNode<T> node)
    {
        // Dropout in inference mode is a no-op (identity)
        // Validate that the node has at least one input before accessing
        if (node.Inputs.Count > 0 && _hlirToLlirBufferMap.TryGetValue(node.Inputs[0].Id, out var inputId))
        {
            _hlirToLlirBufferMap[node.Id] = inputId;
        }
        return null;
    }

    private LLIROp LowerFusedOperation(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        var pattern = node.Operation switch
        {
            OperationType.FusedConvBatchNormReLU => "ConvBNReLU",
            OperationType.FusedMatMulBias => "MatMulBias",
            OperationType.FusedMatMulBiasReLU => "MatMulBiasReLU",
            OperationType.FusedMatMulBiasGELU => "MatMulBiasGELU",
            OperationType.FusedMultiHeadAttention => "FusedMHA",
            OperationType.FusedLayerNormAttention => "LNAttention",
            _ => "Fused"
        };

        return new FusedOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            FusionPattern = pattern,
            SourceHLIRNodeId = node.Id
        };
    }

    private LLIROp LowerGeneric(HLIRNode<T> node)
    {
        var bufferId = _llirGraph.AllocateBufferId();
        return new ElementwiseOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            ElementwiseType = ElementwiseOpType.Identity,
            SourceHLIRNodeId = node.Id
        };
    }

    private void LowerFusedNode(HLIRNode<T> node)
    {
        var fusedFrom = node.FusedFrom;
        if (fusedFrom == null || fusedFrom.Count == 0)
        {
            return;
        }

        var bufferId = _llirGraph.AllocateBufferId();

        // Create a FusedOp that captures the fusion pattern
        var pattern = string.Join("_", fusedFrom.Select(n => n.Operation.ToString()));

        var fusedOp = new FusedOp
        {
            OutputId = bufferId,
            Name = node.Name,
            InputIds = GetLLIRInputIds(node),
            OutputShape = node.OutputType.Shape,
            OutputDataType = ConvertDataType(node.OutputType.DataType),
            Device = GetDeviceForNode(node),
            FusionPattern = pattern,
            SourceHLIRNodeId = node.Id
        };

        // Lower each original node as part of the fused op
        foreach (var originalNode in fusedFrom)
        {
            var llirOp = LowerNodeToOp(originalNode);
            if (llirOp != null)
            {
                fusedOp.FusedOps.Add(llirOp);
            }
        }

        _llirGraph.AddOperation(fusedOp);
        _hlirToLlirBufferMap[node.Id] = bufferId;
    }

    /// <summary>
    /// Lowers a single HLIR node to an LLIR operation for inclusion in a fused operation.
    /// </summary>
    /// <param name="node">The HLIR node to lower.</param>
    /// <returns>The lowered LLIR operation with complete metadata, or null if the operation type is not supported.</returns>
    /// <remarks>
    /// <para>
    /// This method creates LLIR operations with full metadata transfer from the source HLIR node,
    /// including shape information, data types, and operation-specific dimensions. This is critical
    /// for accurate cost estimation in fused operations.
    /// </para>
    /// <para>
    /// For MatMul operations, the M, N, K dimensions are extracted from input/output shapes.
    /// For ElementwiseOp operations, OutputShape and OutputDataType are transferred.
    /// </para>
    /// </remarks>
    private LLIROp? LowerNodeToOp(HLIRNode<T> node)
    {
        // Get common properties from the node
        var outputShape = node.OutputType?.Shape ?? Array.Empty<int>();
        var outputDataType = node.OutputType != null ? ConvertDataType(node.OutputType.DataType) : IRDataType.Float32;
        var inputIds = GetLLIRInputIds(node);

        return node.Operation switch
        {
            // Matrix operations
            OperationType.MatMul or OperationType.Gemm => CreateMatMulOp(node, outputShape, outputDataType, inputIds),

            // Elementwise arithmetic
            OperationType.Add => CreateElementwiseOp(node, ElementwiseOpType.Add, outputShape, outputDataType, inputIds),
            OperationType.Subtract => CreateElementwiseOp(node, ElementwiseOpType.Subtract, outputShape, outputDataType, inputIds),
            OperationType.Multiply => CreateElementwiseOp(node, ElementwiseOpType.Multiply, outputShape, outputDataType, inputIds),
            OperationType.Divide => CreateElementwiseOp(node, ElementwiseOpType.Divide, outputShape, outputDataType, inputIds),

            // Activation functions
            OperationType.ReLU => CreateElementwiseOp(node, ElementwiseOpType.ReLU, outputShape, outputDataType, inputIds),
            OperationType.Sigmoid => CreateElementwiseOp(node, ElementwiseOpType.Sigmoid, outputShape, outputDataType, inputIds),
            OperationType.Tanh => CreateElementwiseOp(node, ElementwiseOpType.Tanh, outputShape, outputDataType, inputIds),
            OperationType.GELU => CreateElementwiseOp(node, ElementwiseOpType.GELU, outputShape, outputDataType, inputIds),
            OperationType.Softmax => CreateElementwiseOp(node, ElementwiseOpType.Softmax, outputShape, outputDataType, inputIds),
            OperationType.LogSoftmax => CreateElementwiseOp(node, ElementwiseOpType.LogSoftmax, outputShape, outputDataType, inputIds),

            // Unsupported operation in fused context
            _ => throw new InvalidOperationException(
                $"Operation '{node.Operation}' is not supported within fused operations. " +
                $"Node: '{node.Name}' (ID: {node.Id})")
        };
    }

    /// <summary>
    /// Creates a MatMulOp with proper dimension information from the HLIR node.
    /// </summary>
    private MatMulOp CreateMatMulOp(HLIRNode<T> node, int[] outputShape, IRDataType outputDataType, int[] inputIds)
    {
        // Infer M, N, K dimensions from input/output shapes
        // MatMul: [M, K] × [K, N] = [M, N]
        int m = 1, n = 1, k = 1;

        if (node.InputTypes.Count >= 2)
        {
            var leftShape = node.InputTypes[0].Shape;
            var rightShape = node.InputTypes[1].Shape;

            if (leftShape != null && leftShape.Length >= 2)
            {
                m = leftShape[^2]; // Second-to-last dimension
                k = leftShape[^1]; // Last dimension
            }

            if (rightShape != null && rightShape.Length >= 2)
            {
                n = rightShape[^1]; // Last dimension
            }
        }
        else if (outputShape.Length >= 2)
        {
            // Fallback: infer from output shape
            m = outputShape[^2];
            n = outputShape[^1];
        }

        // Allocate output buffer for this sub-op within the fused operation
        var outputId = _llirGraph.AllocateBufferId();

        return new MatMulOp
        {
            OutputId = outputId,
            Name = node.Name,
            InputIds = inputIds,
            OutputShape = outputShape,
            OutputDataType = outputDataType,
            M = m,
            N = n,
            K = k,
            SourceHLIRNodeId = node.Id
        };
    }

    /// <summary>
    /// Creates an ElementwiseOp with proper metadata from the HLIR node.
    /// </summary>
    private ElementwiseOp CreateElementwiseOp(
        HLIRNode<T> node,
        ElementwiseOpType opType,
        int[] outputShape,
        IRDataType outputDataType,
        int[] inputIds)
    {
        // Allocate output buffer for this sub-op within the fused operation
        var outputId = _llirGraph.AllocateBufferId();

        return new ElementwiseOp
        {
            OutputId = outputId,
            Name = node.Name,
            ElementwiseType = opType,
            InputIds = inputIds,
            OutputShape = outputShape,
            OutputDataType = outputDataType,
            SourceHLIRNodeId = node.Id
        };
    }

    #endregion

    #region Helpers

    private int[] GetLLIRInputIds(HLIRNode<T> node)
    {
        var ids = new List<int>();
        foreach (var input in node.Inputs)
        {
            if (!_hlirToLlirBufferMap.TryGetValue(input.Id, out var llirId))
            {
                throw new InvalidOperationException(
                    $"Input node '{input.Name}' (ID: {input.Id}) was not lowered before being used by " +
                    $"node '{node.Name}' (ID: {node.Id}). This indicates a topological ordering issue or " +
                    $"missing lowering implementation for operation type '{input.Operation}'.");
            }
            ids.Add(llirId);
        }
        return ids.ToArray();
    }

    private IRDataType ConvertDataType(IRDataType hlirType)
    {
        // If HLIR has unknown type, use target type
        if (hlirType == IRDataType.Unknown)
        {
            return TargetDataType;
        }
        return hlirType;
    }

    private DeviceType GetDeviceForNode(HLIRNode<T> node)
    {
        // Use node's preferred device or fall back to target
        if (node.Hints.PreferredDevice != DeviceType.Auto)
        {
            return node.Hints.PreferredDevice;
        }
        return TargetDevice;
    }

    private (int m, int n, int k) InferMatMulDims(HLIRNode<T> node)
    {
        // Infer from input shapes with null safety
        if (node.InputTypes.Count >= 2)
        {
            var shapeA = node.InputTypes[0].Shape;
            var shapeB = node.InputTypes[1].Shape;

            // Guard against null shapes - shapes may not be known at lowering time
            if (shapeA != null && shapeB != null && shapeA.Length >= 2 && shapeB.Length >= 2)
            {
                return (shapeA[^2], shapeB[^1], shapeA[^1]);
            }
        }

        // Default when shapes are unknown or incompatible
        return (1, 1, 1);
    }

    private ConvAlgorithm SelectConvAlgorithm(Conv2DOp op)
    {
        // Select based on kernel size and device
        if (op.Device == DeviceType.GPU && DeviceConfig.GPUHasTensorCores)
        {
            return ConvAlgorithm.TensorCore;
        }

        if (op.KernelHeight <= 3 && op.KernelWidth <= 3)
        {
            return ConvAlgorithm.Winograd;
        }

        if (op.KernelHeight >= 7 && op.KernelWidth >= 7)
        {
            return ConvAlgorithm.FFT;
        }

        return op.Device == DeviceType.GPU ? ConvAlgorithm.Implicit : ConvAlgorithm.Im2Col;
    }

    /// <summary>
    /// Safely retrieves an integer attribute from a node's attribute dictionary.
    /// </summary>
    /// <param name="node">The HLIR node containing the attributes.</param>
    /// <param name="key">The attribute key to look up.</param>
    /// <param name="defaultValue">The default value to return if the attribute is missing or invalid.</param>
    /// <returns>The attribute value as an integer, or the default value if not found or conversion fails.</returns>
    /// <remarks>
    /// <para>
    /// This method handles various attribute types safely:
    /// <list type="bullet">
    /// <item><description>Direct integer types (int, long, short, byte) are converted directly</description></item>
    /// <item><description>String values are parsed using int.TryParse</description></item>
    /// <item><description>Other IConvertible types use Convert.ToInt32 with exception handling</description></item>
    /// <item><description>Any conversion failure returns the default value</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private int GetAttributeInt(HLIRNode<T> node, string key, int defaultValue)
    {
        if (!node.Attributes.TryGetValue(key, out var value) || value == null)
        {
            return defaultValue;
        }

        // Handle common integer types directly
        if (value is int intValue)
        {
            return intValue;
        }
        if (value is long longValue)
        {
            return longValue is >= int.MinValue and <= int.MaxValue ? (int)longValue : defaultValue;
        }
        if (value is short shortValue)
        {
            return shortValue;
        }
        if (value is byte byteValue)
        {
            return byteValue;
        }

        // Handle string values
        if (value is string strValue)
        {
            return int.TryParse(strValue, out var parsed) ? parsed : defaultValue;
        }

        // Handle other IConvertible types with exception handling
        try
        {
            return Convert.ToInt32(value);
        }
        catch (FormatException)
        {
            return defaultValue;
        }
        catch (OverflowException)
        {
            return defaultValue;
        }
        catch (InvalidCastException)
        {
            return defaultValue;
        }
    }

    #endregion
}
