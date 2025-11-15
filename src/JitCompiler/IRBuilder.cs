using AiDotNet.Autodiff;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler;

/// <summary>
/// Builds an IR graph from a ComputationNode graph.
/// </summary>
/// <remarks>
/// <para>
/// The IRBuilder converts a high-level ComputationNode graph (produced by autodiff)
/// into a low-level IR graph suitable for optimization and compilation. It traverses
/// the computation graph, converts each node to an IR operation, and builds the
/// complete IR representation.
/// </para>
/// <para><b>For Beginners:</b> This translates autodiff graphs into a form the JIT compiler can work with.
///
/// Think of it like translating a recipe:
/// - Input: ComputationNode graph (high-level description of what to compute)
/// - Output: IR graph (low-level description ready for optimization)
///
/// The IRBuilder:
/// - Walks through all the computation nodes
/// - Identifies what operation each node represents
/// - Creates corresponding IR operations
/// - Builds a complete IR graph with inputs, operations, and outputs
///
/// This IR graph can then be optimized and compiled to fast executable code.
/// </para>
/// </remarks>
public class IRBuilder
{
    private int _nextTensorId = 0;
    private readonly Dictionary<object, int> _nodeToTensorId = new();

    /// <summary>
    /// Builds an IR graph from a ComputationNode graph.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the computation.</typeparam>
    /// <param name="outputNode">The output node of the computation graph.</param>
    /// <param name="inputs">The input nodes to the computation graph.</param>
    /// <returns>An IR graph representing the computation.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a topological traversal of the computation graph,
    /// converting each ComputationNode to an IROp and building the complete IR graph.
    /// It handles input mapping, operation conversion, and output identification.
    /// </para>
    /// <para><b>For Beginners:</b> This converts a computation graph to IR format.
    ///
    /// The process:
    /// 1. Identifies all input nodes and assigns them tensor IDs
    /// 2. Traverses the graph in topological order (inputs to outputs)
    /// 3. Converts each node to an IR operation
    /// 4. Builds the final IR graph with all operations connected
    ///
    /// Example:
    /// If you have a graph: result = ReLU(MatMul(input, weights) + bias)
    /// This will create an IR graph with:
    /// - Input tensors: input (t0), weights (t1), bias (t2)
    /// - Operations: MatMul (t3 = MatMul(t0, t1)), Add (t4 = Add(t3, t2)), ReLU (t5 = ReLU(t4))
    /// - Output: t5
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown if a node doesn't have operation type metadata or uses an unsupported operation.
    /// </exception>
    public IRGraph Build<T>(ComputationNode<T> outputNode, List<ComputationNode<T>> inputs)
    {
        var graph = new IRGraph();
        _nextTensorId = 0;
        _nodeToTensorId.Clear();

        // Assign tensor IDs to inputs
        foreach (var input in inputs)
        {
            var tensorId = _nextTensorId++;
            _nodeToTensorId[input] = tensorId;
            graph.InputIds.Add(tensorId);
            graph.TensorShapes[tensorId] = input.Value.Shape;
        }

        // Perform topological sort to process nodes in order
        var topoOrder = TopologicalSort(outputNode);

        // Convert each node to an IR operation
        foreach (var node in topoOrder)
        {
            // Skip input nodes (already processed)
            if (inputs.Contains(node))
            {
                continue;
            }

            // Convert node to IR operation
            var op = ConvertNodeToOp(node);
            if (op != null)
            {
                graph.Operations.Add(op);
                graph.TensorShapes[op.OutputId] = op.OutputShape;
            }
        }

        // Mark output
        if (_nodeToTensorId.TryGetValue(outputNode, out var outputId))
        {
            graph.OutputIds.Add(outputId);
        }

        return graph;
    }

    /// <summary>
    /// Converts a ComputationNode to an IR operation.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the computation.</typeparam>
    /// <param name="node">The computation node to convert.</param>
    /// <returns>An IR operation, or null if the node is an input.</returns>
    /// <remarks>
    /// <para>
    /// This method examines the node's OperationType property and creates the corresponding
    /// IR operation. It also extracts any operation-specific parameters from OperationParams
    /// and sets up input/output tensor IDs.
    /// </para>
    /// <para><b>For Beginners:</b> This creates an IR operation from a computation node.
    ///
    /// For each node, this method:
    /// - Checks what operation type it is (Add, MatMul, etc.)
    /// - Gets the input tensor IDs from parent nodes
    /// - Assigns a new tensor ID for the output
    /// - Creates the appropriate IR operation with all parameters
    /// - Sets the output shape and type
    ///
    /// For example, if the node is an "Add" operation with parents [t0, t1]:
    /// - Creates an AddOp
    /// - Sets InputIds = [0, 1]
    /// - Assigns OutputId = 2
    /// - Sets OutputShape from the node's value
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown if the node doesn't have operation type metadata or uses an unsupported operation.
    /// </exception>
    private IROp? ConvertNodeToOp<T>(ComputationNode<T> node)
    {
        // If already processed, return null
        if (_nodeToTensorId.ContainsKey(node))
        {
            return null;
        }

        // Check if node has operation type metadata
        if (string.IsNullOrEmpty(node.OperationType))
        {
            throw new InvalidOperationException(
                $"Node {node.Name ?? "unnamed"} does not have OperationType metadata. " +
                "JIT compilation requires operation type information. " +
                "Ensure TensorOperations methods set OperationType when creating nodes.");
        }

        // Assign output tensor ID
        var outputId = _nextTensorId++;
        _nodeToTensorId[node] = outputId;

        // Get input tensor IDs
        var inputIds = node.Parents.Select(p => _nodeToTensorId[p]).ToArray();

        // Infer IR type from .NET type
        var irType = InferIRType(typeof(T));

        // Get output shape
        var outputShape = node.Value.Shape;

        // Create IR operation based on operation type
        IROp op = node.OperationType switch
        {
            // Basic arithmetic
            "Add" => new AddOp(),
            "Subtract" => new SubtractOp(),
            "ElementwiseMultiply" => new ElementwiseMultiplyOp(),
            "Divide" => new DivideOp(),
            "Power" => new PowerOp { Exponent = GetParam<double>(node, "Exponent", 2.0) },
            "Negate" => new NegateOp(),

            // Math operations
            "Exp" => new ExpOp(),
            "Log" => new LogOp(),
            "Sqrt" => new SqrtOp(),

            // Activations
            "ReLU" => new ReLUOp(),
            "Sigmoid" => new SigmoidOp(),
            "Tanh" => new TanhOp(),
            "Softmax" => new SoftmaxOp { Axis = GetParam<int>(node, "Axis", -1) },
            "ApplyActivation" => new ApplyActivationOp { ActivationName = GetParam<string>(node, "ActivationName", "") },

            // Matrix operations
            "MatMul" => new MatMulOp(),
            "Transpose" => new TransposeOp(),

            // Reduction operations
            "Sum" => new SumOp
            {
                Axes = GetParam<int[]?>(node, "Axes", null),
                KeepDims = GetParam<bool>(node, "KeepDims", false)
            },
            "Mean" => new MeanOp(),
            "ReduceMax" => new ReduceMaxOp
            {
                Axes = GetParam<int[]?>(node, "Axes", null),
                KeepDims = GetParam<bool>(node, "KeepDims", false)
            },
            "ReduceMean" => new ReduceMeanOp
            {
                Axes = GetParam<int[]?>(node, "Axes", null),
                KeepDims = GetParam<bool>(node, "KeepDims", false)
            },
            "ReduceLogVariance" => new ReduceLogVarianceOp
            {
                Axes = GetParam<int[]?>(node, "Axes", null),
                KeepDims = GetParam<bool>(node, "KeepDims", false)
            },

            // Shape operations
            "Reshape" => new ReshapeOp { NewShape = GetParam<int[]>(node, "NewShape", Array.Empty<int>()) },
            "Concat" => new ConcatOp { Axis = GetParam<int>(node, "Axis", 0) },
            "Pad" => new PadOp { PadWidth = GetParam<int[,]?>(node, "PadWidth", null) },
            "Crop" => new CropOp { Cropping = GetParam<int[]>(node, "Cropping", Array.Empty<int>()) },
            "Upsample" => new UpsampleOp { Scale = GetParam<int>(node, "Scale", 2) },
            "PixelShuffle" => new PixelShuffleOp { UpscaleFactor = GetParam<int>(node, "UpscaleFactor", 2) },

            // Convolution operations
            "Conv2D" => new Conv2DOp
            {
                Stride = GetParam<int[]>(node, "Stride", new int[] { 1, 1 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 }),
                HasBias = GetParam<bool>(node, "HasBias", false)
            },
            "ConvTranspose2D" => new ConvTranspose2DOp
            {
                Stride = GetParam<int[]>(node, "Stride", new int[] { 1, 1 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 }),
                OutputPadding = GetParam<int[]>(node, "OutputPadding", new int[] { 0, 0 })
            },
            "DepthwiseConv2D" => new DepthwiseConv2DOp
            {
                Stride = GetParam<int[]>(node, "Stride", new int[] { 1, 1 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 })
            },
            "DilatedConv2D" => new DilatedConv2DOp
            {
                Stride = GetParam<int[]>(node, "Stride", new int[] { 1, 1 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 }),
                Dilation = GetParam<int[]>(node, "Dilation", new int[] { 1, 1 })
            },
            "LocallyConnectedConv2D" => new LocallyConnectedConv2DOp
            {
                Stride = GetParam<int[]>(node, "Stride", new int[] { 1, 1 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 })
            },

            // Pooling operations
            "MaxPool2D" => new MaxPool2DOp
            {
                PoolSize = GetParam<int[]>(node, "PoolSize", new int[] { 2, 2 }),
                Stride = GetParam<int[]>(node, "Stride", new int[] { 2, 2 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 })
            },
            "AvgPool2D" => new AvgPool2DOp
            {
                PoolSize = GetParam<int[]>(node, "PoolSize", new int[] { 2, 2 }),
                Stride = GetParam<int[]>(node, "Stride", new int[] { 2, 2 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 })
            },

            // Normalization operations
            "LayerNorm" => new LayerNormOp
            {
                NormalizedShape = GetParam<int[]>(node, "NormalizedShape", Array.Empty<int>()),
                Epsilon = GetParam<double>(node, "Epsilon", 1e-5)
            },
            "BatchNorm" => new BatchNormOp
            {
                Epsilon = GetParam<double>(node, "Epsilon", 1e-5),
                Momentum = GetParam<double>(node, "Momentum", 0.1)
            },

            // Advanced operations
            "GraphConv" => new GraphConvOp(),
            "AffineGrid" => new AffineGridOp
            {
                OutputSize = GetParam<int[]>(node, "OutputSize", Array.Empty<int>())
            },
            "GridSample" => new GridSampleOp
            {
                InterpolationMode = GetParam<string>(node, "InterpolationMode", "bilinear"),
                PaddingMode = GetParam<string>(node, "PaddingMode", "zeros")
            },
            "RBFKernel" => new RBFKernelOp
            {
                Gamma = GetParam<double>(node, "Gamma", 1.0)
            },

            _ => throw new InvalidOperationException($"Unsupported operation type: {node.OperationType}")
        };

        // Set common properties
        op.OutputId = outputId;
        op.InputIds = inputIds;
        op.OutputType = irType;
        op.OutputShape = outputShape;

        return op;
    }

    /// <summary>
    /// Gets a parameter from a node's operation parameters dictionary.
    /// </summary>
    /// <typeparam name="TParam">The expected type of the parameter.</typeparam>
    /// <param name="node">The computation node (non-generic).</param>
    /// <param name="paramName">The name of the parameter.</param>
    /// <param name="defaultValue">The default value if the parameter is not found.</param>
    /// <returns>The parameter value, or the default if not found.</returns>
    private TParam GetParam<TParam>(object node, string paramName, TParam defaultValue)
    {
        // Use reflection to get OperationParams property
        var nodeType = node.GetType();
        var paramsProperty = nodeType.GetProperty("OperationParams");

        if (paramsProperty != null)
        {
            var paramsDict = paramsProperty.GetValue(node) as Dictionary<string, object>;
            if (paramsDict != null && paramsDict.TryGetValue(paramName, out var value))
            {
                if (value is TParam typedValue)
                {
                    return typedValue;
                }
            }
        }

        return defaultValue;
    }

    /// <summary>
    /// Infers the IR type from a .NET type.
    /// </summary>
    /// <param name="type">The .NET type.</param>
    /// <returns>The corresponding IR type.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This maps C# types to IR types.
    ///
    /// For example:
    /// - float → Float32
    /// - double → Float64
    /// - int → Int32
    ///
    /// This ensures the IR knows what data type to use for each tensor.
    /// </para>
    /// </remarks>
    private IRType InferIRType(Type type)
    {
        if (type == typeof(float)) return IRType.Float32;
        if (type == typeof(double)) return IRType.Float64;
        if (type == typeof(int)) return IRType.Int32;
        if (type == typeof(long)) return IRType.Int64;
        if (type == typeof(byte)) return IRType.Byte;
        if (type == typeof(sbyte)) return IRType.SByte;
        if (type == typeof(short)) return IRType.Int16;
        if (type == typeof(ushort)) return IRType.UInt16;
        if (type == typeof(uint)) return IRType.UInt32;
        if (type == typeof(ulong)) return IRType.UInt64;
        if (type == typeof(decimal)) return IRType.Decimal;
        return IRType.Float32; // Default
    }

    /// <summary>
    /// Performs a topological sort of the computation graph.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the computation.</typeparam>
    /// <param name="outputNode">The output node of the computation graph.</param>
    /// <returns>A list of nodes in topological order.</returns>
    /// <remarks>
    /// <para>
    /// Topological sorting ensures nodes are processed in the correct order,
    /// with each node appearing after all its dependencies (parents).
    /// </para>
    /// <para><b>For Beginners:</b> This determines the order to process nodes.
    ///
    /// We need to process nodes from inputs to outputs:
    /// - Can't compute c = a + b until we have a and b
    /// - Topological sort finds an order where this always works
    ///
    /// Uses depth-first search to visit all nodes and arrange them correctly.
    /// </para>
    /// </remarks>
    private List<ComputationNode<T>> TopologicalSort<T>(ComputationNode<T> outputNode)
    {
        var visited = new HashSet<ComputationNode<T>>();
        var result = new List<ComputationNode<T>>();

        void Visit(ComputationNode<T> node)
        {
            if (visited.Contains(node))
            {
                return;
            }

            visited.Add(node);

            // Visit parents first
            foreach (var parent in node.Parents)
            {
                Visit(parent);
            }

            result.Add(node);
        }

        Visit(outputNode);
        return result;
    }
}
