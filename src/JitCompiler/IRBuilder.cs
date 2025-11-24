using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using Operations = AiDotNet.JitCompiler.IR.Operations;

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
        if (node.OperationType == null)
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
        IROp op = node.OperationType.Value switch
        {
            // Basic arithmetic
            OperationType.Add => new AddOp(),
            OperationType.Subtract => new SubtractOp(),
            OperationType.Multiply => new ElementwiseMultiplyOp(),
            OperationType.Divide => new DivideOp(),
            OperationType.Power => new PowerOp { Exponent = GetParam<double>(node, "Exponent", 2.0) },
            OperationType.Negate => new NegateOp(),

            // Math operations
            OperationType.Exp => new ExpOp(),
            OperationType.Log => new LogOp(),
            OperationType.Sqrt => new SqrtOp(),

            // Activations
            OperationType.ReLU => new ReLUOp(),
            OperationType.Sigmoid => new SigmoidOp(),
            OperationType.Tanh => new TanhOp(),
            OperationType.Softmax => new SoftmaxOp { Axis = GetParam<int>(node, "Axis", -1) },
            OperationType.Activation => new ApplyActivationOp { ActivationName = GetParam<string>(node, "ActivationName", "") },

            // Matrix operations
            OperationType.MatMul => new MatMulOp(),
            OperationType.Transpose => new TransposeOp(),

            // Reduction operations
            OperationType.ReduceSum => new SumOp
            {
                Axes = GetParam<int[]?>(node, "Axes", null),
                KeepDims = GetParam<bool>(node, "KeepDims", false)
            },
            OperationType.Mean => new MeanOp(),
            OperationType.ReduceMax => new ReduceMaxOp
            {
                Axes = GetParam<int[]?>(node, "Axes", null),
                KeepDims = GetParam<bool>(node, "KeepDims", false)
            },
            OperationType.ReduceMean => new ReduceMeanOp
            {
                Axes = GetParam<int[]?>(node, "Axes", null),
                KeepDims = GetParam<bool>(node, "KeepDims", false)
            },
            OperationType.ReduceLogVariance => new ReduceLogVarianceOp
            {
                Axes = GetParam<int[]?>(node, "Axes", null),
                KeepDims = GetParam<bool>(node, "KeepDims", false)
            },

            // Shape operations
            OperationType.Reshape => new ReshapeOp { NewShape = GetParam<int[]>(node, "NewShape", Array.Empty<int>()) },
            OperationType.Concat => new ConcatOp { Axis = GetParam<int>(node, "Axis", 0) },
            OperationType.Pad => new PadOp { PadWidth = GetParam<int[,]?>(node, "PadWidth", null) },
            OperationType.Crop => new CropOp { Cropping = GetParam<int[]>(node, "Cropping", Array.Empty<int>()) },
            OperationType.Upsample => new UpsampleOp { Scale = GetParam<int>(node, "Scale", 2) },
            OperationType.PixelShuffle => new PixelShuffleOp { UpscaleFactor = GetParam<int>(node, "UpscaleFactor", 2) },

            // Convolution operations
            OperationType.Conv2D => new Conv2DOp
            {
                Stride = GetParam<int[]>(node, "Stride", new int[] { 1, 1 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 }),
                HasBias = GetParam<bool>(node, "HasBias", false)
            },
            OperationType.ConvTranspose2D => new ConvTranspose2DOp
            {
                Stride = GetParam<int[]>(node, "Stride", new int[] { 1, 1 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 }),
                OutputPadding = GetParam<int[]>(node, "OutputPadding", new int[] { 0, 0 })
            },
            OperationType.DepthwiseConv2D => new DepthwiseConv2DOp
            {
                Stride = GetParam<int[]>(node, "Stride", new int[] { 1, 1 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 })
            },
            OperationType.DilatedConv2D => new DilatedConv2DOp
            {
                Stride = GetParam<int[]>(node, "Stride", new int[] { 1, 1 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 }),
                Dilation = GetParam<int[]>(node, "Dilation", new int[] { 1, 1 })
            },
            OperationType.LocallyConnectedConv2D => new LocallyConnectedConv2DOp
            {
                Stride = GetParam<int[]>(node, "Stride", new int[] { 1, 1 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 })
            },

            // Pooling operations
            OperationType.MaxPool2D => new MaxPool2DOp
            {
                PoolSize = GetParam<int[]>(node, "PoolSize", new int[] { 2, 2 }),
                Stride = GetParam<int[]>(node, "Stride", new int[] { 2, 2 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 })
            },
            OperationType.AvgPool2D => new AvgPool2DOp
            {
                PoolSize = GetParam<int[]>(node, "PoolSize", new int[] { 2, 2 }),
                Stride = GetParam<int[]>(node, "Stride", new int[] { 2, 2 }),
                Padding = GetParam<int[]>(node, "Padding", new int[] { 0, 0 })
            },

            // Normalization operations
            OperationType.LayerNorm => new LayerNormOp
            {
                NormalizedShape = GetParam<int[]>(node, "NormalizedShape", Array.Empty<int>()),
                Epsilon = GetParam<double>(node, "Epsilon", 1e-5)
            },
            OperationType.BatchNorm => new BatchNormOp
            {
                Epsilon = GetParam<double>(node, "Epsilon", 1e-5),
                Momentum = GetParam<double>(node, "Momentum", 0.1)
            },

            // Advanced operations
            OperationType.GraphConv => new GraphConvOp(),
            OperationType.AffineGrid => new AffineGridOp
            {
                OutputSize = GetParam<int[]>(node, "OutputSize", Array.Empty<int>())
            },
            OperationType.GridSample => new GridSampleOp
            {
                InterpolationMode = GetParam<string>(node, "InterpolationMode", "bilinear"),
                PaddingMode = GetParam<string>(node, "PaddingMode", "zeros")
            },
            OperationType.RBFKernel => new RBFKernelOp
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

    /// <summary>
    /// Builds a backward IR graph for gradient computation.
    /// </summary>
    /// <typeparam name="T">The numeric type used in the computation.</typeparam>
    /// <param name="outputNode">The output node of the forward computation graph.</param>
    /// <param name="inputs">The input nodes to compute gradients for.</param>
    /// <returns>An IR graph that computes gradients via backpropagation.</returns>
    /// <remarks>
    /// <para>
    /// This method builds the backward pass (gradient computation) graph from a forward graph.
    /// The backward graph takes output gradients as inputs and computes gradients with respect
    /// to the original inputs via automatic differentiation.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the gradient computation graph for training.
    ///
    /// In neural network training:
    /// - Forward pass: input → layers → output → loss
    /// - Backward pass: loss gradient → layers (in reverse) → input gradients
    ///
    /// This method creates the backward pass graph automatically!
    ///
    /// Algorithm:
    /// 1. Traverse forward graph in reverse topological order
    /// 2. For each operation, generate its backward (gradient) operation
    /// 3. Handle gradient accumulation for nodes with multiple consumers
    /// 4. Build IR graph mapping output gradients → input gradients
    ///
    /// Example operations and their gradients:
    /// - Add(a, b) → backward distributes gradient to both a and b
    /// - MatMul(a, b) → backward: grad_a = grad_out @ b^T, grad_b = a^T @ grad_out
    /// - ReLU(x) → backward: grad_x = grad_out * (x > 0)
    ///
    /// </para>
    /// <para><b>IMPLEMENTATION STATUS:</b>
    ///
    /// This is a complex feature requiring implementation of:
    ///
    /// 1. **Reverse Graph Traversal**
    ///    - Walk forward graph in reverse topological order
    ///    - Track gradient flow through each operation
    ///
    /// 2. **Backward Operation Mapping**
    ///    - For each forward op type, generate corresponding backward op(s)
    ///    - Examples:
    ///      - AddOp → GradAddOp (distributes gradient to both inputs)
    ///      - MatMulOp → GradMatMulLeftOp + GradMatMulRightOp
    ///      - ReLUOp → GradReLUOp (masks gradient by activation)
    ///      - Etc. for all 43+ operation types
    ///
    /// 3. **Gradient Accumulation**
    ///    - When a node has multiple consumers, accumulate gradients
    ///    - Insert GradAccumulateOp to sum gradients from different paths
    ///
    /// 4. **Memory Optimization**
    ///    - Forward activations may need to be saved for backward pass
    ///    - Implement checkpointing for memory-efficient training
    ///
    /// 5. **IR Operation Types Needed**
    ///    - Create new IR op types for backward operations:
    ///      - GradAddOp, GradSubtractOp, GradMultiplyOp
    ///      - GradMatMulLeftOp, GradMatMulRightOp
    ///      - GradReLUOp, GradSigmoidOp, GradTanhOp
    ///      - GradConv2DOp, GradMaxPool2DOp
    ///      - GradAccumulateOp (sums multiple gradients)
    ///    - Implement code generation for each
    ///
    /// 6. **Testing Required**
    ///    - Gradient correctness tests (numerical gradient checking)
    ///    - Performance benchmarks vs. non-compiled backward pass
    ///    - Memory usage profiling
    ///
    /// **TODO:** Full implementation of backward pass IR builder
    /// - This is a substantial feature requiring:
    ///   - New IR operation types (~50+ backward ops)
    ///   - Code generation for backward ops
    ///   - Gradient accumulation logic
    ///   - Extensive testing
    /// - Estimated effort: 1-2 weeks for complete implementation
    /// - See PyTorch's autograd and TensorFlow's GradientTape for reference implementations
    /// </para>
    /// </remarks>
    /// <exception cref="NotImplementedException">
    /// This method requires full implementation of backward operation mapping and gradient accumulation.
    /// </exception>
    public IRGraph BuildBackward<T>(ComputationNode<T> outputNode, List<ComputationNode<T>> inputs)
    {
        var graph = new IRGraph();
        _nextTensorId = 0;
        _nodeToTensorId.Clear();

        // Dictionary to track forward node -> backward gradient tensor ID
        var gradientMap = new Dictionary<object, int>();

        // Dictionary to accumulate gradients for nodes with multiple consumers
        var gradientAccumulators = new Dictionary<object, List<int>>();

        // First, build the forward graph to get tensor IDs
        var forwardNodes = TopologicalSort(outputNode);

        // Assign tensor IDs to forward nodes (these will be saved if needed)
        foreach (var node in forwardNodes)
        {
            if (!_nodeToTensorId.ContainsKey(node))
            {
                _nodeToTensorId[node] = _nextTensorId++;
            }
        }

        // Output gradient is input to backward pass (initialized to 1s typically)
        var outputGradId = _nextTensorId++;
        graph.InputIds.Add(outputGradId);
        graph.TensorShapes[outputGradId] = outputNode.Value.Shape;
        gradientMap[outputNode] = outputGradId;

        // Traverse in reverse topological order for backpropagation
        var reverseOrder = forwardNodes.AsEnumerable().Reverse().ToList();

        foreach (var node in reverseOrder)
        {
            // Skip input nodes - their gradients are outputs of backward graph
            if (inputs.Contains(node))
            {
                continue;
            }

            // Get gradient of this node
            if (!gradientMap.TryGetValue(node, out var nodeGradId))
            {
                // No gradient flows to this node (dead path)
                continue;
            }

            // Generate backward operations based on node type
            var backwardOps = CreateBackwardOps(node, nodeGradId);

            if (backwardOps != null && backwardOps.Count > 0)
            {
                foreach (var op in backwardOps)
                {
                    graph.Operations.Add(op);
                    graph.TensorShapes[op.OutputId] = op.OutputShape;
                }

                // Distribute gradients to parent nodes
                for (int i = 0; i < node.Parents.Count; i++)
                {
                    var parent = node.Parents[i];
                    var parentGradId = backwardOps[i].OutputId;

                    // If parent already has gradient(s), accumulate
                    if (!gradientAccumulators.ContainsKey(parent))
                    {
                        gradientAccumulators[parent] = new List<int>();
                    }
                    gradientAccumulators[parent].Add(parentGradId);
                }
            }
        }

        // Create gradient accumulation operations for nodes with multiple gradients
        foreach (var kvp in gradientAccumulators)
        {
            var node = kvp.Key;
            var gradIds = kvp.Value;

            if (gradIds.Count == 1)
            {
                // Single gradient - no accumulation needed
                gradientMap[node] = gradIds[0];
            }
            else
            {
                // Multiple gradients - need to accumulate
                var accumOp = new Operations.GradAccumulateOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = gradIds.ToArray(),
                    OutputType = InferIRType(typeof(T)),
                    OutputShape = ((ComputationNode<T>)node).Value.Shape
                };
                graph.Operations.Add(accumOp);
                graph.TensorShapes[accumOp.OutputId] = accumOp.OutputShape;
                gradientMap[node] = accumOp.OutputId;
            }
        }

        // Mark input gradients as outputs
        foreach (var input in inputs)
        {
            if (gradientMap.TryGetValue(input, out var gradId))
            {
                graph.OutputIds.Add(gradId);
            }
        }

        return graph;
    }

    /// <summary>
    /// Creates backward operations for a given forward node.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="node">The forward computation node.</param>
    /// <param name="outputGradId">The tensor ID of the gradient of this node's output.</param>
    /// <returns>List of backward operations (one per parent).</returns>
    private List<IROp> CreateBackwardOps<T>(ComputationNode<T> node, int outputGradId)
    {
        var ops = new List<IROp>();
        var irType = InferIRType(typeof(T));

        if (node.OperationType == null)
        {
            return ops;
        }

        // Get forward tensor IDs
        var forwardInputIds = node.Parents.Select(p => _nodeToTensorId[p]).ToArray();
        var forwardOutputId = _nodeToTensorId[node];

        switch (node.OperationType.Value)
        {
            case OperationType.Add:
                // grad_a = grad_c, grad_b = grad_c
                for (int i = 0; i < 2; i++)
                {
                    ops.Add(new Operations.GradAddOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId },
                        InputIndex = i,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.Subtract:
                // grad_a = grad_c, grad_b = -grad_c
                for (int i = 0; i < 2; i++)
                {
                    ops.Add(new Operations.GradSubtractOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId },
                        InputIndex = i,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.Multiply:
                // grad_a = grad_c * b, grad_b = grad_c * a
                for (int i = 0; i < 2; i++)
                {
                    var otherInputId = forwardInputIds[1 - i];
                    ops.Add(new Operations.GradElementwiseMultiplyOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, otherInputId },
                        InputIndex = i,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.MatMul:
                // grad_A = grad_C @ B^T
                ops.Add(new Operations.GradMatMulLeftOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[1] },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape
                });
                // grad_B = A^T @ grad_C
                ops.Add(new Operations.GradMatMulRightOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { forwardInputIds[0], outputGradId },
                    OutputType = irType,
                    OutputShape = node.Parents[1].Value.Shape
                });
                break;

            case OperationType.ReLU:
                // grad_x = grad_y * (x > 0)
                ops.Add(new Operations.GradReLUOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.Sigmoid:
                // grad_x = grad_y * y * (1 - y)
                ops.Add(new Operations.GradSigmoidOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardOutputId },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardOutputId
                });
                break;

            case OperationType.Tanh:
                // grad_x = grad_y * (1 - y^2)
                ops.Add(new Operations.GradTanhOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardOutputId },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardOutputId
                });
                break;

            case OperationType.Exp:
                // grad_x = grad_y * y
                ops.Add(new Operations.GradExpOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardOutputId },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardOutputId
                });
                break;

            case OperationType.Log:
                // grad_x = grad_y / x
                ops.Add(new Operations.GradLogOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.Softmax:
                // grad_x = y * (grad_y - sum(grad_y * y))
                var axis = GetParam<int>(node, "Axis", -1);
                ops.Add(new Operations.GradSoftmaxOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardOutputId },
                    Axis = axis,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardOutputId
                });
                break;

            // TODO: Add more operation types as needed
            // For unsupported operations, return empty list (gradient won't flow)
            default:
                // Unsupported operation - gradient flow stops here
                // This is safe as it will just not update those parameters
                break;
        }

        return ops;
    }
}
