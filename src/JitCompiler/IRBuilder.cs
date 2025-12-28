using System.Linq;
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
        foreach (var node in topoOrder.Where(n => !inputs.Contains(n)))
        {
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

            // Activations - Basic
            OperationType.ReLU => new ReLUOp(),
            OperationType.Sigmoid => new SigmoidOp(),
            OperationType.Tanh => new TanhOp(),
            OperationType.Softmax => new SoftmaxOp { Axis = GetParam<int>(node, "Axis", -1) },
            OperationType.Activation => new ApplyActivationOp { ActivationName = GetParam<string>(node, "ActivationName", "") },

            // Activations - Extended
            OperationType.ELU => new ELUOp { Alpha = GetParam<double>(node, "Alpha", 1.0) },
            OperationType.LeakyReLU => new LeakyReLUOp { Alpha = GetParam<double>(node, "Alpha", 0.01) },
            OperationType.GELU => new GELUOp { Approximate = GetParam<bool>(node, "Approximate", false) },
            OperationType.Swish => new SwishOp(),
            OperationType.Mish => new MishOp(),
            OperationType.SoftPlus => new SoftPlusOp
            {
                Beta = GetParam<double>(node, "Beta", 1.0),
                Threshold = GetParam<double>(node, "Threshold", 20.0)
            },
            OperationType.SELU => new SELUOp(),
            OperationType.HardSigmoid => new HardSigmoidOp(),
            OperationType.HardTanh => new HardTanhOp
            {
                MinVal = GetParam<double>(node, "MinVal", -1.0),
                MaxVal = GetParam<double>(node, "MaxVal", 1.0)
            },
            OperationType.SoftSign => new SoftSignOp(),
            OperationType.CELU => new CELUOp { Alpha = GetParam<double>(node, "Alpha", 1.0) },
            OperationType.LogSoftmax => new LogSoftmaxOp { Axis = GetParam<int>(node, "Axis", -1) },
            OperationType.PReLU => new PReLUOp(),
            OperationType.ThresholdedReLU => new ThresholdedReLUOp { Threshold = GetParam<double>(node, "Threshold", 1.0) },
            OperationType.LiSHT => new LiSHTOp(),
            OperationType.BentIdentity => new BentIdentityOp(),
            OperationType.Gaussian => new GaussianOp(),
            OperationType.ScaledTanh => new ScaledTanhOp { Beta = GetParam<double>(node, "Beta", 1.0) },
            OperationType.Squash => new SquashOp(),
            OperationType.ISRU => new ISRUOp { Alpha = GetParam<double>(node, "Alpha", 1.0) },
            OperationType.Sign => new SignOp(),
            OperationType.Softmin => new SoftminOp { Axis = GetParam<int>(node, "Axis", -1) },
            OperationType.LogSoftmin => new LogSoftminOp { Axis = GetParam<int>(node, "Axis", -1) },
            OperationType.SQRBF => new SQRBFOp(),
            OperationType.Maxout => new MaxoutOp { NumPieces = GetParam<int>(node, "NumPieces", 2) },
            OperationType.RReLU => new RReLUOp
            {
                Lower = GetParam<double>(node, "Lower", 0.125),
                Upper = GetParam<double>(node, "Upper", 0.333)
            },
            OperationType.SphericalSoftmax => new SphericalSoftmaxOp { Axis = GetParam<int>(node, "Axis", -1) },
            OperationType.TaylorSoftmax => new TaylorSoftmaxOp
            {
                Axis = GetParam<int>(node, "Axis", -1),
                Order = GetParam<int>(node, "Order", 2)
            },
            OperationType.Sparsemax => new SparsemaxOp { Axis = GetParam<int>(node, "Axis", -1) },
            OperationType.HierarchicalSoftmax => new HierarchicalSoftmaxOp(),

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

            // Recurrent network operations
            OperationType.GRUCell => new GRUCellOp
            {
                HiddenSize = GetParam<int>(node, "HiddenSize", 128),
                HasBias = GetParam<bool>(node, "HasBias", true)
            },
            OperationType.LSTMCell => new LSTMCellOp
            {
                HiddenSize = GetParam<int>(node, "HiddenSize", 128),
                HasBias = GetParam<bool>(node, "HasBias", true)
            },

            // Octonion operations
            OperationType.OctonionMatMul => new OctonionMatMulOp
            {
                BatchSize = GetParam<int>(node, "BatchSize", 1),
                InputFeatures = GetParam<int>(node, "InputFeatures", 1),
                OutputFeatures = GetParam<int>(node, "OutputFeatures", 1)
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
        // Delegate to the centralized type mapping to avoid duplication
        // and ensure consistent behavior (throws on unsupported types)
        return IRTypeExtensions.FromSystemType(type);
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
        // Updated inline during traversal so gradients propagate to all ancestors
        var gradientMap = new Dictionary<object, int>();

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
            // Get gradient of this node
            if (!gradientMap.TryGetValue(node, out var nodeGradId))
            {
                // No gradient flows to this node (dead path)
                continue;
            }

            // Treat input nodes as gradient sinks: don't propagate further,
            // their gradients will be exposed as graph outputs at the end
            if (inputs.Contains(node))
            {
                continue;
            }

            // Generate backward operations based on node type
            var backwardOps = CreateBackwardOps(node, nodeGradId);

            if (backwardOps == null || backwardOps.Count == 0)
            {
                // Warn about missing backward implementation for non-leaf nodes
                if (node.Parents.Count > 0 && node.OperationType.HasValue)
                {
                    System.Diagnostics.Debug.WriteLine(
                        $"Warning: No backward ops generated for {node.OperationType.Value}. " +
                        "Gradients will not propagate through this operation.");
                }
                continue;
            }

            foreach (var op in backwardOps)
            {
                graph.Operations.Add(op);
                graph.TensorShapes[op.OutputId] = op.OutputShape;
            }

            // Distribute gradients to parent nodes with inline accumulation
            // This ensures gradientMap is updated during traversal so deeper nodes get gradients
            //
            // INVARIANT: backwardOps[i] corresponds to the gradient for node.Parents[i].
            // CreateBackwardOps() must generate exactly one backward op per parent, in the same order.
            // The defensive `&& i < backwardOps.Count` guard handles cases where CreateBackwardOps
            // returns fewer ops than expected (e.g., unimplemented backward for some input types),
            // silently skipping gradient propagation for those inputs rather than crashing.
            for (int i = 0; i < node.Parents.Count && i < backwardOps.Count; i++)
            {
                var parent = node.Parents[i];
                var parentGradId = backwardOps[i].OutputId;

                if (gradientMap.TryGetValue(parent, out var existingGradId))
                {
                    // Need to accumulate multiple gradient contributions
                    var accumOp = new Operations.GradAccumulateOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { existingGradId, parentGradId },
                        OutputType = InferIRType(typeof(T)),
                        OutputShape = parent.Value.Shape
                    };
                    graph.Operations.Add(accumOp);
                    graph.TensorShapes[accumOp.OutputId] = accumOp.OutputShape;
                    gradientMap[parent] = accumOp.OutputId;
                }
                else
                {
                    // First gradient for this parent
                    gradientMap[parent] = parentGradId;
                }
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

            case OperationType.Conv2D:
                // Gradients for input, filters, and bias
                var convStride = GetParam<int[]>(node, "Stride", new[] { 1, 1 });
                var convPadding = GetParam<int[]>(node, "Padding", new[] { 0, 0 });
                for (int i = 0; i < node.Parents.Count && i < 3; i++)
                {
                    ops.Add(new Operations.GradConv2DOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, forwardInputIds[i == 0 ? 1 : 0] },
                        InputIndex = i,
                        Stride = convStride,
                        Padding = convPadding,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.MaxPool2D:
                // grad_input routes gradient to max elements
                var maxPoolSize = GetParam<int[]>(node, "PoolSize", new[] { 2, 2 });
                var maxPoolStride = GetParam<int[]>(node, "Stride", new[] { 2, 2 });
                ops.Add(new Operations.GradMaxPool2DOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    PoolSize = maxPoolSize,
                    Stride = maxPoolStride,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.AvgPool2D:
                // grad_input distributes gradient equally to all window elements
                var avgPoolSize = GetParam<int[]>(node, "PoolSize", new[] { 2, 2 });
                var avgPoolStride = GetParam<int[]>(node, "Stride", new[] { 2, 2 });
                ops.Add(new Operations.GradAvgPool2DOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId },
                    PoolSize = avgPoolSize,
                    Stride = avgPoolStride,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape
                });
                break;

            case OperationType.BatchNorm:
                // Gradients for input, gamma, beta
                var bnEpsilon = GetParam<double>(node, "Epsilon", 1e-5);
                for (int i = 0; i < node.Parents.Count && i < 3; i++)
                {
                    ops.Add(new Operations.GradBatchNormOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, forwardOutputId },
                        InputIndex = i,
                        Epsilon = bnEpsilon,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.Divide:
                // grad_a = grad_c / b, grad_b = -grad_c * a / (b^2)
                for (int i = 0; i < 2; i++)
                {
                    ops.Add(new Operations.GradDivideOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, forwardInputIds[0], forwardInputIds[1] },
                        InputIndex = i,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.Power:
                // grad_x = grad_y * p * x^(p-1)
                var exponent = GetParam<double>(node, "Exponent", 2.0);
                ops.Add(new Operations.GradPowerOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    Exponent = exponent,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.Negate:
                // grad_x = -grad_y
                ops.Add(new Operations.GradSubtractOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId },
                    InputIndex = 1, // Use subtrahend path which negates
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape
                });
                break;

            case OperationType.Sqrt:
                // grad_x = grad_y / (2 * sqrt(x)) = grad_y / (2 * y)
                ops.Add(new Operations.GradSqrtOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardOutputId },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardOutputId
                });
                break;

            case OperationType.Reshape:
                // grad_x = reshape(grad_y, original_shape)
                ops.Add(new Operations.GradReshapeOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId },
                    OriginalShape = node.Parents[0].Value.Shape,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape
                });
                break;

            case OperationType.Transpose:
                // grad_x = transpose(grad_y, inverse_axes)
                var transposeAxes = GetParam<int[]?>(node, "Axes", null);
                ops.Add(new Operations.GradTransposeOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId },
                    Axes = transposeAxes,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape
                });
                break;

            case OperationType.Concat:
                // grad_xi = slice(grad_y, start_i, end_i, axis)
                var concatAxis = GetParam<int>(node, "Axis", 0);
                int startIndex = 0;
                for (int i = 0; i < node.Parents.Count; i++)
                {
                    var parentShape = node.Parents[i].Value.Shape;
                    var sizeAlongAxis = parentShape[concatAxis < 0 ? parentShape.Length + concatAxis : concatAxis];
                    ops.Add(new Operations.GradConcatOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId },
                        InputIndex = i,
                        Axis = concatAxis,
                        StartIndex = startIndex,
                        Size = sizeAlongAxis,
                        OutputType = irType,
                        OutputShape = parentShape
                    });
                    startIndex += sizeAlongAxis;
                }
                break;

            case OperationType.Pad:
                // grad_x = slice(grad_y, unpad)
                var padding = GetParam<int[]>(node, "Padding", Array.Empty<int>());
                ops.Add(new Operations.GradPadOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId },
                    Padding = padding,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape
                });
                break;

            case OperationType.Crop:
                // grad_x = pad_with_zeros(grad_y, original_shape)
                var cropOffsets = GetParam<int[]>(node, "Offsets", Array.Empty<int>());
                ops.Add(new Operations.GradCropOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId },
                    OriginalShape = node.Parents[0].Value.Shape,
                    CropOffsets = cropOffsets,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape
                });
                break;

            case OperationType.Upsample:
                // grad_x = downsample(grad_y)
                var upsampleScale = GetParam<int>(node, "Scale", 2);
                var upsampleMode = GetParam<string>(node, "Mode", "nearest");
                ops.Add(new Operations.GradUpsampleOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId },
                    Scale = upsampleScale,
                    Mode = upsampleMode,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape
                });
                break;

            case OperationType.LayerNorm:
                // Gradients for input, gamma, beta
                var lnEpsilon = GetParam<double>(node, "Epsilon", 1e-5);
                var normalizedShape = GetParam<int[]>(node, "NormalizedShape", Array.Empty<int>());
                for (int i = 0; i < node.Parents.Count && i < 3; i++)
                {
                    ops.Add(new Operations.GradLayerNormOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, forwardOutputId },
                        InputIndex = i,
                        Epsilon = lnEpsilon,
                        NormalizedShape = normalizedShape,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.ConvTranspose2D:
                // Gradients for input, weight, bias
                var ctStride = GetParam<int[]>(node, "Stride", new[] { 1, 1 });
                var ctPadding = GetParam<int[]>(node, "Padding", new[] { 0, 0 });
                var ctOutPadding = GetParam<int[]>(node, "OutputPadding", new[] { 0, 0 });
                for (int i = 0; i < node.Parents.Count && i < 3; i++)
                {
                    ops.Add(new Operations.GradConvTranspose2DOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, forwardInputIds[i == 0 ? 1 : 0] },
                        InputIndex = i,
                        Stride = ctStride,
                        Padding = ctPadding,
                        OutputPadding = ctOutPadding,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.DepthwiseConv2D:
                // Gradients for input and weight
                var dwStride = GetParam<int[]>(node, "Stride", new[] { 1, 1 });
                var dwPadding = GetParam<int[]>(node, "Padding", new[] { 0, 0 });
                for (int i = 0; i < node.Parents.Count && i < 2; i++)
                {
                    ops.Add(new Operations.GradDepthwiseConv2DOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, forwardInputIds[i == 0 ? 1 : 0] },
                        InputIndex = i,
                        Stride = dwStride,
                        Padding = dwPadding,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.ReduceSum:
            case OperationType.Mean:
            case OperationType.ReduceMean:
                // grad_x = broadcast(grad_y / count_if_mean, original_shape)
                var reduceAxes = GetParam<int[]?>(node, "Axes", null);
                var originalShape = node.Parents[0].Value.Shape;
                var isMean = node.OperationType.Value == OperationType.Mean ||
                             node.OperationType.Value == OperationType.ReduceMean;
                if (isMean)
                {
                    // Calculate count of elements reduced
                    int count = 1;
                    if (reduceAxes == null)
                    {
                        count = originalShape.Aggregate(1, (a, b) => a * b);
                    }
                    else
                    {
                        foreach (var ax in reduceAxes)
                        {
                            var normalizedAxis = ax < 0 ? originalShape.Length + ax : ax;
                            count *= originalShape[normalizedAxis];
                        }
                    }
                    ops.Add(new Operations.GradMeanOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId },
                        OriginalShape = originalShape,
                        Axes = reduceAxes,
                        Count = count,
                        OutputType = irType,
                        OutputShape = originalShape
                    });
                }
                else
                {
                    ops.Add(new Operations.GradSumOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId },
                        OriginalShape = originalShape,
                        Axes = reduceAxes,
                        OutputType = irType,
                        OutputShape = originalShape
                    });
                }
                break;

            case OperationType.LSTMCell:
                // LSTM backward - gradients for input, hidden state, cell state, and weights
                var lstmHiddenSize = GetParam<int>(node, "HiddenSize", 128);
                // LSTM typically has: input, h_prev, c_prev, weights...
                var lstmInputCount = Math.Min(node.Parents.Count, 6);
                for (int i = 0; i < lstmInputCount; i++)
                {
                    ops.Add(new Operations.GradLSTMCellInputOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, forwardOutputId },
                        HiddenSize = lstmHiddenSize,
                        InputIndex = i,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.GRUCell:
                // GRU backward - gradients for input, hidden state, and weights
                var gruHiddenSize = GetParam<int>(node, "HiddenSize", 128);
                var gruInputCount = Math.Min(node.Parents.Count, 5);
                for (int i = 0; i < gruInputCount; i++)
                {
                    ops.Add(new Operations.GradGRUCellOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, forwardOutputId },
                        HiddenSize = gruHiddenSize,
                        InputIndex = i,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.Activation:
                // Generic activation - try to get activation type and handle accordingly
                var activationType = GetParam<string>(node, "ActivationType", "relu");
                switch (activationType.ToLowerInvariant())
                {
                    case "relu":
                        ops.Add(new Operations.GradReLUOp
                        {
                            OutputId = _nextTensorId++,
                            InputIds = new[] { outputGradId, forwardInputIds[0] },
                            OutputType = irType,
                            OutputShape = node.Parents[0].Value.Shape,
                            SavedForwardTensorId = forwardInputIds[0]
                        });
                        break;
                    case "sigmoid":
                        ops.Add(new Operations.GradSigmoidOp
                        {
                            OutputId = _nextTensorId++,
                            InputIds = new[] { outputGradId, forwardOutputId },
                            OutputType = irType,
                            OutputShape = node.Parents[0].Value.Shape,
                            SavedForwardTensorId = forwardOutputId
                        });
                        break;
                    case "tanh":
                        ops.Add(new Operations.GradTanhOp
                        {
                            OutputId = _nextTensorId++,
                            InputIds = new[] { outputGradId, forwardOutputId },
                            OutputType = irType,
                            OutputShape = node.Parents[0].Value.Shape,
                            SavedForwardTensorId = forwardOutputId
                        });
                        break;
                    case "leakyrelu":
                        var alpha = GetParam<double>(node, "Alpha", 0.01);
                        ops.Add(new Operations.GradLeakyReLUOp
                        {
                            OutputId = _nextTensorId++,
                            InputIds = new[] { outputGradId, forwardInputIds[0] },
                            Alpha = alpha,
                            OutputType = irType,
                            OutputShape = node.Parents[0].Value.Shape,
                            SavedForwardTensorId = forwardInputIds[0]
                        });
                        break;
                    case "gelu":
                        var approximate = GetParam<bool>(node, "Approximate", true);
                        ops.Add(new Operations.GradGELUOp
                        {
                            OutputId = _nextTensorId++,
                            InputIds = new[] { outputGradId, forwardInputIds[0] },
                            Approximate = approximate,
                            OutputType = irType,
                            OutputShape = node.Parents[0].Value.Shape,
                            SavedForwardTensorId = forwardInputIds[0]
                        });
                        break;
                    default:
                        // Unknown activation - gradient flow stops
                        break;
                }
                break;

            case OperationType.Dropout:
                // grad_x = grad_y * mask / (1 - p)
                var dropoutProb = GetParam<double>(node, "Probability", 0.5);
                ops.Add(new Operations.GradDropoutOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds.Length > 1 ? forwardInputIds[1] : forwardOutputId },
                    Probability = dropoutProb,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape
                });
                break;

            case OperationType.Embedding:
                // grad_embedding = scatter_add(grad_y, indices, embedding_shape)
                var embeddingShape = node.Parents[0].Value.Shape;
                ops.Add(new Operations.GradEmbeddingOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds.Length > 1 ? forwardInputIds[1] : forwardInputIds[0] },
                    EmbeddingShape = embeddingShape,
                    OutputType = irType,
                    OutputShape = embeddingShape
                });
                break;

            case OperationType.Gather:
                // grad_x = scatter(grad_y, indices, axis, input_shape)
                var gatherAxis = GetParam<int>(node, "Axis", 0);
                var gatherInputShape = node.Parents[0].Value.Shape;
                ops.Add(new Operations.GradGatherOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds.Length > 1 ? forwardInputIds[1] : forwardInputIds[0] },
                    Axis = gatherAxis,
                    InputShape = gatherInputShape,
                    OutputType = irType,
                    OutputShape = gatherInputShape
                });
                break;

            case OperationType.Slice:
                // grad_x = pad_with_zeros(grad_y, original_shape, start_indices)
                var sliceStartIndices = GetParam<int[]>(node, "StartIndices", Array.Empty<int>());
                var sliceOriginalShape = node.Parents[0].Value.Shape;
                ops.Add(new Operations.GradSliceOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId },
                    OriginalShape = sliceOriginalShape,
                    StartIndices = sliceStartIndices,
                    OutputType = irType,
                    OutputShape = sliceOriginalShape
                });
                break;

            case OperationType.Broadcast:
                // grad_x = reduce_sum(grad_y, broadcasted_axes)
                var broadcastOriginalShape = GetParam<int[]>(node, "OriginalShape", node.Parents[0].Value.Shape);
                var broadcastedAxes = GetParam<int[]>(node, "BroadcastedAxes", Array.Empty<int>());
                ops.Add(new Operations.GradBroadcastOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId },
                    OriginalShape = broadcastOriginalShape,
                    BroadcastedAxes = broadcastedAxes,
                    OutputType = irType,
                    OutputShape = broadcastOriginalShape
                });
                break;

            case OperationType.Attention:
                // Gradient for Q, K, V in attention
                var attentionScale = GetParam<double>(node, "Scale", 1.0);
                var causalMask = GetParam<bool>(node, "CausalMask", false);
                for (int i = 0; i < Math.Min(node.Parents.Count, 3); i++)
                {
                    ops.Add(new Operations.GradAttentionOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, forwardOutputId },
                        InputIndex = i,
                        Scale = attentionScale,
                        CausalMask = causalMask,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.MultiHeadAttention:
                // Gradient for multi-head attention
                var mhaNumHeads = GetParam<int>(node, "NumHeads", 8);
                var mhaHeadDim = GetParam<int>(node, "HeadDim", 64);
                for (int i = 0; i < Math.Min(node.Parents.Count, 4); i++)
                {
                    ops.Add(new Operations.GradMultiHeadAttentionOp
                    {
                        OutputId = _nextTensorId++,
                        InputIds = new[] { outputGradId, forwardOutputId },
                        InputIndex = i,
                        NumHeads = mhaNumHeads,
                        HeadDim = mhaHeadDim,
                        OutputType = irType,
                        OutputShape = node.Parents[i].Value.Shape
                    });
                }
                break;

            case OperationType.LeakyReLU:
                // grad_x = grad_y * (1 if x > 0 else alpha)
                var leakyAlpha = GetParam<double>(node, "Alpha", 0.01);
                ops.Add(new Operations.GradLeakyReLUOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    Alpha = leakyAlpha,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.GELU:
                // grad_x = grad_y * gelu_derivative(x)
                var geluApproximate = GetParam<bool>(node, "Approximate", true);
                ops.Add(new Operations.GradGELUOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    Approximate = geluApproximate,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.Split:
                // grad_x = concat([grad_y1, grad_y2, ...], axis)
                var splitAxis = GetParam<int>(node, "Axis", 0);
                ops.Add(new Operations.GradSplitOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId },
                    Axis = splitAxis,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape
                });
                break;

            // Extended activation operations
            case OperationType.ELU:
                // grad_x = grad_y * (x > 0 ? 1 : alpha * exp(x))
                var eluAlpha = GetParam<double>(node, "Alpha", 1.0);
                ops.Add(new Operations.GradELUOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0], forwardOutputId },
                    Alpha = eluAlpha,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.Swish:
                // grad_x = grad_y * (swish(x) + sigmoid(x) * (1 - swish(x)))
                ops.Add(new Operations.GradSwishOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0], forwardOutputId },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.Mish:
                // grad_x = grad_y * mish_derivative(x)
                ops.Add(new Operations.GradMishOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.SoftPlus:
                // grad_x = grad_y * sigmoid(beta * x)
                var softplusBeta = GetParam<double>(node, "Beta", 1.0);
                ops.Add(new Operations.GradSoftPlusOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    Beta = softplusBeta,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.SELU:
                // grad_x = grad_y * scale * (x > 0 ? 1 : alpha * exp(x))
                ops.Add(new Operations.GradSELUOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.HardSigmoid:
                // grad_x = grad_y * (0 if x < -3 or x > 3, else 1/6)
                ops.Add(new Operations.GradHardSigmoidOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.HardTanh:
                // grad_x = grad_y * (0 if x < min or x > max, else 1)
                var hardTanhMin = GetParam<double>(node, "MinVal", -1.0);
                var hardTanhMax = GetParam<double>(node, "MaxVal", 1.0);
                ops.Add(new Operations.GradHardTanhOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    MinVal = hardTanhMin,
                    MaxVal = hardTanhMax,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.SoftSign:
                // grad_x = grad_y / (1 + |x|)^2
                ops.Add(new Operations.GradSoftSignOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            case OperationType.CELU:
                // grad_x = grad_y * (x > 0 ? 1 : exp(x/alpha))
                var celuAlpha = GetParam<double>(node, "Alpha", 1.0);
                ops.Add(new Operations.GradCELUOp
                {
                    OutputId = _nextTensorId++,
                    InputIds = new[] { outputGradId, forwardInputIds[0] },
                    Alpha = celuAlpha,
                    OutputType = irType,
                    OutputShape = node.Parents[0].Value.Shape,
                    SavedForwardTensorId = forwardInputIds[0]
                });
                break;

            // For unsupported operations, return empty list (gradient won't flow)
            default:
                // Unsupported operation - gradient flow stops here
                // This is safe as it will just not update those parameters
                break;
        }

        return ops;
    }
}
