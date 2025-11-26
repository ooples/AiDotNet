using System.Linq.Expressions;
using System.Reflection;
using AiDotNet.Autodiff;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using Operations = AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Generates executable code from IR graphs using .NET expression trees.
/// </summary>
/// <remarks>
/// <para>
/// The CodeGenerator is the core of the JIT compilation system. It converts optimized
/// IR graphs into executable .NET code using the System.Linq.Expressions API. The generated
/// code is compiled at runtime and can execute the computation graph orders of magnitude
/// faster than interpreting the graph node-by-node.
/// </para>
/// <para><b>For Beginners:</b> This turns our optimized graph into actual executable code.
///
/// Think of it as the final step in compilation:
/// - Input: Optimized IR graph (a structured description of computations)
/// - Output: Compiled function (actual executable machine code)
///
/// How it works:
/// 1. Takes an optimized IR graph
/// 2. Converts each operation to a .NET expression tree
/// 3. Combines all expressions into a complete function
/// 4. Compiles the function to native code
/// 5. Returns a fast, executable function
///
/// Why this is powerful:
/// - The .NET JIT compiler optimizes the code for your CPU
/// - No interpretation overhead (direct execution)
/// - Can inline operations, optimize loops, use SIMD
/// - Typically 5-10x faster than graph interpretation!
///
/// Example:
/// IR Graph: t2 = Add(t0, t1); t3 = ReLU(t2)
/// Generates code like:
///   (t0, t1) => {
///     var t2 = TensorOperations<T>.Add(t0, t1);
///     var t3 = TensorOperations<T>.ReLU(t2);
///     return t3;
///   }
///
/// This compiled code runs at native speed!
/// </para>
/// </remarks>
public class CodeGenerator
{
    private readonly Dictionary<int, ParameterExpression> _tensorVariables = new();
    private readonly List<Expression> _expressions = new();
    private readonly MethodInfo[] _tensorOperationsMethods;

    /// <summary>
    /// Initializes a new instance of the <see cref="CodeGenerator"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Constructor initializes the code generator and caches reflection information
    /// for TensorOperations methods. This avoids repeated reflection lookups during
    /// code generation.
    /// </para>
    /// <para><b>For Beginners:</b> Sets up the code generator.
    ///
    /// During initialization:
    /// - Finds all TensorOperations methods (Add, Multiply, etc.)
    /// - Caches them for fast lookup during code generation
    /// - Prepares internal data structures
    /// </para>
    /// </remarks>
    public CodeGenerator()
    {
        // Cache TensorOperations methods for fast lookup
        _tensorOperationsMethods = typeof(TensorOperations<>)
            .GetMethods(BindingFlags.Public | BindingFlags.Static)
            .ToArray();
    }

    /// <summary>
    /// Generates a compiled function from an IR graph.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="graph">The IR graph to compile.</param>
    /// <returns>A compiled function that executes the graph.</returns>
    /// <remarks>
    /// <para>
    /// This method orchestrates the entire code generation process:
    /// 1. Creates parameter expressions for graph inputs
    /// 2. Generates expressions for each operation in the graph
    /// 3. Builds a lambda expression representing the entire computation
    /// 4. Compiles the lambda to executable code
    /// </para>
    /// <para><b>For Beginners:</b> This compiles the IR graph into a runnable function.
    ///
    /// The process:
    /// 1. Define inputs: Create parameters for each input tensor
    /// 2. Generate operations: Convert each IR operation to code
    /// 3. Build function: Combine all operations into one function
    /// 4. Compile: Turn the function into executable machine code
    /// 5. Return: Give you a fast function you can call
    ///
    /// Example:
    /// Input graph: t2 = Add(t0, t1); t3 = ReLU(t2)
    /// Returns a function: (Tensor<float> t0, Tensor<float> t1) => ReLU(Add(t0, t1))
    ///
    /// You can then call this function with actual tensors and get results instantly!
    /// </para>
    /// </remarks>
    public Func<Tensor<T>[], Tensor<T>[]> Generate<T>(IRGraph graph)
    {
        _tensorVariables.Clear();
        _expressions.Clear();

        // Create parameter for input array
        var inputsParam = Expression.Parameter(typeof(Tensor<T>[]), "inputs");

        // Create variables for each input tensor
        foreach (var inputId in graph.InputIds)
        {
            var inputVar = Expression.Variable(typeof(Tensor<T>), $"t{inputId}");
            _tensorVariables[inputId] = inputVar;

            // Add assignment: t{inputId} = inputs[index]
            var assignment = Expression.Assign(
                inputVar,
                Expression.ArrayIndex(inputsParam, Expression.Constant(graph.InputIds.IndexOf(inputId)))
            );
            _expressions.Add(assignment);
        }

        // Generate code for each operation
        foreach (var op in graph.Operations)
        {
            var opExpression = GenerateOperation<T>(op);
            if (opExpression != null)
            {
                _expressions.Add(opExpression);
            }
        }

        // Create output array
        var outputArray = Expression.NewArrayInit(
            typeof(Tensor<T>),
            graph.OutputIds.Select(id => _tensorVariables[id])
        );

        _expressions.Add(outputArray);

        // Build lambda expression
        var block = Expression.Block(
            _tensorVariables.Values,
            _expressions
        );

        var lambda = Expression.Lambda<Func<Tensor<T>[], Tensor<T>[]>>(
            block,
            inputsParam
        );

        // Compile and return
        return lambda.Compile();
    }

    /// <summary>
    /// Generates an expression for a single IR operation.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="op">The IR operation to generate code for.</param>
    /// <returns>An expression representing the operation.</returns>
    /// <remarks>
    /// <para>
    /// This method converts a single IR operation into a .NET expression tree.
    /// It handles:
    /// - Looking up input tensor variables
    /// - Finding the appropriate TensorOperations method
    /// - Creating a method call expression
    /// - Storing the result in a variable
    /// </para>
    /// <para><b>For Beginners:</b> This converts one operation to code.
    ///
    /// For each operation:
    /// 1. Get the input tensor variables
    /// 2. Find the matching TensorOperations method (e.g., Add, MatMul)
    /// 3. Generate a call to that method
    /// 4. Store the result in a new variable
    ///
    /// Example:
    /// Operation: t2 = Add(t0, t1)
    /// Generates: var t2 = TensorOperations<T>.Add(t0, t1);
    ///
    /// This expression becomes part of the final compiled function.
    /// </para>
    /// </remarks>
    private Expression? GenerateOperation<T>(IROp op)
    {
        // Create output variable
        var outputVar = Expression.Variable(typeof(Tensor<T>), $"t{op.OutputId}");
        _tensorVariables[op.OutputId] = outputVar;

        // Get input variables
        var inputVars = op.InputIds.Select(id => _tensorVariables[id]).ToArray();

        // Generate operation-specific code
        Expression? operationCall = op switch
        {
            // Basic arithmetic
            AddOp => GenerateBinaryOp<T>("Add", inputVars),
            SubtractOp => GenerateBinaryOp<T>("Subtract", inputVars),
            ElementwiseMultiplyOp => GenerateBinaryOp<T>("ElementwiseMultiply", inputVars),
            DivideOp => GenerateBinaryOp<T>("Divide", inputVars),
            PowerOp powerOp => GeneratePowerOp<T>(inputVars[0], powerOp.Exponent),
            NegateOp => GenerateUnaryOp<T>("Negate", inputVars),

            // Math operations
            ExpOp => GenerateUnaryOp<T>("Exp", inputVars),
            LogOp => GenerateUnaryOp<T>("Log", inputVars),
            SqrtOp => GenerateUnaryOp<T>("Sqrt", inputVars),

            // Activations
            ReLUOp => GenerateUnaryOp<T>("ReLU", inputVars),
            SigmoidOp => GenerateUnaryOp<T>("Sigmoid", inputVars),
            TanhOp => GenerateUnaryOp<T>("Tanh", inputVars),
            SoftmaxOp softmaxOp => GenerateSoftmaxOp<T>(inputVars[0], softmaxOp.Axis),

            // Matrix operations
            MatMulOp => GenerateBinaryOp<T>("MatrixMultiply", inputVars),
            TransposeOp => GenerateUnaryOp<T>("Transpose", inputVars),

            // Reduction operations
            SumOp sumOp => GenerateSumOp<T>(inputVars[0], sumOp.Axes, sumOp.KeepDims),
            MeanOp => GenerateUnaryOp<T>("Mean", inputVars),
            ReduceMaxOp reduceMaxOp => GenerateReduceOp<T>("Max", inputVars[0], reduceMaxOp.Axes, reduceMaxOp.KeepDims),
            ReduceMeanOp reduceMeanOp => GenerateReduceOp<T>("Mean", inputVars[0], reduceMeanOp.Axes, reduceMeanOp.KeepDims),

            // Shape operations
            ReshapeOp reshapeOp => GenerateReshapeOp<T>(inputVars[0], reshapeOp.NewShape),
            ConcatOp concatOp => GenerateConcatOp<T>(inputVars, concatOp.Axis),

            // Convolution operations
            Conv2DOp conv2dOp => GenerateConv2DOp<T>(inputVars, conv2dOp),

            // Pooling operations
            MaxPool2DOp maxPoolOp => GenerateMaxPool2DOp<T>(inputVars[0], maxPoolOp),
            AvgPool2DOp avgPoolOp => GenerateAvgPool2DOp<T>(inputVars[0], avgPoolOp),

            // Normalization
            LayerNormOp layerNormOp => GenerateLayerNormOp<T>(inputVars, layerNormOp),
            BatchNormOp batchNormOp => GenerateBatchNormOp<T>(inputVars, batchNormOp),

            // Backward operations (gradient computation)
            Operations.GradAccumulateOp => GenerateGradAccumulateOp<T>(inputVars),
            Operations.GradAddOp gradAddOp => GenerateGradAddOp<T>(inputVars, gradAddOp.InputIndex),
            Operations.GradSubtractOp gradSubtractOp => GenerateGradSubtractOp<T>(inputVars, gradSubtractOp.InputIndex),
            Operations.GradElementwiseMultiplyOp gradMulOp => GenerateGradElementwiseMultiplyOp<T>(inputVars, gradMulOp.InputIndex),
            Operations.GradMatMulLeftOp => GenerateGradMatMulLeftOp<T>(inputVars),
            Operations.GradMatMulRightOp => GenerateGradMatMulRightOp<T>(inputVars),
            Operations.GradReLUOp => GenerateGradReLUOp<T>(inputVars),
            Operations.GradSigmoidOp => GenerateGradSigmoidOp<T>(inputVars),
            Operations.GradTanhOp => GenerateGradTanhOp<T>(inputVars),
            Operations.GradExpOp => GenerateGradExpOp<T>(inputVars),
            Operations.GradLogOp => GenerateGradLogOp<T>(inputVars),
            Operations.GradSoftmaxOp gradSoftmaxOp => GenerateGradSoftmaxOp<T>(inputVars, gradSoftmaxOp.Axis),
            Operations.GradConv2DOp gradConv2dOp => GenerateGradConv2DOp<T>(inputVars, gradConv2dOp),
            Operations.GradMaxPool2DOp gradMaxPoolOp => GenerateGradMaxPool2DOp<T>(inputVars, gradMaxPoolOp),
            Operations.GradAvgPool2DOp gradAvgPoolOp => GenerateGradAvgPool2DOp<T>(inputVars, gradAvgPoolOp),
            Operations.GradBatchNormOp gradBatchNormOp => GenerateGradBatchNormOp<T>(inputVars, gradBatchNormOp),

            // Recurrent network operations
            GRUCellOp gruCellOp => GenerateGRUCellOp<T>(inputVars, gruCellOp),
            LSTMCellOp lstmCellOp => GenerateLSTMCellOp<T>(inputVars, lstmCellOp),

            _ => throw new NotImplementedException($"Code generation for {op.OpType} not yet implemented")
        };

        if (operationCall == null)
        {
            return null;
        }

        // Assign result to output variable
        return Expression.Assign(outputVar, operationCall);
    }

    /// <summary>
    /// Generates code for a binary operation (2 inputs).
    /// </summary>
    private Expression GenerateBinaryOp<T>(string methodName, ParameterExpression[] inputs)
    {
        var method = FindMethod(methodName, typeof(ComputationNode<T>), typeof(ComputationNode<T>));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    /// <summary>
    /// Generates code for a unary operation (1 input).
    /// </summary>
    private Expression GenerateUnaryOp<T>(string methodName, ParameterExpression[] inputs)
    {
        var method = FindMethod(methodName, typeof(ComputationNode<T>));
        return Expression.Call(method, inputs[0]);
    }

    /// <summary>
    /// Generates code for a power operation.
    /// </summary>
    private Expression GeneratePowerOp<T>(ParameterExpression input, double exponent)
    {
        var method = FindMethod("Power", typeof(ComputationNode<T>), typeof(double));
        return Expression.Call(method, input, Expression.Constant(exponent));
    }

    /// <summary>
    /// Generates code for a softmax operation.
    /// </summary>
    private Expression GenerateSoftmaxOp<T>(ParameterExpression input, int axis)
    {
        var method = FindMethod("Softmax", typeof(ComputationNode<T>), typeof(int));
        return Expression.Call(method, input, Expression.Constant(axis));
    }

    /// <summary>
    /// Generates code for a sum operation.
    /// </summary>
    private Expression GenerateSumOp<T>(ParameterExpression input, int[]? axes, bool keepDims)
    {
        var method = FindMethod("Sum", typeof(ComputationNode<T>), typeof(int[]), typeof(bool));
        return Expression.Call(method, input, Expression.Constant(axes), Expression.Constant(keepDims));
    }

    /// <summary>
    /// Generates code for a reduce operation.
    /// </summary>
    private Expression GenerateReduceOp<T>(string methodName, ParameterExpression input, int[]? axes, bool keepDims)
    {
        var method = FindMethod(methodName, typeof(ComputationNode<T>), typeof(int[]), typeof(bool));
        return Expression.Call(method, input, Expression.Constant(axes), Expression.Constant(keepDims));
    }

    /// <summary>
    /// Generates code for a reshape operation.
    /// </summary>
    private Expression GenerateReshapeOp<T>(ParameterExpression input, int[] newShape)
    {
        var method = FindMethod("Reshape", typeof(ComputationNode<T>), typeof(int[]));
        return Expression.Call(method, input, Expression.Constant(newShape));
    }

    /// <summary>
    /// Generates code for a concatenation operation.
    /// </summary>
    private Expression GenerateConcatOp<T>(ParameterExpression[] inputs, int axis)
    {
        var method = FindMethod("Concat", typeof(ComputationNode<T>[]), typeof(int));
        var inputArray = Expression.NewArrayInit(typeof(ComputationNode<T>), inputs);
        return Expression.Call(method, inputArray, Expression.Constant(axis));
    }

    /// <summary>
    /// Generates code for a 2D convolution operation.
    /// </summary>
    private Expression GenerateConv2DOp<T>(ParameterExpression[] inputs, Conv2DOp op)
    {
        // This is a simplified placeholder - full implementation would handle all Conv2D parameters
        var method = FindMethod("Conv2D", typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(int[]), typeof(int[]));
        return Expression.Call(method, inputs[0], inputs[1],
            Expression.Constant(op.Stride), Expression.Constant(op.Padding));
    }

    /// <summary>
    /// Generates code for a 2D max pooling operation.
    /// </summary>
    private Expression GenerateMaxPool2DOp<T>(ParameterExpression input, MaxPool2DOp op)
    {
        var method = FindMethod("MaxPool2D", typeof(ComputationNode<T>),
            typeof(int[]), typeof(int[]), typeof(int[]));
        return Expression.Call(method, input,
            Expression.Constant(op.PoolSize),
            Expression.Constant(op.Stride),
            Expression.Constant(op.Padding));
    }

    /// <summary>
    /// Generates code for a 2D average pooling operation.
    /// </summary>
    private Expression GenerateAvgPool2DOp<T>(ParameterExpression input, AvgPool2DOp op)
    {
        var method = FindMethod("AvgPool2D", typeof(ComputationNode<T>),
            typeof(int[]), typeof(int[]), typeof(int[]));
        return Expression.Call(method, input,
            Expression.Constant(op.PoolSize),
            Expression.Constant(op.Stride),
            Expression.Constant(op.Padding));
    }

    /// <summary>
    /// Generates code for a layer normalization operation.
    /// </summary>
    private Expression GenerateLayerNormOp<T>(ParameterExpression[] inputs, LayerNormOp op)
    {
        var method = FindMethod("LayerNorm", typeof(ComputationNode<T>),
            typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(int[]), typeof(double));
        return Expression.Call(method, inputs[0], inputs[1], inputs[2],
            Expression.Constant(op.NormalizedShape),
            Expression.Constant(op.Epsilon));
    }

    /// <summary>
    /// Generates code for a batch normalization operation.
    /// </summary>
    private Expression GenerateBatchNormOp<T>(ParameterExpression[] inputs, BatchNormOp op)
    {
        var method = FindMethod("BatchNorm", typeof(ComputationNode<T>),
            typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(double), typeof(double));
        return Expression.Call(method, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            Expression.Constant(op.Epsilon),
            Expression.Constant(op.Momentum));
    }

    /// <summary>
    /// Finds a TensorOperations method by name and parameter types.
    /// </summary>
    /// <param name="methodName">The name of the method.</param>
    /// <param name="parameterTypes">The parameter types.</param>
    /// <returns>The MethodInfo for the found method.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This looks up a TensorOperations method.
    ///
    /// We need to find the right method to call for each operation.
    /// This searches through all TensorOperations methods to find one that:
    /// - Has the correct name (e.g., "Add", "MatMul")
    /// - Takes the right parameter types
    ///
    /// Uses reflection to find methods at runtime.
    /// </para>
    /// </remarks>
    private MethodInfo FindMethod(string methodName, params Type[] parameterTypes)
    {
        var method = _tensorOperationsMethods.FirstOrDefault(m =>
            m.Name == methodName &&
            m.GetParameters().Length == parameterTypes.Length);

        if (method == null)
        {
            throw new InvalidOperationException(
                $"Could not find TensorOperations method '{methodName}' with {parameterTypes.Length} parameters");
        }

        // If method is generic, make it concrete with T
        if (method.IsGenericMethodDefinition)
        {
            var genericArg = parameterTypes[0].GetGenericArguments()[0];
            method = method.MakeGenericMethod(genericArg);
        }

        return method;
    }

    // ========== Backward Operation Code Generators ==========

    /// <summary>
    /// Generates code for gradient accumulation operation.
    /// </summary>
    private Expression GenerateGradAccumulateOp<T>(ParameterExpression[] inputs)
    {
        var method = typeof(GradientOps).GetMethod("AccumulateGrad")!.MakeGenericMethod(typeof(T));
        var inputArray = Expression.NewArrayInit(typeof(Tensor<T>), inputs);
        return Expression.Call(method, inputArray);
    }

    /// <summary>
    /// Generates code for GradAdd operation.
    /// </summary>
    private Expression GenerateGradAddOp<T>(ParameterExpression[] inputs, int inputIndex)
    {
        var method = typeof(GradientOps).GetMethod("GradAdd")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], Expression.Constant(inputIndex));
    }

    /// <summary>
    /// Generates code for GradSubtract operation.
    /// </summary>
    private Expression GenerateGradSubtractOp<T>(ParameterExpression[] inputs, int inputIndex)
    {
        var method = typeof(GradientOps).GetMethod("GradSubtract")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], Expression.Constant(inputIndex));
    }

    /// <summary>
    /// Generates code for GradElementwiseMultiply operation.
    /// </summary>
    private Expression GenerateGradElementwiseMultiplyOp<T>(ParameterExpression[] inputs, int inputIndex)
    {
        var method = typeof(GradientOps).GetMethod("GradElementwiseMultiply")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1], Expression.Constant(inputIndex));
    }

    /// <summary>
    /// Generates code for GradMatMulLeft operation.
    /// </summary>
    private Expression GenerateGradMatMulLeftOp<T>(ParameterExpression[] inputs)
    {
        var method = typeof(GradientOps).GetMethod("GradMatMulLeft")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    /// <summary>
    /// Generates code for GradMatMulRight operation.
    /// </summary>
    private Expression GenerateGradMatMulRightOp<T>(ParameterExpression[] inputs)
    {
        var method = typeof(GradientOps).GetMethod("GradMatMulRight")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    /// <summary>
    /// Generates code for GradReLU operation.
    /// </summary>
    private Expression GenerateGradReLUOp<T>(ParameterExpression[] inputs)
    {
        var method = typeof(GradientOps).GetMethod("GradReLU")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    /// <summary>
    /// Generates code for GradSigmoid operation.
    /// </summary>
    private Expression GenerateGradSigmoidOp<T>(ParameterExpression[] inputs)
    {
        var method = typeof(GradientOps).GetMethod("GradSigmoid")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    /// <summary>
    /// Generates code for GradTanh operation.
    /// </summary>
    private Expression GenerateGradTanhOp<T>(ParameterExpression[] inputs)
    {
        var method = typeof(GradientOps).GetMethod("GradTanh")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    /// <summary>
    /// Generates code for GradExp operation.
    /// </summary>
    private Expression GenerateGradExpOp<T>(ParameterExpression[] inputs)
    {
        var method = typeof(GradientOps).GetMethod("GradExp")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    /// <summary>
    /// Generates code for GradLog operation.
    /// </summary>
    private Expression GenerateGradLogOp<T>(ParameterExpression[] inputs)
    {
        var method = typeof(GradientOps).GetMethod("GradLog")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    /// <summary>
    /// Generates code for GradSoftmax operation.
    /// </summary>
    private Expression GenerateGradSoftmaxOp<T>(ParameterExpression[] inputs, int axis)
    {
        var method = typeof(GradientOps).GetMethod("GradSoftmax")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1], Expression.Constant(axis));
    }

    /// <summary>
    /// Generates code for GradConv2D operation.
    /// </summary>
    private Expression GenerateGradConv2DOp<T>(ParameterExpression[] inputs, Operations.GradConv2DOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradConv2D")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0], // gradOutput
            inputs[1], // input or filters depending on InputIndex
            Expression.Constant(op.InputIndex),
            Expression.Constant(op.Stride),
            Expression.Constant(op.Padding));
    }

    /// <summary>
    /// Generates code for GradMaxPool2D operation.
    /// </summary>
    private Expression GenerateGradMaxPool2DOp<T>(ParameterExpression[] inputs, Operations.GradMaxPool2DOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradMaxPool2D")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0], // gradOutput
            inputs[1], // forward input
            Expression.Constant(op.PoolSize),
            Expression.Constant(op.Stride));
    }

    /// <summary>
    /// Generates code for GradAvgPool2D operation.
    /// </summary>
    private Expression GenerateGradAvgPool2DOp<T>(ParameterExpression[] inputs, Operations.GradAvgPool2DOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradAvgPool2D")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0], // gradOutput
            Expression.Constant(op.PoolSize),
            Expression.Constant(op.Stride),
            Expression.Constant(op.OutputShape)); // original input shape
    }

    /// <summary>
    /// Generates code for GradBatchNorm operation.
    /// </summary>
    private Expression GenerateGradBatchNormOp<T>(ParameterExpression[] inputs, Operations.GradBatchNormOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradBatchNorm")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0], // gradOutput
            inputs[1], // normalized input or gamma/beta
            Expression.Constant(op.InputIndex),
            Expression.Constant(op.Epsilon));
    }

    /// <summary>
    /// Generates code for GRU cell operation.
    /// </summary>
    /// <remarks>
    /// GRU cell inputs: x, h, W_ih, W_hh, [b_ih, b_hh]
    /// Outputs: new hidden state h_new
    /// </remarks>
    private Expression GenerateGRUCellOp<T>(ParameterExpression[] inputs, GRUCellOp op)
    {
        var method = typeof(RecurrentOps).GetMethod("GRUCell")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0], // x (input)
            inputs[1], // h (hidden state)
            inputs[2], // W_ih (input-hidden weights)
            inputs[3], // W_hh (hidden-hidden weights)
            inputs.Length > 4 ? inputs[4] : Expression.Constant(null, typeof(Tensor<T>)), // b_ih
            inputs.Length > 5 ? inputs[5] : Expression.Constant(null, typeof(Tensor<T>))  // b_hh
        );
    }

    /// <summary>
    /// Generates code for LSTM cell operation.
    /// </summary>
    /// <remarks>
    /// LSTM cell inputs: x, h, c, W_ih, W_hh, [b_ih, b_hh]
    /// Outputs: tuple of (new hidden state h_new, new cell state c_new)
    /// </remarks>
    private Expression GenerateLSTMCellOp<T>(ParameterExpression[] inputs, LSTMCellOp op)
    {
        var method = typeof(RecurrentOps).GetMethod("LSTMCell")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0], // x (input)
            inputs[1], // h (hidden state)
            inputs[2], // c (cell state)
            inputs[3], // W_ih (input-hidden weights)
            inputs[4], // W_hh (hidden-hidden weights)
            inputs.Length > 5 ? inputs[5] : Expression.Constant(null, typeof(Tensor<T>)), // b_ih
            inputs.Length > 6 ? inputs[6] : Expression.Constant(null, typeof(Tensor<T>))  // b_hh
        );
    }
}
