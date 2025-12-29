using System.Linq.Expressions;
using System.Reflection;
using AiDotNet.Autodiff;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using AiDotNet.JitCompiler.Runtime;
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
        // Use local variables instead of instance fields to ensure thread safety
        var tensorVariables = new Dictionary<int, ParameterExpression>();
        var expressions = new List<Expression>();

        // Create parameter for input array
        var inputsParam = Expression.Parameter(typeof(Tensor<T>[]), "inputs");

        // Create variables for each input tensor (as ComputationNode<T> for TensorOperations compatibility)
        foreach (var inputId in graph.InputIds)
        {
            var inputVar = Expression.Variable(typeof(ComputationNode<T>), $"t{inputId}");
            tensorVariables[inputId] = inputVar;

            // Wrap tensor in ComputationNode: t{inputId} = TensorOperations<T>.Variable(inputs[index], name, requiresGradient)
            var variableMethod = typeof(TensorOperations<T>).GetMethod("Variable", new[] { typeof(Tensor<T>), typeof(string), typeof(bool) });
            var wrapCall = Expression.Call(variableMethod!,
                Expression.ArrayIndex(inputsParam, Expression.Constant(graph.InputIds.IndexOf(inputId))),
                Expression.Constant($"input_{inputId}"),
                Expression.Constant(true)); // requiresGradient = true
            var assignment = Expression.Assign(inputVar, wrapCall);
            expressions.Add(assignment);
        }

        // Generate code for each operation
        foreach (var op in graph.Operations)
        {
            var opExpression = GenerateOperation<T>(op, tensorVariables, expressions);
            if (opExpression != null)
            {
                expressions.Add(opExpression);
            }
        }

        // Create output array - extract Tensor<T> from ComputationNode<T>.Value
        var valueProperty = typeof(ComputationNode<T>).GetProperty("Value");
        var outputArray = Expression.NewArrayInit(
            typeof(Tensor<T>),
            graph.OutputIds.Select(id => Expression.Property(tensorVariables[id], valueProperty!))
        );

        expressions.Add(outputArray);

        // Build lambda expression
        var block = Expression.Block(
            tensorVariables.Values,
            expressions
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
    private Expression? GenerateOperation<T>(IROp op, Dictionary<int, ParameterExpression> tensorVariables, List<Expression> expressions)
    {
        // Create output variable (as ComputationNode<T> to match TensorOperations return types)
        var outputVar = Expression.Variable(typeof(ComputationNode<T>), $"t{op.OutputId}");
        tensorVariables[op.OutputId] = outputVar;

        // Get input variables
        var inputVars = op.InputIds.Select(id => tensorVariables[id]).ToArray();

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
            AbsOp => GenerateUnaryOp<T>("Abs", inputVars),

            // Math operations
            ExpOp => GenerateUnaryOp<T>("Exp", inputVars),
            LogOp => GenerateUnaryOp<T>("Log", inputVars),
            SqrtOp => GenerateUnaryOp<T>("Sqrt", inputVars),

            // Activations - Basic
            ReLUOp => GenerateUnaryOp<T>("ReLU", inputVars),
            SigmoidOp => GenerateUnaryOp<T>("Sigmoid", inputVars),
            TanhOp => GenerateUnaryOp<T>("Tanh", inputVars),
            SoftmaxOp softmaxOp => GenerateSoftmaxOp<T>(inputVars[0], softmaxOp.Axis),

            // Activations - Extended
            ELUOp eluOp => GenerateELUOp<T>(inputVars[0], eluOp.Alpha),
            LeakyReLUOp leakyReluOp => GenerateLeakyReLUOp<T>(inputVars[0], leakyReluOp.Alpha),
            GELUOp geluOp => GenerateGELUOp<T>(inputVars[0], geluOp.Approximate),
            SwishOp => GenerateUnaryOp<T>("Swish", inputVars),
            MishOp => GenerateUnaryOp<T>("Mish", inputVars),
            SoftPlusOp softPlusOp => GenerateSoftPlusOp<T>(inputVars[0], softPlusOp.Beta, softPlusOp.Threshold),
            SELUOp => GenerateUnaryOp<T>("SELU", inputVars),
            HardSigmoidOp => GenerateUnaryOp<T>("HardSigmoid", inputVars),
            HardTanhOp hardTanhOp => GenerateHardTanhOp<T>(inputVars[0], hardTanhOp.MinVal, hardTanhOp.MaxVal),
            SoftSignOp => GenerateUnaryOp<T>("SoftSign", inputVars),
            CELUOp celuOp => GenerateCELUOp<T>(inputVars[0], celuOp.Alpha),
            LogSoftmaxOp logSoftmaxOp => GenerateLogSoftmaxOp<T>(inputVars[0], logSoftmaxOp.Axis),
            PReLUOp => GenerateBinaryOp<T>("PReLU", inputVars),
            ThresholdedReLUOp threshRelu => GenerateThresholdedReLUOp<T>(inputVars[0], threshRelu.Threshold),

            // Activations - Additional Extended Set
            LiSHTOp => GenerateUnaryOp<T>("LiSHT", inputVars),
            BentIdentityOp => GenerateUnaryOp<T>("BentIdentity", inputVars),
            GaussianOp => GenerateUnaryOp<T>("Gaussian", inputVars),
            ScaledTanhOp scaledTanh => GenerateScaledTanhOp<T>(inputVars[0], scaledTanh.Beta),
            SquashOp => GenerateUnaryOp<T>("Squash", inputVars),
            ISRUOp isru => GenerateISRUOp<T>(inputVars[0], isru.Alpha),
            SignOp => GenerateUnaryOp<T>("Sign", inputVars),
            SoftminOp softmin => GenerateSoftminOp<T>(inputVars[0], softmin.Axis),
            LogSoftminOp logSoftmin => GenerateLogSoftminOp<T>(inputVars[0], logSoftmin.Axis),
            SQRBFOp => GenerateUnaryOp<T>("SQRBF", inputVars),
            MaxoutOp maxout => GenerateMaxoutOp<T>(inputVars[0], maxout.NumPieces),
            RReLUOp rrelu => GenerateRReLUOp<T>(inputVars[0], rrelu.Lower, rrelu.Upper),
            SphericalSoftmaxOp spherical => GenerateSphericalSoftmaxOp<T>(inputVars[0], spherical.Axis),
            TaylorSoftmaxOp taylor => GenerateTaylorSoftmaxOp<T>(inputVars[0], taylor.Axis, taylor.Order),
            SparsemaxOp sparsemax => GenerateSparsemaxOp<T>(inputVars[0], sparsemax.Axis),
            HierarchicalSoftmaxOp hierarchical => GenerateHierarchicalSoftmaxOp<T>(inputVars[0], hierarchical.TreeStructure),

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
            SplitOp splitOp => GenerateSplitOp<T>(inputVars[0], splitOp),
            SliceOp sliceOp => GenerateSliceOp<T>(inputVars[0], sliceOp),
            SquareOp => GenerateUnaryOp<T>("Square", inputVars),
            NormOp normOp => GenerateNormOp<T>(inputVars[0], normOp.Axis, normOp.KeepDims),

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
            Operations.GradLayerNormOp gradLayerNormOp => GenerateGradLayerNormOp<T>(inputVars, gradLayerNormOp),

            // Additional backward operations
            Operations.GradReshapeOp gradReshapeOp => GenerateGradReshapeOp<T>(inputVars, gradReshapeOp),
            Operations.GradTransposeOp gradTransposeOp => GenerateGradTransposeOp<T>(inputVars, gradTransposeOp),
            Operations.GradConcatOp gradConcatOp => GenerateGradConcatOp<T>(inputVars, gradConcatOp),
            Operations.GradSplitOp gradSplitOp => GenerateGradSplitOp<T>(inputVars, gradSplitOp),
            Operations.GradDivideOp gradDivideOp => GenerateGradDivideOp<T>(inputVars, gradDivideOp),
            Operations.GradPowerOp gradPowerOp => GenerateGradPowerOp<T>(inputVars, gradPowerOp),
            Operations.GradSqrtOp => GenerateGradSqrtOp<T>(inputVars),
            Operations.GradSumOp gradSumOp => GenerateGradSumOp<T>(inputVars, gradSumOp),
            Operations.GradMeanOp gradMeanOp => GenerateGradMeanOp<T>(inputVars, gradMeanOp),
            Operations.GradSliceOp gradSliceOp => GenerateGradSliceOp<T>(inputVars, gradSliceOp),
            Operations.GradPadOp gradPadOp => GenerateGradPadOp<T>(inputVars, gradPadOp),
            Operations.GradDropoutOp gradDropoutOp => GenerateGradDropoutOp<T>(inputVars, gradDropoutOp),
            Operations.GradEmbeddingOp gradEmbeddingOp => GenerateGradEmbeddingOp<T>(inputVars, gradEmbeddingOp),
            Operations.GradGatherOp gradGatherOp => GenerateGradGatherOp<T>(inputVars, gradGatherOp),
            Operations.GradLeakyReLUOp gradLeakyReLUOp => GenerateGradLeakyReLUOp<T>(inputVars, gradLeakyReLUOp),
            Operations.GradGELUOp gradGELUOp => GenerateGradGELUOp<T>(inputVars, gradGELUOp),
            Operations.GradBroadcastOp gradBroadcastOp => GenerateGradBroadcastOp<T>(inputVars, gradBroadcastOp),

            // Recurrent network operations
            GRUCellOp gruCellOp => GenerateGRUCellOp<T>(inputVars, gruCellOp),
            LSTMCellOp lstmCellOp => GenerateLSTMCellOp<T>(inputVars, lstmCellOp),

            // Embedding and attention operations
            EmbeddingOp embeddingOp => GenerateEmbeddingOp<T>(inputVars, embeddingOp),
            ScaledDotProductAttentionOp sdpaOp => GenerateScaledDotProductAttentionOp<T>(inputVars, sdpaOp),
            MultiHeadAttentionOp mhaOp => GenerateMultiHeadAttentionOp<T>(inputVars, mhaOp),

            // Fused operations
            FusedMatMulAddOp => GenerateFusedMatMulAddOp<T>(inputVars),
            FusedLinearReLUOp => GenerateFusedLinearReLUOp<T>(inputVars),
            FusedConvBatchNormOp fusedConvBnOp => GenerateFusedConvBatchNormOp<T>(inputVars, fusedConvBnOp),
            FusedAddReLUOp => GenerateFusedAddReLUOp<T>(inputVars),

            // Complex number operations
            ComplexMatMulOp => GenerateComplexMatMulOp<T>(inputVars),
            ComplexMultiplyOp => GenerateComplexMultiplyOp<T>(inputVars),

            // Octonion operations
            OctonionMatMulOp octonionMatMulOp => GenerateOctonionMatMulOp<T>(inputVars, octonionMatMulOp),

            // Dropout
            DropoutOp dropoutOp => GenerateDropoutOp<T>(inputVars[0], dropoutOp),

            // Unrolled operations (from LoopUnrollingPass)
            Operations.UnrolledSequenceOp unrolledSeq => GenerateUnrolledSequenceOp<T>(inputVars, unrolledSeq),
            Operations.UnrolledElementwiseOp unrolledElem => GenerateUnrolledElementwiseOp<T>(inputVars, unrolledElem),
            Operations.UnrolledReductionOp unrolledRed => GenerateUnrolledReductionOp<T>(inputVars, unrolledRed),

            // Vectorized operations (from VectorizationPass)
            Operations.VectorizedBinaryOp vecBinary => GenerateVectorizedBinaryOp<T>(inputVars, vecBinary),
            Operations.VectorizedUnaryOp vecUnary => GenerateVectorizedUnaryOp<T>(inputVars, vecUnary),
            Operations.VectorizedReductionOp vecReduce => GenerateVectorizedReductionOp<T>(inputVars, vecReduce),
            Operations.VectorizedMatMulOp vecMatMul => GenerateVectorizedMatMulOp<T>(inputVars, vecMatMul),

            // Differentiable approximation operations
            Operations.SoftSplitOp softSplit => GenerateSoftSplitOp<T>(inputVars, softSplit),
            Operations.SoftKNNOp softKnn => GenerateSoftKNNOp<T>(inputVars, softKnn),
            Operations.SoftLocallyWeightedOp softLw => GenerateSoftLocallyWeightedOp<T>(inputVars, softLw),
            Operations.FakeQuantizationOp fakeQuant => GenerateFakeQuantizationOp<T>(inputVars, fakeQuant),

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
        var method = FindMethod<T>(methodName, typeof(ComputationNode<T>), typeof(ComputationNode<T>));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    /// <summary>
    /// Generates code for a unary operation (1 input).
    /// </summary>
    private Expression GenerateUnaryOp<T>(string methodName, ParameterExpression[] inputs)
    {
        var method = FindMethod<T>(methodName, typeof(ComputationNode<T>));
        return Expression.Call(method, inputs[0]);
    }

    /// <summary>
    /// Generates code for a power operation.
    /// </summary>
    private Expression GeneratePowerOp<T>(ParameterExpression input, double exponent)
    {
        var method = FindMethod<T>("Power", typeof(ComputationNode<T>), typeof(double));
        return Expression.Call(method, input, Expression.Constant(exponent));
    }

    /// <summary>
    /// Generates code for a softmax operation.
    /// </summary>
    private Expression GenerateSoftmaxOp<T>(ParameterExpression input, int axis)
    {
        var method = FindMethod<T>("Softmax", typeof(ComputationNode<T>), typeof(int));
        return Expression.Call(method, input, Expression.Constant(axis));
    }

    /// <summary>
    /// Generates code for a sum operation.
    /// </summary>
    private Expression GenerateSumOp<T>(ParameterExpression input, int[]? axes, bool keepDims)
    {
        var method = FindMethod<T>("Sum", typeof(ComputationNode<T>), typeof(int[]), typeof(bool));
        return Expression.Call(method, input, Expression.Constant(axes), Expression.Constant(keepDims));
    }

    /// <summary>
    /// Generates code for a reduce operation.
    /// </summary>
    private Expression GenerateReduceOp<T>(string methodName, ParameterExpression input, int[]? axes, bool keepDims)
    {
        var method = FindMethod<T>(methodName, typeof(ComputationNode<T>), typeof(int[]), typeof(bool));
        return Expression.Call(method, input, Expression.Constant(axes), Expression.Constant(keepDims));
    }

    /// <summary>
    /// Generates code for a reshape operation.
    /// </summary>
    private Expression GenerateReshapeOp<T>(ParameterExpression input, int[] newShape)
    {
        var method = FindMethod<T>("Reshape", typeof(ComputationNode<T>), typeof(int[]));
        return Expression.Call(method, input, Expression.Constant(newShape));
    }

    /// <summary>
    /// Generates code for a concatenation operation.
    /// </summary>
    private Expression GenerateConcatOp<T>(ParameterExpression[] inputs, int axis)
    {
        var method = FindMethod<T>("Concat", typeof(ComputationNode<T>[]), typeof(int));
        var inputArray = Expression.NewArrayInit(typeof(ComputationNode<T>), inputs);
        return Expression.Call(method, inputArray, Expression.Constant(axis));
    }

    /// <summary>
    /// Generates code for a 2D convolution operation.
    /// </summary>
    private Expression GenerateConv2DOp<T>(ParameterExpression[] inputs, Conv2DOp op)
    {
        // This is a simplified placeholder - full implementation would handle all Conv2D parameters
        var method = FindMethod<T>("Conv2D", typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(int[]), typeof(int[]));
        return Expression.Call(method, inputs[0], inputs[1],
            Expression.Constant(op.Stride), Expression.Constant(op.Padding));
    }

    /// <summary>
    /// Generates code for a 2D max pooling operation.
    /// </summary>
    private Expression GenerateMaxPool2DOp<T>(ParameterExpression input, MaxPool2DOp op)
    {
        var method = FindMethod<T>("MaxPool2D", typeof(ComputationNode<T>),
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
        var method = FindMethod<T>("AvgPool2D", typeof(ComputationNode<T>),
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
        var method = FindMethod<T>("LayerNorm", typeof(ComputationNode<T>),
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
        var method = FindMethod<T>("BatchNorm", typeof(ComputationNode<T>),
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
    private MethodInfo FindMethod<T>(string methodName, params Type[] parameterTypes)
    {
        var method = _tensorOperationsMethods.FirstOrDefault(m =>
            m.Name == methodName &&
            m.GetParameters().Length == parameterTypes.Length);

        if (method == null)
        {
            throw new InvalidOperationException(
                $"Could not find TensorOperations method '{methodName}' with {parameterTypes.Length} parameters");
        }

        // If method is generic, specialize it with the element type T
        if (method.IsGenericMethodDefinition)
        {
            method = method.MakeGenericMethod(typeof(T));
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

    // ========== Additional Backward Operation Code Generators ==========

    /// <summary>
    /// Generates code for GradLayerNorm operation.
    /// </summary>
    private Expression GenerateGradLayerNormOp<T>(ParameterExpression[] inputs, Operations.GradLayerNormOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradLayerNorm")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0], // gradOutput
            inputs[1], // saved tensor
            Expression.Constant(op.InputIndex),
            Expression.Constant(op.Epsilon),
            Expression.Constant(op.NormalizedShape));
    }

    /// <summary>
    /// Generates code for GradReshape operation.
    /// </summary>
    private Expression GenerateGradReshapeOp<T>(ParameterExpression[] inputs, Operations.GradReshapeOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradReshape")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], Expression.Constant(op.OriginalShape));
    }

    /// <summary>
    /// Generates code for GradTranspose operation.
    /// </summary>
    private Expression GenerateGradTransposeOp<T>(ParameterExpression[] inputs, Operations.GradTransposeOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradTranspose")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], Expression.Constant(op.Axes, typeof(int[])));
    }

    /// <summary>
    /// Generates code for GradConcat operation.
    /// </summary>
    private Expression GenerateGradConcatOp<T>(ParameterExpression[] inputs, Operations.GradConcatOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradConcat")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0],
            Expression.Constant(op.Axis),
            Expression.Constant(op.StartIndex),
            Expression.Constant(op.Size));
    }

    /// <summary>
    /// Generates code for GradSplit operation.
    /// </summary>
    private Expression GenerateGradSplitOp<T>(ParameterExpression[] inputs, Operations.GradSplitOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradSplit")!.MakeGenericMethod(typeof(T));
        var inputArray = Expression.NewArrayInit(typeof(Tensor<T>), inputs);
        return Expression.Call(method, inputArray, Expression.Constant(op.Axis));
    }

    /// <summary>
    /// Generates code for GradDivide operation.
    /// </summary>
    private Expression GenerateGradDivideOp<T>(ParameterExpression[] inputs, Operations.GradDivideOp op)
    {
        if (op.InputIndex == 0)
        {
            // Gradient for numerator
            var method = typeof(GradientOps).GetMethod("GradDivideNumerator")!.MakeGenericMethod(typeof(T));
            return Expression.Call(method, inputs[0], inputs[1]);
        }
        else
        {
            // Gradient for denominator
            var method = typeof(GradientOps).GetMethod("GradDivideDenominator")!.MakeGenericMethod(typeof(T));
            return Expression.Call(method, inputs[0], inputs[1], inputs[2]);
        }
    }

    /// <summary>
    /// Generates code for GradPower operation.
    /// </summary>
    private Expression GenerateGradPowerOp<T>(ParameterExpression[] inputs, Operations.GradPowerOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradPower")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1], Expression.Constant(op.Exponent));
    }

    /// <summary>
    /// Generates code for GradSqrt operation.
    /// </summary>
    private Expression GenerateGradSqrtOp<T>(ParameterExpression[] inputs)
    {
        var method = typeof(GradientOps).GetMethod("GradSqrt")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    /// <summary>
    /// Generates code for GradSum operation.
    /// </summary>
    private Expression GenerateGradSumOp<T>(ParameterExpression[] inputs, Operations.GradSumOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradSum")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0],
            Expression.Constant(op.OriginalShape),
            Expression.Constant(op.Axes, typeof(int[])));
    }

    /// <summary>
    /// Generates code for GradMean operation.
    /// </summary>
    private Expression GenerateGradMeanOp<T>(ParameterExpression[] inputs, Operations.GradMeanOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradMean")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0],
            Expression.Constant(op.OriginalShape),
            Expression.Constant(op.Count));
    }

    /// <summary>
    /// Generates code for GradSlice operation.
    /// </summary>
    private Expression GenerateGradSliceOp<T>(ParameterExpression[] inputs, Operations.GradSliceOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradSlice")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0],
            Expression.Constant(op.OriginalShape),
            Expression.Constant(op.StartIndices));
    }

    /// <summary>
    /// Generates code for GradPad operation.
    /// </summary>
    private Expression GenerateGradPadOp<T>(ParameterExpression[] inputs, Operations.GradPadOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradPad")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], Expression.Constant(op.Padding));
    }

    /// <summary>
    /// Generates code for GradDropout operation.
    /// </summary>
    private Expression GenerateGradDropoutOp<T>(ParameterExpression[] inputs, Operations.GradDropoutOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradDropout")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1], Expression.Constant(op.Probability));
    }

    /// <summary>
    /// Generates code for GradEmbedding operation.
    /// </summary>
    private Expression GenerateGradEmbeddingOp<T>(ParameterExpression[] inputs, Operations.GradEmbeddingOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradEmbedding")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1], Expression.Constant(op.EmbeddingShape));
    }

    /// <summary>
    /// Generates code for GradGather operation.
    /// </summary>
    private Expression GenerateGradGatherOp<T>(ParameterExpression[] inputs, Operations.GradGatherOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradGather")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0],
            inputs[1],
            Expression.Constant(op.Axis),
            Expression.Constant(op.InputShape));
    }

    /// <summary>
    /// Generates code for GradLeakyReLU operation.
    /// </summary>
    private Expression GenerateGradLeakyReLUOp<T>(ParameterExpression[] inputs, Operations.GradLeakyReLUOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradLeakyReLU")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1], Expression.Constant(op.Alpha));
    }

    /// <summary>
    /// Generates code for GradGELU operation.
    /// </summary>
    private Expression GenerateGradGELUOp<T>(ParameterExpression[] inputs, Operations.GradGELUOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradGELU")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method, inputs[0], inputs[1], Expression.Constant(op.Approximate));
    }

    /// <summary>
    /// Generates code for GradBroadcast operation.
    /// </summary>
    private Expression GenerateGradBroadcastOp<T>(ParameterExpression[] inputs, Operations.GradBroadcastOp op)
    {
        var method = typeof(GradientOps).GetMethod("GradBroadcast")!.MakeGenericMethod(typeof(T));
        return Expression.Call(method,
            inputs[0],
            Expression.Constant(op.OriginalShape),
            Expression.Constant(op.BroadcastedAxes));
    }

    // ========== Unrolled Operation Code Generators ==========

    /// <summary>
    /// Generates code for an unrolled sequence of operations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Unrolled sequences combine multiple element-wise operations into a single fused kernel.
    /// The sequence is executed inline without loop overhead, improving instruction-level parallelism.
    /// </para>
    /// </remarks>
    private Expression GenerateUnrolledSequenceOp<T>(ParameterExpression[] inputs, Operations.UnrolledSequenceOp op)
    {
        var method = typeof(UnrolledOps).GetMethod("ExecuteUnrolledSequence")!.MakeGenericMethod(typeof(T));
        var operationsArray = Expression.Constant(op.Operations.ToArray());
        // Extract Tensor<T> from ComputationNode<T>.Value for runtime operations
        var valueProperty = typeof(ComputationNode<T>).GetProperty("Value")!;
        var inputValue = Expression.Property(inputs[0], valueProperty);
        var tensorResult = Expression.Call(method,
            inputValue,
            operationsArray,
            Expression.Constant(op.UnrollFactor));
        // Wrap the Tensor<T> result back into ComputationNode<T>
        var variableMethod = typeof(TensorOperations<T>).GetMethod("Variable", new[] { typeof(Tensor<T>), typeof(string), typeof(bool) })!;
        return Expression.Call(variableMethod, tensorResult, Expression.Constant("unrolled_seq"), Expression.Constant(false));
    }

    /// <summary>
    /// Generates code for an unrolled element-wise operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Processes small tensors with loop unrolling to reduce loop overhead and enable
    /// better instruction pipelining. Particularly effective for tensors up to 64 elements.
    /// </para>
    /// </remarks>
    private Expression GenerateUnrolledElementwiseOp<T>(ParameterExpression[] inputs, Operations.UnrolledElementwiseOp op)
    {
        var method = typeof(UnrolledOps).GetMethod("ExecuteUnrolledElementwise")!.MakeGenericMethod(typeof(T));
        // Extract Tensor<T> from ComputationNode<T>.Value for runtime operations
        var valueProperty = typeof(ComputationNode<T>).GetProperty("Value")!;
        var inputValue = Expression.Property(inputs[0], valueProperty);
        var tensorResult = Expression.Call(method,
            inputValue,
            Expression.Constant(op.BaseOperation),
            Expression.Constant(op.UnrollFactor),
            Expression.Constant(op.TotalElements));
        // Wrap the Tensor<T> result back into ComputationNode<T>
        var variableMethod = typeof(TensorOperations<T>).GetMethod("Variable", new[] { typeof(Tensor<T>), typeof(string), typeof(bool) })!;
        return Expression.Call(variableMethod, tensorResult, Expression.Constant("unrolled_elem"), Expression.Constant(false));
    }

    /// <summary>
    /// Generates code for an unrolled reduction operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Performs reductions (sum, mean, max) with loop unrolling for small tensor sizes.
    /// Uses tree reduction pattern for better parallelism.
    /// </para>
    /// </remarks>
    private Expression GenerateUnrolledReductionOp<T>(ParameterExpression[] inputs, Operations.UnrolledReductionOp op)
    {
        var method = typeof(UnrolledOps).GetMethod("ExecuteUnrolledReduction")!.MakeGenericMethod(typeof(T));
        // Extract Tensor<T> from ComputationNode<T>.Value for runtime operations
        var valueProperty = typeof(ComputationNode<T>).GetProperty("Value")!;
        var inputValue = Expression.Property(inputs[0], valueProperty);
        var tensorResult = Expression.Call(method,
            inputValue,
            Expression.Constant(op.ReductionType),
            Expression.Constant(op.UnrollFactor));
        // Wrap the Tensor<T> result back into ComputationNode<T>
        var variableMethod = typeof(TensorOperations<T>).GetMethod("Variable", new[] { typeof(Tensor<T>), typeof(string), typeof(bool) })!;
        return Expression.Call(variableMethod, tensorResult, Expression.Constant("unrolled_red"), Expression.Constant(false));
    }

    // ========== Vectorized Operation Code Generators ==========

    /// <summary>
    /// Generates code for a vectorized binary operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses SIMD instructions (SSE/AVX) to process multiple elements in parallel.
    /// Handles both the vectorized portion and any scalar remainder.
    /// </para>
    /// </remarks>
    private Expression GenerateVectorizedBinaryOp<T>(ParameterExpression[] inputs, Operations.VectorizedBinaryOp op)
    {
        // Find the string-based overload (3rd parameter is string)
        var method = typeof(VectorizedOps)
            .GetMethods()
            .First(m => m.Name == "ExecuteVectorizedBinary" && m.GetParameters()[2].ParameterType == typeof(string))
            .MakeGenericMethod(typeof(T));
        // Extract Tensor<T> from ComputationNode<T>.Value for runtime operations
        var valueProperty = typeof(ComputationNode<T>).GetProperty("Value")!;
        var leftValue = Expression.Property(inputs[0], valueProperty);
        var rightValue = Expression.Property(inputs[1], valueProperty);
        var tensorResult = Expression.Call(method,
            leftValue,
            rightValue,
            Expression.Constant(op.Operation),
            Expression.Constant(op.VectorWidth),
            Expression.Constant(op.NumVectors),
            Expression.Constant(op.Remainder));
        // Wrap the Tensor<T> result back into ComputationNode<T>
        var variableMethod = typeof(TensorOperations<T>).GetMethod("Variable", new[] { typeof(Tensor<T>), typeof(string), typeof(bool) })!;
        return Expression.Call(variableMethod, tensorResult, Expression.Constant("vec_binary"), Expression.Constant(false));
    }

    /// <summary>
    /// Generates code for a vectorized unary operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Applies unary operations (Negate, Exp, Log, ReLU, etc.) using SIMD instructions.
    /// Significantly faster than scalar operations for large tensors.
    /// </para>
    /// </remarks>
    private Expression GenerateVectorizedUnaryOp<T>(ParameterExpression[] inputs, Operations.VectorizedUnaryOp op)
    {
        // Find the string-based overload (2nd parameter is string)
        var method = typeof(VectorizedOps)
            .GetMethods()
            .First(m => m.Name == "ExecuteVectorizedUnary" && m.GetParameters()[1].ParameterType == typeof(string))
            .MakeGenericMethod(typeof(T));
        // Extract Tensor<T> from ComputationNode<T>.Value for runtime operations
        var valueProperty = typeof(ComputationNode<T>).GetProperty("Value")!;
        var inputValue = Expression.Property(inputs[0], valueProperty);
        var tensorResult = Expression.Call(method,
            inputValue,
            Expression.Constant(op.Operation),
            Expression.Constant(op.VectorWidth),
            Expression.Constant(op.NumVectors),
            Expression.Constant(op.Remainder));
        // Wrap the Tensor<T> result back into ComputationNode<T>
        var variableMethod = typeof(TensorOperations<T>).GetMethod("Variable", new[] { typeof(Tensor<T>), typeof(string), typeof(bool) })!;
        return Expression.Call(variableMethod, tensorResult, Expression.Constant("vec_unary"), Expression.Constant(false));
    }

    /// <summary>
    /// Generates code for a vectorized reduction operation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Performs reductions (sum, mean, max) using SIMD instructions with horizontal
    /// reduction for combining vector lanes at the end.
    /// </para>
    /// </remarks>
    private Expression GenerateVectorizedReductionOp<T>(ParameterExpression[] inputs, Operations.VectorizedReductionOp op)
    {
        // Find the string-based overload (2nd parameter is string)
        var method = typeof(VectorizedOps)
            .GetMethods()
            .First(m => m.Name == "ExecuteVectorizedReduction" && m.GetParameters()[1].ParameterType == typeof(string))
            .MakeGenericMethod(typeof(T));
        // Extract Tensor<T> from ComputationNode<T>.Value for runtime operations
        var valueProperty = typeof(ComputationNode<T>).GetProperty("Value")!;
        var inputValue = Expression.Property(inputs[0], valueProperty);
        var tensorResult = Expression.Call(method,
            inputValue,
            Expression.Constant(op.ReductionType),
            Expression.Constant(op.VectorWidth),
            Expression.Constant(op.Axes, typeof(int[])),
            Expression.Constant(op.KeepDims));
        // Wrap the Tensor<T> result back into ComputationNode<T>
        var variableMethod = typeof(TensorOperations<T>).GetMethod("Variable", new[] { typeof(Tensor<T>), typeof(string), typeof(bool) })!;
        return Expression.Call(variableMethod, tensorResult, Expression.Constant("vec_reduce"), Expression.Constant(false));
    }

    /// <summary>
    /// Generates code for a vectorized matrix multiplication.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses tiled matrix multiplication with SIMD instructions for the inner loops.
    /// Optimized for cache locality and instruction-level parallelism.
    /// </para>
    /// </remarks>
    private Expression GenerateVectorizedMatMulOp<T>(ParameterExpression[] inputs, Operations.VectorizedMatMulOp op)
    {
        var method = typeof(VectorizedOps).GetMethod("ExecuteVectorizedMatMul")!.MakeGenericMethod(typeof(T));
        // Extract Tensor<T> from ComputationNode<T>.Value for runtime operations
        var valueProperty = typeof(ComputationNode<T>).GetProperty("Value")!;
        var leftValue = Expression.Property(inputs[0], valueProperty);
        var rightValue = Expression.Property(inputs[1], valueProperty);
        var tensorResult = Expression.Call(method,
            leftValue,
            rightValue,
            Expression.Constant(op.VectorWidth),
            Expression.Constant(op.TileSize));
        // Wrap the Tensor<T> result back into ComputationNode<T>
        var variableMethod = typeof(TensorOperations<T>).GetMethod("Variable", new[] { typeof(Tensor<T>), typeof(string), typeof(bool) })!;
        return Expression.Call(variableMethod, tensorResult, Expression.Constant("vec_matmul"), Expression.Constant(false));
    }

    // ========== Extended Activation Operation Code Generators ==========

    /// <summary>
    /// Generates code for ELU activation.
    /// </summary>
    private Expression GenerateELUOp<T>(ParameterExpression input, double alpha)
    {
        var method = FindMethod<T>("ELU", typeof(ComputationNode<T>), typeof(double));
        return Expression.Call(method, input, Expression.Constant(alpha));
    }

    /// <summary>
    /// Generates code for Leaky ReLU activation.
    /// </summary>
    private Expression GenerateLeakyReLUOp<T>(ParameterExpression input, double alpha)
    {
        var method = FindMethod<T>("LeakyReLU", typeof(ComputationNode<T>), typeof(double));
        return Expression.Call(method, input, Expression.Constant(alpha));
    }

    /// <summary>
    /// Generates code for GELU activation.
    /// </summary>
    private Expression GenerateGELUOp<T>(ParameterExpression input, bool approximate)
    {
        var method = FindMethod<T>("GELU", typeof(ComputationNode<T>), typeof(bool));
        return Expression.Call(method, input, Expression.Constant(approximate));
    }

    /// <summary>
    /// Generates code for SoftPlus activation.
    /// </summary>
    private Expression GenerateSoftPlusOp<T>(ParameterExpression input, double beta, double threshold)
    {
        var method = FindMethod<T>("SoftPlus", typeof(ComputationNode<T>), typeof(double), typeof(double));
        return Expression.Call(method, input, Expression.Constant(beta), Expression.Constant(threshold));
    }

    /// <summary>
    /// Generates code for HardTanh activation.
    /// </summary>
    private Expression GenerateHardTanhOp<T>(ParameterExpression input, double minVal, double maxVal)
    {
        var method = FindMethod<T>("HardTanh", typeof(ComputationNode<T>), typeof(double), typeof(double));
        return Expression.Call(method, input, Expression.Constant(minVal), Expression.Constant(maxVal));
    }

    /// <summary>
    /// Generates code for CELU activation.
    /// </summary>
    private Expression GenerateCELUOp<T>(ParameterExpression input, double alpha)
    {
        var method = FindMethod<T>("CELU", typeof(ComputationNode<T>), typeof(double));
        return Expression.Call(method, input, Expression.Constant(alpha));
    }

    /// <summary>
    /// Generates code for LogSoftmax activation.
    /// </summary>
    private Expression GenerateLogSoftmaxOp<T>(ParameterExpression input, int axis)
    {
        var method = FindMethod<T>("LogSoftmax", typeof(ComputationNode<T>), typeof(int));
        return Expression.Call(method, input, Expression.Constant(axis));
    }

    /// <summary>
    /// Generates code for ThresholdedReLU activation.
    /// </summary>
    private Expression GenerateThresholdedReLUOp<T>(ParameterExpression input, double threshold)
    {
        var method = FindMethod<T>("ThresholdedReLU", typeof(ComputationNode<T>), typeof(double));
        return Expression.Call(method, input, Expression.Constant(threshold));
    }

    // ========== Shape Operation Code Generators ==========

    /// <summary>
    /// Generates code for split operation.
    /// </summary>
    private Expression GenerateSplitOp<T>(ParameterExpression input, SplitOp op)
    {
        if (op.SplitSizes.Length > 0)
        {
            var method = FindMethod<T>("Split", typeof(ComputationNode<T>), typeof(int[]), typeof(int));
            return Expression.Call(method, input, Expression.Constant(op.SplitSizes), Expression.Constant(op.Axis));
        }
        else
        {
            var method = FindMethod<T>("Split", typeof(ComputationNode<T>), typeof(int), typeof(int));
            return Expression.Call(method, input, Expression.Constant(op.NumSplits), Expression.Constant(op.Axis));
        }
    }

    /// <summary>
    /// Generates code for slice operation.
    /// </summary>
    private Expression GenerateSliceOp<T>(ParameterExpression input, SliceOp op)
    {
        var method = FindMethod<T>("Slice", typeof(ComputationNode<T>), typeof(int[]), typeof(int[]), typeof(int[]), typeof(int[]));
        return Expression.Call(method, input,
            Expression.Constant(op.Starts),
            Expression.Constant(op.Ends),
            Expression.Constant(op.Steps),
            Expression.Constant(op.Axes));
    }

    /// <summary>
    /// Generates code for norm operation.
    /// </summary>
    private Expression GenerateNormOp<T>(ParameterExpression input, int axis, bool keepDims)
    {
        var method = FindMethod<T>("Norm", typeof(ComputationNode<T>), typeof(int), typeof(bool));
        return Expression.Call(method, input, Expression.Constant(axis), Expression.Constant(keepDims));
    }

    // ========== Embedding and Attention Code Generators ==========

    /// <summary>
    /// Generates code for embedding operation.
    /// </summary>
    private Expression GenerateEmbeddingOp<T>(ParameterExpression[] inputs, EmbeddingOp op)
    {
        var method = FindMethod<T>("Embedding", typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(int?));
        return Expression.Call(method, inputs[0], inputs[1],
            op.PaddingIdx.HasValue
                ? Expression.Constant(op.PaddingIdx, typeof(int?))
                : Expression.Constant(null, typeof(int?)));
    }

    /// <summary>
    /// Generates code for scaled dot-product attention.
    /// </summary>
    private Expression GenerateScaledDotProductAttentionOp<T>(ParameterExpression[] inputs, ScaledDotProductAttentionOp op)
    {
        var method = FindMethod<T>("ScaledDotProductAttention",
            typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(ComputationNode<T>), typeof(double?), typeof(bool), typeof(double));

        Expression maskInput = inputs.Length > 3
            ? inputs[3]
            : Expression.Constant(null, typeof(Tensor<T>));

        return Expression.Call(method,
            inputs[0], // query
            inputs[1], // key
            inputs[2], // value
            maskInput,
            op.Scale.HasValue
                ? Expression.Constant(op.Scale, typeof(double?))
                : Expression.Constant(null, typeof(double?)),
            Expression.Constant(op.IsCausal),
            Expression.Constant(op.DropoutProbability));
    }

    /// <summary>
    /// Generates code for multi-head attention.
    /// </summary>
    private Expression GenerateMultiHeadAttentionOp<T>(ParameterExpression[] inputs, MultiHeadAttentionOp op)
    {
        var method = FindMethod<T>("MultiHeadAttention",
            typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(int), typeof(double));

        return Expression.Call(method,
            inputs[0], // query
            inputs[1], // key
            inputs[2], // value
            inputs[3], // W_q
            inputs[4], // W_k
            inputs[5], // W_v
            inputs[6], // W_o
            Expression.Constant(op.NumHeads),
            Expression.Constant(op.DropoutProbability));
    }

    // ========== Fused Operation Code Generators ==========

    /// <summary>
    /// Generates code for fused MatMul + Add operation.
    /// </summary>
    private Expression GenerateFusedMatMulAddOp<T>(ParameterExpression[] inputs)
    {
        var method = FindMethod<T>("FusedMatMulAdd",
            typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>));
        return Expression.Call(method, inputs[0], inputs[1], inputs[2]);
    }

    /// <summary>
    /// Generates code for fused Linear + ReLU operation.
    /// </summary>
    private Expression GenerateFusedLinearReLUOp<T>(ParameterExpression[] inputs)
    {
        var method = FindMethod<T>("FusedLinearReLU",
            typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>));
        return Expression.Call(method, inputs[0], inputs[1], inputs[2]);
    }

    /// <summary>
    /// Generates code for fused Conv + BatchNorm operation.
    /// </summary>
    private Expression GenerateFusedConvBatchNormOp<T>(ParameterExpression[] inputs, FusedConvBatchNormOp op)
    {
        var method = FindMethod<T>("FusedConvBatchNorm",
            typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(int[]), typeof(int[]), typeof(double));
        return Expression.Call(method,
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            Expression.Constant(op.Stride),
            Expression.Constant(op.Padding),
            Expression.Constant(op.Epsilon));
    }

    /// <summary>
    /// Generates code for fused Add + ReLU operation.
    /// </summary>
    private Expression GenerateFusedAddReLUOp<T>(ParameterExpression[] inputs)
    {
        var method = FindMethod<T>("FusedAddReLU", typeof(ComputationNode<T>), typeof(ComputationNode<T>));
        return Expression.Call(method, inputs[0], inputs[1]);
    }

    // ========== Complex Number Operation Code Generators ==========

    /// <summary>
    /// Generates code for complex matrix multiplication.
    /// </summary>
    private Expression GenerateComplexMatMulOp<T>(ParameterExpression[] inputs)
    {
        var method = FindMethod<T>("ComplexMatMul",
            typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(ComputationNode<T>), typeof(ComputationNode<T>));
        return Expression.Call(method, inputs[0], inputs[1], inputs[2], inputs[3]);
    }

    /// <summary>
    /// Generates code for complex element-wise multiplication.
    /// </summary>
    private Expression GenerateComplexMultiplyOp<T>(ParameterExpression[] inputs)
    {
        var method = FindMethod<T>("ComplexMultiply",
            typeof(ComputationNode<T>), typeof(ComputationNode<T>),
            typeof(ComputationNode<T>), typeof(ComputationNode<T>));
        return Expression.Call(method, inputs[0], inputs[1], inputs[2], inputs[3]);
    }

    // ========== Octonion Operation Code Generators ==========

    /// <summary>
    /// Generates code for octonion matrix multiplication.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Octonion matrix multiplication operates on 8-dimensional octonion values.
    /// Each input/weight element represents an octonion (8 components per value).
    /// </para>
    /// </remarks>
    private Expression GenerateOctonionMatMulOp<T>(ParameterExpression[] inputs, OctonionMatMulOp op)
    {
        // OctonionMatMul takes input and weights, with optional biases
        if (inputs.Length >= 3)
        {
            // With biases
            var method = FindMethod<T>("OctonionMatMul",
                typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>));
            return Expression.Call(method, inputs[0], inputs[1], inputs[2]);
        }
        else
        {
            // Without biases (biases = null)
            var method = FindMethod<T>("OctonionMatMul",
                typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>));
            return Expression.Call(method, inputs[0], inputs[1],
                Expression.Constant(null, typeof(ComputationNode<T>)));
        }
    }

    // ========== Dropout Operation Code Generator ==========

    /// <summary>
    /// Generates code for dropout operation.
    /// </summary>
    private Expression GenerateDropoutOp<T>(ParameterExpression input, DropoutOp op)
    {
        var method = FindMethod<T>("Dropout", typeof(ComputationNode<T>), typeof(double), typeof(bool));
        return Expression.Call(method, input,
            Expression.Constant(op.Probability),
            Expression.Constant(op.Training));
    }

    // ========== Additional Extended Activation Operation Code Generators ==========

    /// <summary>
    /// Generates code for ScaledTanh activation.
    /// </summary>
    private Expression GenerateScaledTanhOp<T>(ParameterExpression input, double beta)
    {
        var method = FindMethod<T>("ScaledTanh", typeof(ComputationNode<T>), typeof(double));
        return Expression.Call(method, input, Expression.Constant(beta));
    }

    /// <summary>
    /// Generates code for ISRU activation.
    /// </summary>
    private Expression GenerateISRUOp<T>(ParameterExpression input, double alpha)
    {
        var method = FindMethod<T>("ISRU", typeof(ComputationNode<T>), typeof(double));
        return Expression.Call(method, input, Expression.Constant(alpha));
    }

    /// <summary>
    /// Generates code for Softmin activation.
    /// </summary>
    private Expression GenerateSoftminOp<T>(ParameterExpression input, int axis)
    {
        var method = FindMethod<T>("Softmin", typeof(ComputationNode<T>), typeof(int));
        return Expression.Call(method, input, Expression.Constant(axis));
    }

    /// <summary>
    /// Generates code for LogSoftmin activation.
    /// </summary>
    private Expression GenerateLogSoftminOp<T>(ParameterExpression input, int axis)
    {
        var method = FindMethod<T>("LogSoftmin", typeof(ComputationNode<T>), typeof(int));
        return Expression.Call(method, input, Expression.Constant(axis));
    }

    /// <summary>
    /// Generates code for Maxout activation.
    /// </summary>
    private Expression GenerateMaxoutOp<T>(ParameterExpression input, int numPieces)
    {
        var method = FindMethod<T>("Maxout", typeof(ComputationNode<T>), typeof(int));
        return Expression.Call(method, input, Expression.Constant(numPieces));
    }

    /// <summary>
    /// Generates code for RReLU activation.
    /// </summary>
    private Expression GenerateRReLUOp<T>(ParameterExpression input, double lower, double upper)
    {
        var method = FindMethod<T>("RReLU", typeof(ComputationNode<T>), typeof(double), typeof(double));
        return Expression.Call(method, input, Expression.Constant(lower), Expression.Constant(upper));
    }

    /// <summary>
    /// Generates code for SphericalSoftmax activation.
    /// </summary>
    private Expression GenerateSphericalSoftmaxOp<T>(ParameterExpression input, int axis)
    {
        var method = FindMethod<T>("SphericalSoftmax", typeof(ComputationNode<T>), typeof(int));
        return Expression.Call(method, input, Expression.Constant(axis));
    }

    /// <summary>
    /// Generates code for TaylorSoftmax activation.
    /// </summary>
    private Expression GenerateTaylorSoftmaxOp<T>(ParameterExpression input, int axis, int order)
    {
        var method = FindMethod<T>("TaylorSoftmax", typeof(ComputationNode<T>), typeof(int), typeof(int));
        return Expression.Call(method, input, Expression.Constant(axis), Expression.Constant(order));
    }

    /// <summary>
    /// Generates code for Sparsemax activation.
    /// </summary>
    private Expression GenerateSparsemaxOp<T>(ParameterExpression input, int axis)
    {
        var method = FindMethod<T>("Sparsemax", typeof(ComputationNode<T>), typeof(int));
        return Expression.Call(method, input, Expression.Constant(axis));
    }

    /// <summary>
    /// Generates code for HierarchicalSoftmax activation.
    /// </summary>
    private Expression GenerateHierarchicalSoftmaxOp<T>(ParameterExpression input, int[] treeStructure)
    {
        var method = FindMethod<T>("HierarchicalSoftmax", typeof(ComputationNode<T>), typeof(int[]));
        return Expression.Call(method, input, Expression.Constant(treeStructure));
    }

    // ========================================================================
    // Differentiable Approximation Operation Code Generators
    // ========================================================================

    /// <summary>
    /// Generates code for SoftSplit operation (differentiable decision tree split).
    /// </summary>
    private Expression GenerateSoftSplitOp<T>(ParameterExpression[] inputs, Operations.SoftSplitOp op)
    {
        // inputs[0] = input features, inputs[1] = leftValue, inputs[2] = rightValue
        var method = typeof(TensorOperations<T>).GetMethod("SoftSplit",
            new[] { typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>),
                    typeof(int), typeof(T), typeof(T) });

        if (method == null)
            throw new InvalidOperationException("SoftSplit method not found on TensorOperations");

        return Expression.Call(method,
            inputs[0],
            inputs[1],
            inputs[2],
            Expression.Constant(op.FeatureIndex),
            Expression.Constant((T)(object)op.Threshold, typeof(T)),
            Expression.Constant((T)(object)op.Temperature, typeof(T)));
    }

    /// <summary>
    /// Generates code for SoftKNN operation (differentiable k-nearest neighbors).
    /// </summary>
    private Expression GenerateSoftKNNOp<T>(ParameterExpression[] inputs, Operations.SoftKNNOp op)
    {
        // inputs[0] = input, inputs[1] = supportVectors, inputs[2] = labels
        var method = typeof(TensorOperations<T>).GetMethod("SoftKNN",
            new[] { typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(T) });

        if (method == null)
            throw new InvalidOperationException("SoftKNN method not found on TensorOperations");

        return Expression.Call(method,
            inputs[0],
            inputs[1],
            inputs[2],
            Expression.Constant((T)(object)op.Temperature, typeof(T)));
    }

    /// <summary>
    /// Generates code for SoftLocallyWeighted operation (differentiable locally-weighted regression).
    /// </summary>
    private Expression GenerateSoftLocallyWeightedOp<T>(ParameterExpression[] inputs, Operations.SoftLocallyWeightedOp op)
    {
        // inputs[0] = input, inputs[1] = xTrain, inputs[2] = yTrain
        var method = typeof(TensorOperations<T>).GetMethod("SoftLocallyWeighted",
            new[] { typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(ComputationNode<T>), typeof(T) });

        if (method == null)
            throw new InvalidOperationException("SoftLocallyWeighted method not found on TensorOperations");

        return Expression.Call(method,
            inputs[0],
            inputs[1],
            inputs[2],
            Expression.Constant((T)(object)op.Bandwidth, typeof(T)));
    }

    /// <summary>
    /// Generates code for FakeQuantization operation (differentiable quantization with STE).
    /// </summary>
    private Expression GenerateFakeQuantizationOp<T>(ParameterExpression[] inputs, Operations.FakeQuantizationOp op)
    {
        // inputs[0] = input
        var method = typeof(TensorOperations<T>).GetMethod("FakeQuantize",
            new[] { typeof(ComputationNode<T>), typeof(int), typeof(T), typeof(T), typeof(bool) });

        if (method == null)
            throw new InvalidOperationException("FakeQuantize method not found on TensorOperations");

        var scale = op.Scale.HasValue ? (T)(object)op.Scale.Value : default(T);
        var zeroPoint = (T)(object)op.ZeroPoint;

        return Expression.Call(method,
            inputs[0],
            Expression.Constant(op.NumBits),
            Expression.Constant(scale, typeof(T)),
            Expression.Constant(zeroPoint, typeof(T)),
            Expression.Constant(op.Symmetric));
    }
}
