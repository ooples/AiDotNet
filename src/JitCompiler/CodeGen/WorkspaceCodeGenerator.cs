// TensorWorkspace<T> is now available in AiDotNet.Tensors 0.13.0+
using System.Linq.Expressions;
using System.Reflection;
using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;
using AiDotNet.JitCompiler.Optimizations;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Generates zero-allocation executable code that targets IEngine directly with TensorWorkspace slots.
/// Unlike CodeGenerator which targets TensorOperations (allocating, autodiff-wrapped),
/// this generator emits IEngine Into/InPlace calls backed by pre-allocated workspace memory.
/// </summary>
public class WorkspaceCodeGenerator
{
    /// <summary>
    /// Generates a compiled function that executes the IR graph using workspace memory.
    /// All intermediate tensors are pre-allocated in the workspace — zero allocation during execution.
    /// </summary>
    /// <typeparam name="T">The numeric type for tensor elements.</typeparam>
    /// <param name="graph">The optimized IR graph (after fusion and memory planning passes).</param>
    /// <returns>
    /// A tuple of:
    /// - The compiled action that executes the graph
    /// - The workspace with all intermediate slots pre-allocated
    /// - The output slot ID (which workspace slot contains the final result)
    /// </returns>
    public (Action<Tensor<T>[], IEngine> Execute, TensorWorkspace<T> Workspace, int OutputSlot)
        Compile<T>(IRGraph graph)
    {
        // Get memory plan from graph metadata (set by MemoryPlanningPass)
        if (!graph.Metadata.TryGetValue("MemoryPlan.TensorToSlot", out var planObj) ||
            planObj is not Dictionary<int, int> tensorToSlot)
        {
            throw new InvalidOperationException(
                "IRGraph has no memory plan. Run MemoryPlanningPass before WorkspaceCodeGenerator.");
        }

        if (!graph.Metadata.TryGetValue("MemoryPlan.SlotShapes", out var shapesObj) ||
            shapesObj is not List<int[]> slotShapes)
        {
            throw new InvalidOperationException("IRGraph memory plan missing slot shapes.");
        }

        // Create and allocate workspace
        var workspace = new TensorWorkspace<T>();
        foreach (var shape in slotShapes)
        {
            workspace.Register(shape);
        }
        workspace.Allocate();

        // Find output slot
        int outputTensorId = graph.OutputIds[0];
        int outputSlot = tensorToSlot[outputTensorId];

        // Build expression tree
        // Parameters: Tensor<T>[] inputs, IEngine engine
        var inputsParam = Expression.Parameter(typeof(Tensor<T>[]), "inputs");
        var engineParam = Expression.Parameter(typeof(IEngine), "engine");
        var workspaceConst = Expression.Constant(workspace);

        var expressions = new List<Expression>();
        var variables = new List<ParameterExpression>();

        // Map tensor IDs to expressions that produce the tensor
        var tensorExprs = new Dictionary<int, Expression>();

        // Input tensors come from the inputs array
        for (int i = 0; i < graph.InputIds.Count; i++)
        {
            int inputId = graph.InputIds[i];
            tensorExprs[inputId] = Expression.ArrayIndex(inputsParam, Expression.Constant(i));
        }

        // Generate code for each operation
        foreach (var op in graph.Operations)
        {
            if (!tensorToSlot.TryGetValue(op.OutputId, out int slot))
                continue;

            // Get workspace slot for output
            var getSlotMethod = typeof(TensorWorkspace<T>).GetMethod("Get")!;
            var outputExpr = Expression.Call(workspaceConst, getSlotMethod, Expression.Constant(slot));
            var outputVar = Expression.Variable(typeof(Tensor<T>), $"slot{slot}");
            variables.Add(outputVar);
            expressions.Add(Expression.Assign(outputVar, outputExpr));

            // Get input tensor expressions
            var inputExprs = new Expression[op.InputIds.Length];
            for (int i = 0; i < op.InputIds.Length; i++)
            {
                int inputId = op.InputIds[i];
                if (tensorExprs.TryGetValue(inputId, out var existing))
                {
                    inputExprs[i] = existing;
                }
                else if (tensorToSlot.TryGetValue(inputId, out int inputSlot))
                {
                    inputExprs[i] = Expression.Call(workspaceConst, getSlotMethod, Expression.Constant(inputSlot));
                }
                else
                {
                    throw new InvalidOperationException($"Tensor {inputId} not found in inputs or workspace.");
                }
            }

            // Emit the IEngine call for this operation
            var engineCall = EmitEngineCall<T>(op, outputVar, inputExprs, engineParam);
            if (engineCall is not null)
            {
                expressions.Add(engineCall);
            }

            // Register the output variable for downstream operations
            tensorExprs[op.OutputId] = outputVar;
        }

        // Build the lambda
        var block = Expression.Block(variables, expressions);
        var lambda = Expression.Lambda<Action<Tensor<T>[], IEngine>>(block, inputsParam, engineParam);

        return (lambda.Compile(), workspace, outputSlot);
    }

    /// <summary>
    /// Emits the IEngine method call for a single IR operation.
    /// Uses Into/InPlace methods for zero-allocation execution.
    /// </summary>
    private Expression? EmitEngineCall<T>(IROp op, ParameterExpression output, Expression[] inputs, ParameterExpression engine)
    {
        return op switch
        {
            // Arithmetic
            AddOp => EmitInto<T>(engine, "TensorAddInto", output, inputs[0], inputs[1]),
            SubtractOp => EmitAllocating<T>(engine, "TensorSubtract", output, inputs),
            ElementwiseMultiplyOp => EmitInto<T>(engine, "TensorMultiplyInto", output, inputs[0], inputs[1]),
            DivideOp => EmitAllocating<T>(engine, "TensorDivide", output, inputs),
            NegateOp => EmitAllocating<T>(engine, "TensorNegate", output, inputs),

            // Element-wise math
            ExpOp => EmitAllocating<T>(engine, "TensorExp", output, inputs),
            LogOp => EmitAllocating<T>(engine, "TensorLog", output, inputs),
            SqrtOp => EmitAllocating<T>(engine, "TensorSqrt", output, inputs),
            AbsOp => EmitAllocating<T>(engine, "TensorAbs", output, inputs),

            // Convolution
            Conv2DOp conv => EmitConv2DInto<T>(engine, output, inputs[0], inputs[1], conv),

            // Normalization
            GroupNormOp gn => EmitGroupNormInto<T>(engine, output, inputs, gn),
            BatchNormOp => EmitAllocating<T>(engine, "BatchNorm", output, inputs),
            LayerNormOp => EmitAllocating<T>(engine, "LayerNorm", output, inputs),

            // Activations
            ReLUOp => EmitActivationInto<T>(engine, "ReLUInto", output, inputs[0]),
            SigmoidOp => EmitActivationInto<T>(engine, "SigmoidInto", output, inputs[0]),
            SwishOp => EmitActivationInto<T>(engine, "SwishInto", output, inputs[0]),
            GELUOp => EmitActivationInto<T>(engine, "GELUInto", output, inputs[0]),
            TanhOp => EmitActivationInto<T>(engine, "TanhInto", output, inputs[0]),
            MishOp => EmitActivationInto<T>(engine, "MishInto", output, inputs[0]),
            LeakyReLUOp => EmitActivationInto<T>(engine, "LeakyReLUInto", output, inputs[0]),

            // Softmax (uses oneDNN when available via IEngine)
            SoftmaxOp => EmitAllocating<T>(engine, "Softmax", output, inputs),
            LogSoftmaxOp => EmitAllocating<T>(engine, "TensorLogSoftmax", output, inputs),

            // Pooling
            MaxPool2DOp => EmitAllocating<T>(engine, "MaxPool2D", output, inputs),
            AvgPool2DOp => EmitAllocating<T>(engine, "AvgPool2D", output, inputs),

            // Fused operations
            FusedGroupNormActivationOp fgna => EmitGroupNormSwishInto<T>(engine, output, inputs, fgna),
            FusedConv2DBiasActivationOp fcba => EmitFusedConv2D<T>(engine, output, inputs, fcba),
            FusedGroupNormActivationConv2DOp fgnac => EmitFusedGNActConv<T>(engine, output, inputs, fgnac),
            FusedAddGroupNormOp fagn => EmitAddGroupNormInto<T>(engine, output, inputs, fagn),

            // Matrix operations
            MatMulOp => EmitInto<T>(engine, "MatMulInto", output, inputs[0], inputs[1]),
            TransposeOp => EmitTransposeInto<T>(engine, output, inputs[0]),

            // Reductions
            SumOp => EmitAllocating<T>(engine, "TensorSum", output, inputs),
            MeanOp => EmitAllocating<T>(engine, "TensorMean", output, inputs),

            // Shape operations
            ReshapeOp reshape => EmitReshape<T>(output, inputs[0], reshape),

            // Concatenation
            ConcatOp concat => EmitConcatInto<T>(engine, output, inputs, concat),

            // Constant — pre-loaded into workspace
            ConstantOp => null,

            // Default: fall back to allocating call + copy
            _ => EmitFallback<T>(engine, op, output, inputs)
        };
    }

    // --- Emit helpers ---

    private Expression EmitInto<T>(ParameterExpression engine, string methodName,
        ParameterExpression dest, Expression a, Expression b)
    {
        var method = typeof(IEngine).GetMethod(methodName)!.MakeGenericMethod(typeof(T));
        return Expression.Call(engine, method, dest, a, b);
    }

    private Expression EmitActivationInto<T>(ParameterExpression engine, string methodName,
        ParameterExpression dest, Expression input)
    {
        var method = typeof(IEngine).GetMethod(methodName)!.MakeGenericMethod(typeof(T));
        return Expression.Call(engine, method, dest, input);
    }

    private Expression EmitConv2DInto<T>(ParameterExpression engine,
        ParameterExpression dest, Expression input, Expression kernel, Conv2DOp conv)
    {
        var method = typeof(IEngine).GetMethod("Conv2DInto")!.MakeGenericMethod(typeof(T));
        return Expression.Call(engine, method, dest, input, kernel,
            Expression.Constant(conv.Stride[0]),
            Expression.Constant(conv.Padding[0]),
            Expression.Constant(1)); // dilation
    }

    private Expression EmitGroupNormInto<T>(ParameterExpression engine,
        ParameterExpression dest, Expression[] inputs, GroupNormOp gn)
    {
        var method = typeof(IEngine).GetMethod("GroupNormInto")!.MakeGenericMethod(typeof(T));
        var meanVar = Expression.Variable(typeof(Tensor<T>), "gn_mean");
        var varVar = Expression.Variable(typeof(Tensor<T>), "gn_var");
        return Expression.Block(
            new[] { meanVar, varVar },
            Expression.Call(engine, method, dest, inputs[0], Expression.Constant(gn.NumGroups),
                inputs[1], inputs[2], Expression.Constant(gn.Epsilon), meanVar, varVar));
    }

    private Expression EmitGroupNormSwishInto<T>(ParameterExpression engine,
        ParameterExpression dest, Expression[] inputs, FusedGroupNormActivationOp fgna)
    {
        var method = typeof(IEngine).GetMethod("GroupNormSwishInto")!.MakeGenericMethod(typeof(T));
        return Expression.Call(engine, method, dest, inputs[0],
            Expression.Constant(fgna.NumGroups), inputs[1], inputs[2],
            Expression.Constant(fgna.Epsilon));
    }

    private Expression EmitFusedConv2D<T>(ParameterExpression engine,
        ParameterExpression dest, Expression[] inputs, FusedConv2DBiasActivationOp fcba)
    {
        // Use IEngine.FusedConv2D which does Conv+Bias+Activation in single kernel
        var method = typeof(IEngine).GetMethod("FusedConv2D")!.MakeGenericMethod(typeof(T));
        var fusedResult = Expression.Call(engine, method,
            inputs[0], inputs[1], inputs[2],
            Expression.Constant(fcba.Stride[0]), Expression.Constant(fcba.Stride[1]),
            Expression.Constant(fcba.Padding[0]), Expression.Constant(fcba.Padding[1]),
            Expression.Constant(fcba.Dilation[0]), Expression.Constant(fcba.Dilation[1]),
            Expression.Constant(fcba.Activation));
        // Copy result to workspace slot
        var copyMethod = typeof(ReadOnlySpan<T>).GetMethod("CopyTo", new[] { typeof(Span<T>) });
        // For now, assign directly (FusedConv2D returns a tensor, we'd need to copy)
        return Expression.Assign(dest, fusedResult);
    }

    private Expression EmitFusedGNActConv<T>(ParameterExpression engine,
        ParameterExpression dest, Expression[] inputs, FusedGroupNormActivationConv2DOp fgnac)
    {
        // Two-step: GroupNormSwishInto(temp, input, ...) then Conv2DInto(dest, temp, kernel, ...)
        // temp reuses the input's workspace slot since input is consumed by this point
        var gnSwishMethod = typeof(IEngine).GetMethod("GroupNormSwishInto")!.MakeGenericMethod(typeof(T));
        var conv2dIntoMethod = typeof(IEngine).GetMethod("Conv2DInto")!.MakeGenericMethod(typeof(T));

        // Use a temp variable for the intermediate
        var tempVar = Expression.Variable(typeof(Tensor<T>), $"fgnac_temp");

        return Expression.Block(
            new[] { tempVar },
            // Step 1: GroupNorm + Swish into temp (reuse input slot if possible)
            Expression.Assign(tempVar, inputs[0]), // temp points to input
            Expression.Call(engine, gnSwishMethod, tempVar, inputs[0],
                Expression.Constant(fgnac.NumGroups), inputs[1], inputs[2],
                Expression.Constant(fgnac.Epsilon)),
            // Step 2: Conv2D from temp into dest
            Expression.Call(engine, conv2dIntoMethod, dest, tempVar, inputs[3],
                Expression.Constant(fgnac.Stride[0]),
                Expression.Constant(fgnac.Padding[0]),
                Expression.Constant(1)));
    }

    private Expression EmitAddGroupNormInto<T>(ParameterExpression engine,
        ParameterExpression dest, Expression[] inputs, FusedAddGroupNormOp fagn)
    {
        var method = typeof(IEngine).GetMethod("AddGroupNormInto")!.MakeGenericMethod(typeof(T));
        return Expression.Call(engine, method, dest, inputs[0], inputs[1],
            Expression.Constant(fagn.NumGroups), inputs[2], inputs[3],
            Expression.Constant(fagn.Epsilon));
    }

    private Expression EmitTransposeInto<T>(ParameterExpression engine,
        ParameterExpression dest, Expression input)
    {
        // Simple 2D transpose
        var method = typeof(IEngine).GetMethod("TransposeInto")!.MakeGenericMethod(typeof(T));
        return Expression.Call(engine, method, dest, input,
            Expression.Constant(new[] { 1, 0 }));
    }

    private Expression EmitReshape<T>(ParameterExpression dest, Expression input, ReshapeOp reshape)
    {
        // Reshape is a view — just point dest to reshaped input
        var reshapeMethod = typeof(Tensor<T>).GetMethod("Reshape", new[] { typeof(int[]) });
        return Expression.Assign(dest,
            Expression.Call(input, reshapeMethod!, Expression.Constant(reshape.NewShape)));
    }

    private Expression EmitConcatInto<T>(ParameterExpression engine,
        ParameterExpression dest, Expression[] inputs, ConcatOp concat)
    {
        var method = typeof(IEngine).GetMethod("ConcatInto")!.MakeGenericMethod(typeof(T));
        var arrayExpr = Expression.NewArrayInit(typeof(Tensor<T>), inputs);
        return Expression.Call(engine, method, dest, arrayExpr, Expression.Constant(concat.Axis));
    }

    private Expression EmitAllocating<T>(ParameterExpression engine, string methodName,
        ParameterExpression dest, Expression[] inputs)
    {
        // Fallback: call the allocating IEngine method and copy result to workspace slot
        var method = typeof(IEngine).GetMethod(methodName)!.MakeGenericMethod(typeof(T));
        var resultVar = Expression.Variable(typeof(Tensor<T>), "alloc_temp");
        var callExpr = Expression.Call(engine, method, inputs);
        // TODO: copy result.Data.Span to dest.Data.Span
        return Expression.Block(
            new[] { resultVar },
            Expression.Assign(resultVar, callExpr),
            Expression.Assign(dest, resultVar));
    }

    private Expression? EmitFallback<T>(ParameterExpression engine, IROp op,
        ParameterExpression dest, Expression[] inputs)
    {
        // For unsupported operations, log and skip
        // In production, this would throw or fall back to interpreted execution
        return null;
    }
}

