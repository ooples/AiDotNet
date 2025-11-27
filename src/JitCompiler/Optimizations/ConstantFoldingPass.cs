using AiDotNet.JitCompiler.IR;
using AiDotNet.JitCompiler.IR.Operations;

namespace AiDotNet.JitCompiler.Optimizations;

/// <summary>
/// Optimization pass that evaluates constant expressions at compile time.
/// </summary>
/// <remarks>
/// <para>
/// Constant folding is a compiler optimization that evaluates expressions with
/// constant inputs during compilation rather than at runtime. This reduces the
/// number of operations that need to be executed and can significantly improve
/// performance for graphs with many constant operations.
/// </para>
/// <para><b>For Beginners:</b> This optimization pre-computes results that never change.
///
/// Think of it like simplifying math:
/// - Original: x = 2 + 3, y = x * 4
/// - Optimized: x = 5, y = x * 4 (we computed 2 + 3 ahead of time)
/// - Even better: y = 20 (if x is only used here)
///
/// Why this helps:
/// - Fewer operations to execute at runtime
/// - Less memory needed for intermediate results
/// - Can enable other optimizations (if everything becomes constant)
///
/// Example in neural networks:
/// - If you have weight_scaled = weight * scale_factor
/// - And both weight and scale_factor are constants
/// - We can compute weight_scaled once at compile time
/// - Runtime just uses the pre-computed value
///
/// This is especially useful for operations on model architecture parameters
/// that don't change during inference.
/// </para>
/// </remarks>
public class ConstantFoldingPass : IOptimizationPass
{
    /// <summary>
    /// Gets the name of this optimization pass.
    /// </summary>
    public string Name => "Constant Folding";

    /// <summary>
    /// Configuration for constant folding behavior.
    /// </summary>
    public class FoldingConfig
    {
        /// <summary>Maximum tensor size to fold (in elements). Larger tensors are skipped.</summary>
        public int MaxTensorSizeToFold { get; set; } = 10000;

        /// <summary>Whether to fold expensive operations like MatMul.</summary>
        public bool FoldExpensiveOps { get; set; } = true;

        /// <summary>Whether to propagate constants through the graph.</summary>
        public bool PropagateConstants { get; set; } = true;
    }

    private readonly FoldingConfig _config;

    /// <summary>
    /// Initializes a new instance with default configuration.
    /// </summary>
    public ConstantFoldingPass() : this(new FoldingConfig()) { }

    /// <summary>
    /// Initializes a new instance with custom configuration.
    /// </summary>
    public ConstantFoldingPass(FoldingConfig config)
    {
        _config = config;
    }

    /// <summary>
    /// Applies constant folding optimization to an IR graph.
    /// </summary>
    /// <param name="graph">The IR graph to optimize.</param>
    /// <returns>An optimized IR graph with constant expressions folded.</returns>
    public IRGraph Optimize(IRGraph graph)
    {
        // Track which tensors are constants and their values
        var constantTensors = new HashSet<int>();
        var constantValues = new Dictionary<int, double[]>();

        // First pass: identify existing ConstantOp operations
        foreach (var op in graph.Operations)
        {
            if (op is ConstantOp constOp)
            {
                constantTensors.Add(constOp.OutputId);
                constantValues[constOp.OutputId] = constOp.Values;
            }
            else if (op is ScalarConstantOp scalarOp)
            {
                constantTensors.Add(scalarOp.OutputId);
                constantValues[scalarOp.OutputId] = new[] { scalarOp.Value };
            }
        }

        // Build a new optimized graph
        var optimizedOps = new List<IROp>();
        int foldedCount = 0;

        // Process each operation
        foreach (var op in graph.Operations)
        {
            // Skip already-constant operations
            if (op is ConstantOp or ScalarConstantOp)
            {
                optimizedOps.Add(op);
                continue;
            }

            // Check if all inputs to this operation are constants
            bool allInputsConstant = op.InputIds.All(id => constantTensors.Contains(id));

            if (allInputsConstant && CanFold(op) && ShouldFold(op, constantValues))
            {
                // This operation can be folded - evaluate it at compile time
                var result = EvaluateOperation(op, constantValues);

                if (result != null)
                {
                    // Create a ConstantOp with the computed result
                    var constantOp = new ConstantOp
                    {
                        OutputId = op.OutputId,
                        InputIds = Array.Empty<int>(),
                        OutputType = op.OutputType,
                        OutputShape = op.OutputShape,
                        Values = result
                    };

                    optimizedOps.Add(constantOp);

                    // Mark output as constant for downstream operations
                    constantTensors.Add(op.OutputId);
                    constantValues[op.OutputId] = result;
                    foldedCount++;
                }
                else
                {
                    // Evaluation failed, keep original operation
                    optimizedOps.Add(op);
                }
            }
            else
            {
                // Cannot fold this operation, keep it as-is
                optimizedOps.Add(op);
            }
        }

        // Create optimized graph
        var optimizedGraph = new IRGraph
        {
            InputIds = new List<int>(graph.InputIds),
            OutputIds = new List<int>(graph.OutputIds),
            Operations = optimizedOps,
            TensorShapes = new Dictionary<int, int[]>(graph.TensorShapes),
            Metadata = new Dictionary<string, object>(graph.Metadata)
        };

        // Add folding metadata
        optimizedGraph.Metadata["ConstantFolding_FoldedOps"] = foldedCount;
        optimizedGraph.Metadata["ConstantFolding_ConstantTensors"] = constantTensors.Count;

        return optimizedGraph;
    }

    /// <summary>
    /// Determines if an operation can be constant-folded.
    /// </summary>
    private bool CanFold(IROp op)
    {
        return op switch
        {
            // Arithmetic operations - always foldable
            AddOp => true,
            SubtractOp => true,
            ElementwiseMultiplyOp => true,
            DivideOp => true,
            PowerOp => true,
            NegateOp => true,

            // Math operations - always foldable
            ExpOp => true,
            LogOp => true,
            SqrtOp => true,

            // Activations - always foldable
            ReLUOp => true,
            SigmoidOp => true,
            TanhOp => true,
            SoftmaxOp => true,

            // Matrix operations - foldable (expensive but allowed)
            MatMulOp => _config.FoldExpensiveOps,
            TransposeOp => true,

            // Reduction operations - foldable
            SumOp => true,
            MeanOp => true,
            ReduceMaxOp => true,
            ReduceMeanOp => true,

            // Shape operations - foldable
            ReshapeOp => true,
            ConcatOp => true,

            // Default: be conservative
            _ => false
        };
    }

    /// <summary>
    /// Determines if we should actually fold this operation (size check).
    /// </summary>
    private bool ShouldFold(IROp op, Dictionary<int, double[]> constantValues)
    {
        // Check output size
        var outputSize = op.OutputShape.Length == 0 ? 1 : op.OutputShape.Aggregate(1, (a, b) => a * b);
        if (outputSize > _config.MaxTensorSizeToFold)
            return false;

        // Check input sizes
        foreach (var inputId in op.InputIds)
        {
            if (constantValues.TryGetValue(inputId, out var values) && values.Length > _config.MaxTensorSizeToFold)
                return false;
        }

        return true;
    }

    /// <summary>
    /// Evaluates an operation with constant inputs.
    /// </summary>
    private double[]? EvaluateOperation(IROp op, Dictionary<int, double[]> constantValues)
    {
        try
        {
            return op switch
            {
                // Binary arithmetic operations
                AddOp => EvaluateBinaryElementwise(op, constantValues, (a, b) => a + b),
                SubtractOp => EvaluateBinaryElementwise(op, constantValues, (a, b) => a - b),
                ElementwiseMultiplyOp => EvaluateBinaryElementwise(op, constantValues, (a, b) => a * b),
                DivideOp => EvaluateBinaryElementwise(op, constantValues, (a, b) => b != 0 ? a / b : double.NaN),

                // Unary operations
                NegateOp => EvaluateUnary(op, constantValues, x => -x),
                ExpOp => EvaluateUnary(op, constantValues, Math.Exp),
                LogOp => EvaluateUnary(op, constantValues, x => x > 0 ? Math.Log(x) : double.NaN),
                SqrtOp => EvaluateUnary(op, constantValues, x => x >= 0 ? Math.Sqrt(x) : double.NaN),

                // Activations
                ReLUOp => EvaluateUnary(op, constantValues, x => Math.Max(0, x)),
                SigmoidOp => EvaluateUnary(op, constantValues, x => 1.0 / (1.0 + Math.Exp(-x))),
                TanhOp => EvaluateUnary(op, constantValues, Math.Tanh),
                SoftmaxOp => EvaluateSoftmax(op, constantValues),

                // Power
                PowerOp powerOp => EvaluateUnary(op, constantValues, x => Math.Pow(x, powerOp.Exponent)),

                // Matrix operations
                MatMulOp => EvaluateMatMul(op, constantValues),
                TransposeOp => EvaluateTranspose(op, constantValues),

                // Reductions
                SumOp sumOp => EvaluateSum(op, constantValues, sumOp.Axes, sumOp.KeepDims),
                MeanOp => EvaluateMean(op, constantValues),
                ReduceMaxOp reduceMaxOp => EvaluateReduceMax(op, constantValues, reduceMaxOp.Axes),
                ReduceMeanOp reduceMeanOp => EvaluateReduceMean(op, constantValues, reduceMeanOp.Axes, reduceMeanOp.KeepDims),

                // Shape operations
                ReshapeOp => EvaluateReshape(op, constantValues),
                ConcatOp concatOp => EvaluateConcat(op, constantValues, concatOp.Axis),

                _ => null
            };
        }
        catch
        {
            // If evaluation fails for any reason, return null to keep the original op
            return null;
        }
    }

    /// <summary>
    /// Evaluates a binary element-wise operation.
    /// </summary>
    private double[]? EvaluateBinaryElementwise(IROp op, Dictionary<int, double[]> constantValues, Func<double, double, double> operation)
    {
        if (op.InputIds.Length != 2) return null;

        if (!constantValues.TryGetValue(op.InputIds[0], out var a)) return null;
        if (!constantValues.TryGetValue(op.InputIds[1], out var b)) return null;

        // Handle broadcasting
        var outputSize = op.OutputShape.Length == 0 ? 1 : op.OutputShape.Aggregate(1, (a, b) => a * b);
        var result = new double[outputSize];

        if (a.Length == b.Length)
        {
            // Same size - simple element-wise
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = operation(a[i], b[i]);
            }
        }
        else if (a.Length == 1)
        {
            // Scalar broadcast
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = operation(a[0], b[i]);
            }
        }
        else if (b.Length == 1)
        {
            // Scalar broadcast
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = operation(a[i], b[0]);
            }
        }
        else
        {
            // Complex broadcasting - need to match shapes
            result = EvaluateBroadcastBinary(a, b, op.OutputShape, operation);
            if (result == null) return null;
        }

        return result;
    }

    /// <summary>
    /// Evaluates a binary operation with broadcasting.
    /// </summary>
    private double[]? EvaluateBroadcastBinary(double[] a, double[] b, int[] outputShape, Func<double, double, double> operation)
    {
        var outputSize = outputShape.Aggregate(1, (x, y) => x * y);
        var result = new double[outputSize];

        // Simple case: one is a multiple of the other
        if (outputSize == a.Length && a.Length % b.Length == 0)
        {
            for (int i = 0; i < outputSize; i++)
            {
                result[i] = operation(a[i], b[i % b.Length]);
            }
        }
        else if (outputSize == b.Length && b.Length % a.Length == 0)
        {
            for (int i = 0; i < outputSize; i++)
            {
                result[i] = operation(a[i % a.Length], b[i]);
            }
        }
        else
        {
            // Cannot handle this broadcasting case
            return null;
        }

        return result;
    }

    /// <summary>
    /// Evaluates a unary operation.
    /// </summary>
    private double[]? EvaluateUnary(IROp op, Dictionary<int, double[]> constantValues, Func<double, double> operation)
    {
        if (op.InputIds.Length != 1) return null;
        if (!constantValues.TryGetValue(op.InputIds[0], out var input)) return null;

        var result = new double[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = operation(input[i]);
        }

        return result;
    }

    /// <summary>
    /// Evaluates softmax operation.
    /// </summary>
    private double[]? EvaluateSoftmax(IROp op, Dictionary<int, double[]> constantValues)
    {
        if (op.InputIds.Length != 1) return null;
        if (!constantValues.TryGetValue(op.InputIds[0], out var input)) return null;

        var result = new double[input.Length];

        // Compute max for numerical stability
        double max = input.Max();

        // Compute exp(x - max) and sum
        double sum = 0;
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = Math.Exp(input[i] - max);
            sum += result[i];
        }

        // Normalize
        for (int i = 0; i < result.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }

    /// <summary>
    /// Evaluates matrix multiplication.
    /// </summary>
    private double[]? EvaluateMatMul(IROp op, Dictionary<int, double[]> constantValues)
    {
        if (op.InputIds.Length != 2) return null;
        if (!constantValues.TryGetValue(op.InputIds[0], out var a)) return null;
        if (!constantValues.TryGetValue(op.InputIds[1], out var b)) return null;

        // For simplicity, handle 2D matrices only
        if (op.OutputShape.Length != 2) return null;

        var m = op.OutputShape[0];
        var n = op.OutputShape[1];

        // Infer k from input sizes
        var k = a.Length / m;
        if (k * n != b.Length) return null;

        var result = new double[m * n];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int l = 0; l < k; l++)
                {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        return result;
    }

    /// <summary>
    /// Evaluates transpose operation.
    /// </summary>
    private double[]? EvaluateTranspose(IROp op, Dictionary<int, double[]> constantValues)
    {
        if (op.InputIds.Length != 1) return null;
        if (!constantValues.TryGetValue(op.InputIds[0], out var input)) return null;

        // Handle 2D transpose
        if (op.OutputShape.Length != 2) return null;

        var rows = op.OutputShape[0];
        var cols = op.OutputShape[1];

        var result = new double[input.Length];

        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                result[j * cols + i] = input[i * rows + j];
            }
        }

        return result;
    }

    /// <summary>
    /// Evaluates sum reduction.
    /// </summary>
    private double[]? EvaluateSum(IROp op, Dictionary<int, double[]> constantValues, int[]? axes, bool keepDims)
    {
        if (op.InputIds.Length != 1) return null;
        if (!constantValues.TryGetValue(op.InputIds[0], out var input)) return null;

        // Simple case: sum all elements
        if (axes == null || axes.Length == 0)
        {
            return new[] { input.Sum() };
        }

        // For now, handle simple case of reducing to scalar or single axis
        var outputSize = op.OutputShape.Length == 0 ? 1 : op.OutputShape.Aggregate(1, (a, b) => a * b);

        if (outputSize == 1)
        {
            return new[] { input.Sum() };
        }

        // More complex reduction - return null to skip folding
        return null;
    }

    /// <summary>
    /// Evaluates mean reduction.
    /// </summary>
    private double[]? EvaluateMean(IROp op, Dictionary<int, double[]> constantValues)
    {
        if (op.InputIds.Length != 1) return null;
        if (!constantValues.TryGetValue(op.InputIds[0], out var input)) return null;

        if (input.Length == 0) return null;

        return new[] { input.Average() };
    }

    /// <summary>
    /// Evaluates max reduction.
    /// </summary>
    private double[]? EvaluateReduceMax(IROp op, Dictionary<int, double[]> constantValues, int[]? axes)
    {
        if (op.InputIds.Length != 1) return null;
        if (!constantValues.TryGetValue(op.InputIds[0], out var input)) return null;

        if (input.Length == 0) return null;

        // Simple case: max of all elements
        if (axes == null || axes.Length == 0)
        {
            return new[] { input.Max() };
        }

        // More complex reduction - return null to skip folding
        return null;
    }

    /// <summary>
    /// Evaluates mean reduction along axes.
    /// </summary>
    private double[]? EvaluateReduceMean(IROp op, Dictionary<int, double[]> constantValues, int[]? axes, bool keepDims)
    {
        if (op.InputIds.Length != 1) return null;
        if (!constantValues.TryGetValue(op.InputIds[0], out var input)) return null;

        if (input.Length == 0) return null;

        // Simple case: mean of all elements
        var outputSize = op.OutputShape.Length == 0 ? 1 : op.OutputShape.Aggregate(1, (a, b) => a * b);

        if (outputSize == 1)
        {
            return new[] { input.Average() };
        }

        // More complex reduction - return null to skip folding
        return null;
    }

    /// <summary>
    /// Evaluates reshape operation.
    /// </summary>
    private double[]? EvaluateReshape(IROp op, Dictionary<int, double[]> constantValues)
    {
        if (op.InputIds.Length != 1) return null;
        if (!constantValues.TryGetValue(op.InputIds[0], out var input)) return null;

        // Reshape just returns the same data (element order is preserved)
        return input.ToArray();
    }

    /// <summary>
    /// Evaluates concatenation operation.
    /// </summary>
    private double[]? EvaluateConcat(IROp op, Dictionary<int, double[]> constantValues, int axis)
    {
        if (op.InputIds.Length < 2) return null;

        var inputs = new List<double[]>();
        foreach (var inputId in op.InputIds)
        {
            if (!constantValues.TryGetValue(inputId, out var values)) return null;
            inputs.Add(values);
        }

        // Simple case: 1D concat or concat along last axis of equal-sized tensors
        var totalSize = inputs.Sum(i => i.Length);
        var result = new double[totalSize];

        int offset = 0;
        foreach (var input in inputs)
        {
            Array.Copy(input, 0, result, offset, input.Length);
            offset += input.Length;
        }

        return result;
    }
}
