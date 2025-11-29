using System.Linq.Expressions;
using System.Numerics;
using System.Reflection;
using AiDotNet.JitCompiler.IR;
using Operations = AiDotNet.JitCompiler.IR.Operations;

// This file uses System.Numerics.Vector<T> for SIMD operations
// Suppress ambiguity with AiDotNet.Tensors.LinearAlgebra.Vector<T>
using Vector = System.Numerics.Vector;

namespace AiDotNet.JitCompiler.CodeGen;

/// <summary>
/// Provides SIMD (Single Instruction Multiple Data) optimization for code generation.
/// </summary>
/// <remarks>
/// <para>
/// SIMD optimization allows operations to be performed on multiple data elements
/// simultaneously using vector instructions (AVX, AVX-512, NEON, etc.). This can
/// provide significant performance improvements for element-wise tensor operations.
/// </para>
/// <para><b>For Beginners:</b> SIMD makes operations much faster by processing multiple numbers at once.
///
/// Normal processing: Process one number at a time
/// - Add 1+2=3
/// - Add 4+5=9
/// - Add 7+8=15
/// (3 separate operations)
///
/// SIMD processing: Process multiple numbers together
/// - Add [1,4,7] + [2,5,8] = [3,9,15]
/// (1 operation processing 3 pairs simultaneously!)
///
/// Modern CPUs can process 4, 8, or even 16 numbers at once using SIMD.
/// This is especially powerful for AI/ML where we process huge arrays of numbers.
///
/// Example speedups:
/// - Element-wise operations: 4-8x faster
/// - Matrix operations: 2-4x faster
/// - Activation functions: 3-6x faster
/// </para>
/// </remarks>
public class SIMDOptimizer
{
    private readonly bool _enableSIMD;
    private readonly int _vectorSize;
    private readonly SIMDCapabilities _capabilities;

    /// <summary>
    /// SIMD capabilities detected on the current hardware.
    /// </summary>
    public class SIMDCapabilities
    {
        /// <summary>Whether SSE (128-bit) is available.</summary>
        public bool HasSSE { get; set; }

        /// <summary>Whether AVX (256-bit) is available.</summary>
        public bool HasAVX { get; set; }

        /// <summary>Whether AVX2 is available.</summary>
        public bool HasAVX2 { get; set; }

        /// <summary>Whether AVX-512 is available.</summary>
        public bool HasAVX512 { get; set; }

        /// <summary>Whether FMA (Fused Multiply-Add) is available.</summary>
        public bool HasFMA { get; set; }

        /// <summary>Whether ARM NEON is available.</summary>
        public bool HasNEON { get; set; }

        /// <summary>Maximum vector width in bytes.</summary>
        public int MaxVectorWidth { get; set; }
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="SIMDOptimizer"/> class.
    /// </summary>
    /// <param name="enableSIMD">Whether to enable SIMD optimizations.</param>
    public SIMDOptimizer(bool enableSIMD = true)
    {
        _enableSIMD = enableSIMD;
        _capabilities = DetectCapabilities();

        // Vector<T>.Count gives us the number of elements that fit in a SIMD register
        _vectorSize = Vector.IsHardwareAccelerated ? Vector<float>.Count : 1;
    }

    /// <summary>
    /// Gets the hardware vector width for a specific type.
    /// </summary>
    public int GetVectorWidth<T>() where T : struct
    {
        if (!_enableSIMD || !Vector.IsHardwareAccelerated)
            return 1;

        // Vector<T>.Count varies by type
        return typeof(T) switch
        {
            var t when t == typeof(float) => Vector<float>.Count,
            var t when t == typeof(double) => Vector<double>.Count,
            var t when t == typeof(int) => Vector<int>.Count,
            var t when t == typeof(long) => Vector<long>.Count,
            _ => 1
        };
    }

    /// <summary>
    /// Detects SIMD capabilities of the current hardware.
    /// </summary>
    private SIMDCapabilities DetectCapabilities()
    {
        var caps = new SIMDCapabilities();

        if (Vector.IsHardwareAccelerated)
        {
            var vectorSize = Vector<float>.Count;

            // Infer capabilities from vector size
            caps.HasSSE = vectorSize >= 4;  // 128-bit = 4 floats
            caps.HasAVX = vectorSize >= 8;  // 256-bit = 8 floats
            caps.HasAVX512 = vectorSize >= 16; // 512-bit = 16 floats
            caps.HasAVX2 = caps.HasAVX;

            // .NET's System.Numerics.Vector doesn't directly expose FMA
            // but modern CPUs with AVX2 typically have FMA
            caps.HasFMA = caps.HasAVX2;

            caps.MaxVectorWidth = vectorSize * sizeof(float);
        }

        return caps;
    }

    /// <summary>
    /// Checks if an operation should use SIMD optimization.
    /// </summary>
    public bool ShouldUseSIMD(IROp op)
    {
        if (!_enableSIMD) return false;
        if (!Vector.IsHardwareAccelerated) return false;

        // Check tensor size - must be large enough to benefit
        var totalElements = op.OutputShape.Aggregate(1, (a, b) => a * b);
        if (totalElements < _vectorSize * 4) return false;

        // Check if operation type supports SIMD
        return IsVectorizable(op);
    }

    /// <summary>
    /// Checks if an operation is vectorizable.
    /// </summary>
    private bool IsVectorizable(IROp op)
    {
        return op.OpType switch
        {
            "Add" or "Subtract" or "ElementwiseMultiply" or "Divide" => true,
            "Negate" or "Sqrt" => true,
            "ReLU" or "Sigmoid" or "Tanh" => true,
            "Exp" or "Log" => true,
            "Sum" or "Mean" or "ReduceMax" => true,
            _ => false
        };
    }

    /// <summary>
    /// Generates SIMD-optimized code for a binary operation.
    /// </summary>
    public Expression GenerateSIMDBinaryOp<T>(
        string operation,
        ParameterExpression leftInput,
        ParameterExpression rightInput,
        ParameterExpression output,
        int totalElements) where T : struct
    {
        var vectorSize = GetVectorWidth<T>();
        var numVectors = totalElements / vectorSize;
        var remainder = totalElements % vectorSize;

        var statements = new List<Expression>();

        // Generate vectorized loop
        var loopVar = Expression.Variable(typeof(int), "i");
        var vectorType = typeof(Vector<T>);

        // Loop: for (int i = 0; i < numVectors * vectorSize; i += vectorSize)
        var loopInit = Expression.Assign(loopVar, Expression.Constant(0));
        var loopCondition = Expression.LessThan(loopVar, Expression.Constant(numVectors * vectorSize));
        var loopIncrement = Expression.AddAssign(loopVar, Expression.Constant(vectorSize));

        // Get vector operation method
        var vectorOpMethod = GetVectorOperationMethod<T>(operation);

        // Vector body: Vector<T> result = op(Vector<T> left, Vector<T> right)
        var leftVector = Expression.Call(
            typeof(VectorHelper).GetMethod("LoadVector")!.MakeGenericMethod(typeof(T)),
            leftInput, loopVar);
        var rightVector = Expression.Call(
            typeof(VectorHelper).GetMethod("LoadVector")!.MakeGenericMethod(typeof(T)),
            rightInput, loopVar);

        var resultVector = vectorOpMethod != null
            ? Expression.Call(vectorOpMethod, leftVector, rightVector)
            : Expression.Add(leftVector, rightVector);

        var storeVector = Expression.Call(
            typeof(VectorHelper).GetMethod("StoreVector")!.MakeGenericMethod(typeof(T)),
            output, loopVar, resultVector);

        var loopBody = storeVector;

        // Create loop
        var breakLabel = Expression.Label();
        var loop = Expression.Loop(
            Expression.IfThenElse(
                loopCondition,
                Expression.Block(loopBody, loopIncrement),
                Expression.Break(breakLabel)),
            breakLabel);

        statements.Add(loopInit);
        statements.Add(loop);

        // Handle remainder with scalar operations
        if (remainder > 0)
        {
            var remainderStart = numVectors * vectorSize;
            for (int i = 0; i < remainder; i++)
            {
                var idx = Expression.Constant(remainderStart + i);
                var scalarOp = GenerateScalarOp<T>(operation,
                    Expression.ArrayIndex(leftInput, idx),
                    Expression.ArrayIndex(rightInput, idx));
                statements.Add(Expression.Assign(Expression.ArrayAccess(output, idx), scalarOp));
            }
        }

        return Expression.Block(new[] { loopVar }, statements);
    }

    /// <summary>
    /// Generates SIMD-optimized code for a unary operation.
    /// </summary>
    public Expression GenerateSIMDUnaryOp<T>(
        string operation,
        ParameterExpression input,
        ParameterExpression output,
        int totalElements) where T : struct
    {
        var vectorSize = GetVectorWidth<T>();
        var numVectors = totalElements / vectorSize;

        var statements = new List<Expression>();
        var loopVar = Expression.Variable(typeof(int), "i");

        // Get unary operation method
        var unaryMethod = GetUnaryVectorMethod<T>(operation);

        if (unaryMethod != null)
        {
            // Vectorized path
            var loopInit = Expression.Assign(loopVar, Expression.Constant(0));
            var loopCondition = Expression.LessThan(loopVar, Expression.Constant(numVectors * vectorSize));
            var loopIncrement = Expression.AddAssign(loopVar, Expression.Constant(vectorSize));

            var inputVector = Expression.Call(
                typeof(VectorHelper).GetMethod("LoadVector")!.MakeGenericMethod(typeof(T)),
                input, loopVar);

            var resultVector = Expression.Call(unaryMethod, inputVector);

            var storeVector = Expression.Call(
                typeof(VectorHelper).GetMethod("StoreVector")!.MakeGenericMethod(typeof(T)),
                output, loopVar, resultVector);

            var breakLabel = Expression.Label();
            var loop = Expression.Loop(
                Expression.IfThenElse(
                    loopCondition,
                    Expression.Block(storeVector, loopIncrement),
                    Expression.Break(breakLabel)),
                breakLabel);

            statements.Add(loopInit);
            statements.Add(loop);
        }

        return Expression.Block(new[] { loopVar }, statements);
    }

    /// <summary>
    /// Generates SIMD-optimized code for a reduction operation.
    /// </summary>
    public Expression GenerateSIMDReduction<T>(
        string reductionType,
        ParameterExpression input,
        int totalElements) where T : struct
    {
        var vectorSize = GetVectorWidth<T>();
        var numVectors = totalElements / vectorSize;

        var statements = new List<Expression>();
        var loopVar = Expression.Variable(typeof(int), "i");
        var accumulator = Expression.Variable(typeof(Vector<T>), "acc");

        // Initialize accumulator
        var initValue = reductionType switch
        {
            "Sum" or "Mean" => Expression.Call(typeof(Vector<T>).GetProperty("Zero")!.GetMethod!),
            "Max" => Expression.Call(typeof(VectorHelper).GetMethod("MinValue")!.MakeGenericMethod(typeof(T))),
            "Min" => Expression.Call(typeof(VectorHelper).GetMethod("MaxValue")!.MakeGenericMethod(typeof(T))),
            _ => Expression.Call(typeof(Vector<T>).GetProperty("Zero")!.GetMethod!)
        };

        statements.Add(Expression.Assign(accumulator, initValue));

        // Vectorized reduction loop
        var loopInit = Expression.Assign(loopVar, Expression.Constant(0));
        var loopCondition = Expression.LessThan(loopVar, Expression.Constant(numVectors * vectorSize));
        var loopIncrement = Expression.AddAssign(loopVar, Expression.Constant(vectorSize));

        var inputVector = Expression.Call(
            typeof(VectorHelper).GetMethod("LoadVector")!.MakeGenericMethod(typeof(T)),
            input, loopVar);

        Expression loopBody = reductionType switch
        {
            "Sum" or "Mean" => Expression.Assign(accumulator, Expression.Add(accumulator, inputVector)),
            "Max" => Expression.Assign(accumulator, Expression.Call(
                typeof(Vector).GetMethod("Max")!.MakeGenericMethod(typeof(T)),
                accumulator, inputVector)),
            "Min" => Expression.Assign(accumulator, Expression.Call(
                typeof(Vector).GetMethod("Min")!.MakeGenericMethod(typeof(T)),
                accumulator, inputVector)),
            _ => Expression.Assign(accumulator, Expression.Add(accumulator, inputVector))
        };

        var breakLabel = Expression.Label();
        var loop = Expression.Loop(
            Expression.IfThenElse(
                loopCondition,
                Expression.Block(loopBody, loopIncrement),
                Expression.Break(breakLabel)),
            breakLabel);

        statements.Add(loopInit);
        statements.Add(loop);

        // Horizontal reduction to get final scalar
        var result = Expression.Variable(typeof(T), "result");
        var horizontalReduce = Expression.Call(
            typeof(VectorHelper).GetMethod($"HorizontalReduce{reductionType}")!.MakeGenericMethod(typeof(T)),
            accumulator);

        statements.Add(Expression.Assign(result, horizontalReduce));

        return Expression.Block(new[] { loopVar, accumulator, result }, statements);
    }

    /// <summary>
    /// Gets the vector operation method for binary operations.
    /// </summary>
    private MethodInfo? GetVectorOperationMethod<T>(string operation) where T : struct
    {
        return operation switch
        {
            "Add" => typeof(Vector).GetMethod("Add")?.MakeGenericMethod(typeof(T)),
            "Subtract" => typeof(Vector).GetMethod("Subtract")?.MakeGenericMethod(typeof(T)),
            "Multiply" => typeof(Vector).GetMethod("Multiply", new[] { typeof(Vector<T>), typeof(Vector<T>) })?.MakeGenericMethod(typeof(T)),
            "Divide" => typeof(Vector).GetMethod("Divide")?.MakeGenericMethod(typeof(T)),
            _ => null
        };
    }

    /// <summary>
    /// Gets the vector method for unary operations.
    /// </summary>
    private MethodInfo? GetUnaryVectorMethod<T>(string operation) where T : struct
    {
        return operation switch
        {
            "Negate" => typeof(Vector).GetMethod("Negate")?.MakeGenericMethod(typeof(T)),
            "Sqrt" => typeof(Vector).GetMethod("SquareRoot")?.MakeGenericMethod(typeof(T)),
            "Abs" => typeof(Vector).GetMethod("Abs")?.MakeGenericMethod(typeof(T)),
            _ => null
        };
    }

    /// <summary>
    /// Generates a scalar operation expression.
    /// </summary>
    private Expression GenerateScalarOp<T>(string operation, Expression left, Expression right)
    {
        return operation switch
        {
            "Add" => Expression.Add(left, right),
            "Subtract" => Expression.Subtract(left, right),
            "Multiply" => Expression.Multiply(left, right),
            "Divide" => Expression.Divide(left, right),
            _ => Expression.Add(left, right)
        };
    }

    /// <summary>
    /// Gets SIMD capabilities.
    /// </summary>
    public SIMDCapabilities Capabilities => _capabilities;

    /// <summary>
    /// Gets optimization statistics for a graph.
    /// </summary>
    public SIMDStats GetStats(IRGraph graph)
    {
        var stats = new SIMDStats
        {
            TotalOperations = graph.Operations.Count,
            VectorizableOperations = graph.Operations.Count(ShouldUseSIMD),
            VectorSize = _vectorSize,
            HardwareAccelerated = Vector.IsHardwareAccelerated
        };

        return stats;
    }
}

/// <summary>
/// Statistics about SIMD optimization opportunities.
/// </summary>
public class SIMDStats
{
    /// <summary>Total number of operations in the graph.</summary>
    public int TotalOperations { get; set; }

    /// <summary>Number of operations that can be vectorized.</summary>
    public int VectorizableOperations { get; set; }

    /// <summary>Size of SIMD vectors on this hardware.</summary>
    public int VectorSize { get; set; }

    /// <summary>Whether hardware acceleration is available.</summary>
    public bool HardwareAccelerated { get; set; }

    /// <summary>Estimated speedup from vectorization.</summary>
    public double EstimatedSpeedup
    {
        get
        {
            if (!HardwareAccelerated || TotalOperations == 0)
                return 1.0;

            var vectorizableRatio = (double)VectorizableOperations / TotalOperations;
            var perOpSpeedup = VectorSize * 0.75; // Account for overhead
            return 1.0 + (vectorizableRatio * (perOpSpeedup - 1.0));
        }
    }

    public override string ToString()
    {
        return $"SIMD Stats: {VectorizableOperations}/{TotalOperations} operations vectorizable, " +
               $"Vector size: {VectorSize}, " +
               $"Estimated speedup: {EstimatedSpeedup:F2}x";
    }
}

/// <summary>
/// Helper methods for SIMD operations.
/// </summary>
public static class VectorHelper
{
    /// <summary>
    /// Loads a vector from an array at the specified offset.
    /// </summary>
    public static System.Numerics.Vector<T> LoadVector<T>(T[] array, int offset) where T : struct
    {
        return new System.Numerics.Vector<T>(array, offset);
    }

    /// <summary>
    /// Stores a vector to an array at the specified offset.
    /// </summary>
    public static void StoreVector<T>(T[] array, int offset, System.Numerics.Vector<T> vector) where T : struct
    {
        vector.CopyTo(array, offset);
    }

    /// <summary>
    /// Creates a vector with all elements set to the minimum value.
    /// </summary>
    public static System.Numerics.Vector<T> MinValue<T>() where T : struct
    {
        // Use reflection to get MinValue for the type
        var minValue = typeof(T).GetField("MinValue")?.GetValue(null);
        if (minValue != null)
        {
            return new System.Numerics.Vector<T>((T)minValue);
        }
        return System.Numerics.Vector<T>.Zero;
    }

    /// <summary>
    /// Creates a vector with all elements set to the maximum value.
    /// </summary>
    public static System.Numerics.Vector<T> MaxValue<T>() where T : struct
    {
        var maxValue = typeof(T).GetField("MaxValue")?.GetValue(null);
        if (maxValue != null)
        {
            return new System.Numerics.Vector<T>((T)maxValue);
        }
        return System.Numerics.Vector<T>.Zero;
    }

    /// <summary>
    /// Performs horizontal sum reduction on a vector.
    /// </summary>
    public static T HorizontalReduceSum<T>(System.Numerics.Vector<T> vector) where T : struct
    {
        var array = new T[System.Numerics.Vector<T>.Count];
        vector.CopyTo(array);

        dynamic sum = default(T)!;
        foreach (var val in array)
        {
            sum = sum + (dynamic)val;
        }
        return sum;
    }

    /// <summary>
    /// Performs horizontal max reduction on a vector.
    /// </summary>
    public static T HorizontalReduceMax<T>(System.Numerics.Vector<T> vector) where T : struct
    {
        var array = new T[System.Numerics.Vector<T>.Count];
        vector.CopyTo(array);

        dynamic max = array[0];
        for (int i = 1; i < array.Length; i++)
        {
            if ((dynamic)array[i] > max)
                max = array[i];
        }
        return max;
    }

    /// <summary>
    /// Performs horizontal min reduction on a vector.
    /// </summary>
    public static T HorizontalReduceMin<T>(System.Numerics.Vector<T> vector) where T : struct
    {
        var array = new T[System.Numerics.Vector<T>.Count];
        vector.CopyTo(array);

        dynamic min = array[0];
        for (int i = 1; i < array.Length; i++)
        {
            if ((dynamic)array[i] < min)
                min = array[i];
        }
        return min;
    }

    /// <summary>
    /// Performs horizontal mean reduction on a vector.
    /// </summary>
    public static T HorizontalReduceMean<T>(System.Numerics.Vector<T> vector) where T : struct
    {
        var sum = HorizontalReduceSum(vector);
        return (T)(object)((dynamic)sum / System.Numerics.Vector<T>.Count);
    }

    /// <summary>
    /// Applies ReLU activation to a vector.
    /// </summary>
    public static System.Numerics.Vector<T> VectorReLU<T>(System.Numerics.Vector<T> input) where T : struct
    {
        return Vector.Max(input, System.Numerics.Vector<T>.Zero);
    }

    /// <summary>
    /// Applies element-wise comparison for ReLU gradient.
    /// </summary>
    public static System.Numerics.Vector<T> VectorReLUGrad<T>(System.Numerics.Vector<T> gradOutput, System.Numerics.Vector<T> forwardInput) where T : struct
    {
        var mask = Vector.GreaterThan(forwardInput, System.Numerics.Vector<T>.Zero);
        return Vector.ConditionalSelect(mask, gradOutput, System.Numerics.Vector<T>.Zero);
    }
}
