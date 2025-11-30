using System.Linq.Expressions;
using System.Reflection;
using AiDotNet.JitCompiler.IR;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

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
/// <para>
/// This class uses the <see cref="INumericOperations{T}"/> interface for type-safe
/// arithmetic operations and leverages TensorPrimitives for hardware-accelerated
/// SIMD computations when available.
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
    /// Initializes a new instance of the <see cref="SIMDOptimizer"/> class.
    /// </summary>
    /// <param name="enableSIMD">Whether to enable SIMD optimizations.</param>
    public SIMDOptimizer(bool enableSIMD = true)
    {
        _enableSIMD = enableSIMD;
        _capabilities = SIMDCapabilities.Detect();

        // Get the number of float elements that fit in a SIMD register
        _vectorSize = _capabilities.IsHardwareAccelerated
            ? _capabilities.GetVectorCount(sizeof(float))
            : 1;
    }

    /// <summary>
    /// Gets the SIMD capabilities detected on the current hardware.
    /// </summary>
    public SIMDCapabilities Capabilities => _capabilities;

    /// <summary>
    /// Gets whether SIMD optimization is enabled and hardware-accelerated.
    /// </summary>
    public bool IsEnabled => _enableSIMD && _capabilities.IsHardwareAccelerated;

    /// <summary>
    /// Gets the hardware vector width for a specific type.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <returns>The number of elements that fit in a SIMD register, or 1 if SIMD is not available.</returns>
    public int GetVectorWidth<T>()
    {
        if (!_enableSIMD || !_capabilities.IsHardwareAccelerated)
            return 1;

        // Determine vector width based on type size
        var typeSize = GetTypeSize<T>();
        return typeSize > 0 ? _capabilities.GetVectorCount(typeSize) : 1;
    }

    /// <summary>
    /// Gets the size in bytes of a numeric type.
    /// </summary>
    private static int GetTypeSize<T>()
    {
        return typeof(T) switch
        {
            var t when t == typeof(float) => sizeof(float),
            var t when t == typeof(double) => sizeof(double),
            var t when t == typeof(int) => sizeof(int),
            var t when t == typeof(long) => sizeof(long),
            var t when t == typeof(Half) => 2,
            var t when t == typeof(short) => sizeof(short),
            var t when t == typeof(byte) => sizeof(byte),
            _ => 0
        };
    }

    /// <summary>
    /// Checks if an operation should use SIMD optimization.
    /// </summary>
    /// <param name="op">The IR operation to check.</param>
    /// <returns>True if SIMD optimization should be used; otherwise, false.</returns>
    public bool ShouldUseSIMD(IROp op)
    {
        if (!_enableSIMD) return false;
        if (!_capabilities.IsHardwareAccelerated) return false;

        // Check tensor size - must be large enough to benefit
        var totalElements = op.OutputShape.Aggregate(1, (a, b) => a * b);
        if (totalElements < _vectorSize * 4) return false;

        // Check if operation type supports SIMD
        return IsVectorizable(op);
    }

    /// <summary>
    /// Checks if an operation is vectorizable.
    /// </summary>
    /// <param name="op">The IR operation to check.</param>
    /// <returns>True if the operation can be vectorized; otherwise, false.</returns>
    private static bool IsVectorizable(IROp op)
    {
        return op.OpType switch
        {
            "Add" or "Subtract" or "ElementwiseMultiply" or "Divide" => true,
            "Negate" or "Sqrt" or "Abs" => true,
            "ReLU" or "Sigmoid" or "Tanh" => true,
            "Exp" or "Log" or "Log2" => true,
            "Sum" or "Mean" or "ReduceMax" or "ReduceMin" => true,
            "Dot" or "CosineSimilarity" => true,
            _ => false
        };
    }

    /// <summary>
    /// Generates SIMD-optimized code for a binary operation.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="operation">The operation name (Add, Subtract, Multiply, Divide).</param>
    /// <param name="leftInput">The left input array parameter.</param>
    /// <param name="rightInput">The right input array parameter.</param>
    /// <param name="output">The output array parameter.</param>
    /// <param name="totalElements">The total number of elements to process.</param>
    /// <returns>An expression that performs the vectorized binary operation.</returns>
    public Expression GenerateSIMDBinaryOp<T>(
        string operation,
        ParameterExpression leftInput,
        ParameterExpression rightInput,
        ParameterExpression output,
        int totalElements)
    {
        // Find the array-based overload of the binary operation method
        var methodName = operation switch
        {
            "Add" => nameof(VectorHelper.AddArrays),
            "Subtract" => nameof(VectorHelper.SubtractArrays),
            "Multiply" => nameof(VectorHelper.MultiplyArrays),
            "Divide" => nameof(VectorHelper.DivideArrays),
            _ => throw new ArgumentException($"Unsupported binary operation: {operation}", nameof(operation))
        };

        var helperMethod = typeof(VectorHelper)
            .GetMethods()
            .First(m => m.Name == methodName && m.IsGenericMethod)
            .MakeGenericMethod(typeof(T));

        // Generate: VectorHelper.OperationArrays<T>(leftInput, rightInput, output)
        return Expression.Call(null, helperMethod, leftInput, rightInput, output);
    }

    /// <summary>
    /// Generates SIMD-optimized code for a unary operation.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="operation">The operation name (ReLU, Sigmoid, Tanh, Exp, Log).</param>
    /// <param name="input">The input array parameter.</param>
    /// <param name="output">The output array parameter.</param>
    /// <param name="totalElements">The total number of elements to process.</param>
    /// <returns>An expression that performs the vectorized unary operation.</returns>
    public Expression GenerateSIMDUnaryOp<T>(
        string operation,
        ParameterExpression input,
        ParameterExpression output,
        int totalElements)
    {
        // Get the appropriate VectorHelper method for the operation (array-based)
        var methodName = operation switch
        {
            "ReLU" => nameof(VectorHelper.ApplyReLUArrays),
            "Sigmoid" => nameof(VectorHelper.ApplySigmoidArrays),
            "Tanh" => nameof(VectorHelper.ApplyTanhArrays),
            "Exp" => nameof(VectorHelper.ApplyExpArrays),
            "Log" => nameof(VectorHelper.ApplyLogArrays),
            "SoftMax" => nameof(VectorHelper.ApplySoftMaxArrays),
            _ => throw new ArgumentException($"Unsupported unary operation: {operation}", nameof(operation))
        };

        var helperMethod = typeof(VectorHelper)
            .GetMethods()
            .First(m => m.Name == methodName && m.IsGenericMethod)
            .MakeGenericMethod(typeof(T));

        // Generate: VectorHelper.OperationArrays<T>(input, output)
        return Expression.Call(null, helperMethod, input, output);
    }

    /// <summary>
    /// Generates SIMD-optimized code for a reduction operation.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="reductionType">The reduction type (Sum, Mean, Max, Min).</param>
    /// <param name="input">The input array parameter.</param>
    /// <param name="totalElements">The total number of elements.</param>
    /// <returns>An expression that performs the vectorized reduction and returns the result.</returns>
    public Expression GenerateSIMDReduction<T>(
        string reductionType,
        ParameterExpression input,
        int totalElements)
    {
        // Get the appropriate VectorHelper method for the reduction (array overload)
        var methodName = reductionType switch
        {
            "Sum" => nameof(VectorHelper.HorizontalReduceSumArray),
            "Mean" => nameof(VectorHelper.HorizontalReduceMeanArray),
            "Max" or "ReduceMax" => nameof(VectorHelper.HorizontalReduceMaxArray),
            "Min" or "ReduceMin" => nameof(VectorHelper.HorizontalReduceMinArray),
            _ => throw new ArgumentException($"Unsupported reduction operation: {reductionType}", nameof(reductionType))
        };

        var helperMethod = typeof(VectorHelper)
            .GetMethods()
            .First(m => m.Name == methodName && m.IsGenericMethod)
            .MakeGenericMethod(typeof(T));

        return Expression.Call(null, helperMethod, input);
    }

    /// <summary>
    /// Generates an expression to compute the dot product of two arrays.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <param name="leftInput">The left input array parameter.</param>
    /// <param name="rightInput">The right input array parameter.</param>
    /// <returns>An expression that computes the dot product.</returns>
    public Expression GenerateDotProduct<T>(
        ParameterExpression leftInput,
        ParameterExpression rightInput)
    {
        var helperMethod = typeof(VectorHelper)
            .GetMethods()
            .First(m => m.Name == nameof(VectorHelper.DotArray) && m.IsGenericMethod)
            .MakeGenericMethod(typeof(T));

        return Expression.Call(null, helperMethod, leftInput, rightInput);
    }

    /// <summary>
    /// Gets optimization statistics for a graph.
    /// </summary>
    /// <param name="graph">The IR graph to analyze.</param>
    /// <returns>Statistics about SIMD optimization opportunities in the graph.</returns>
    public SIMDStats GetStats(IRGraph graph)
    {
        var stats = new SIMDStats
        {
            TotalOperations = graph.Operations.Count,
            VectorizableOperations = graph.Operations.Count(ShouldUseSIMD),
            VectorSize = _vectorSize,
            HardwareAccelerated = _capabilities.IsHardwareAccelerated
        };

        return stats;
    }

    /// <summary>
    /// Gets the numeric operations implementation for a type.
    /// </summary>
    /// <typeparam name="T">The numeric type.</typeparam>
    /// <returns>The INumericOperations implementation.</returns>
    public static INumericOperations<T> GetOperations<T>()
    {
        return MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Checks if a type supports SIMD acceleration.
    /// </summary>
    /// <typeparam name="T">The numeric type to check.</typeparam>
    /// <returns>True if the type supports SIMD acceleration; otherwise, false.</returns>
    public static bool SupportsSIMD<T>()
    {
        return MathHelper.SupportsCpuAcceleration<T>();
    }
}
