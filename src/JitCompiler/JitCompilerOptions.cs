namespace AiDotNet.JitCompiler;

/// <summary>
/// Configuration options for the JIT compiler.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Settings to control how the JIT compiler works.
///
/// You can:
/// - Enable/disable specific optimizations
/// - Turn caching on/off
/// - Configure compilation behavior
/// - Control how unsupported operations are handled
///
/// For most users, the defaults work great!
/// </para>
/// </remarks>
public class JitCompilerOptions
{
    /// <summary>
    /// Gets or sets a value indicating whether to enable constant folding optimization.
    /// Default: true.
    /// </summary>
    public bool EnableConstantFolding { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable dead code elimination.
    /// Default: true.
    /// </summary>
    public bool EnableDeadCodeElimination { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable operation fusion.
    /// Default: true.
    /// </summary>
    public bool EnableOperationFusion { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable caching of compiled graphs.
    /// Default: true.
    /// </summary>
    public bool EnableCaching { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable loop unrolling optimization.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Loop unrolling improves performance for small, fixed-size loops by eliminating
    /// loop overhead and enabling better instruction pipelining. The optimizer automatically
    /// determines which loops benefit from unrolling based on tensor size and operation type.
    /// </para>
    /// </remarks>
    public bool EnableLoopUnrolling { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable adaptive fusion strategies.
    /// Default: false (currently uses standard fusion when enabled).
    /// </summary>
    /// <remarks>
    /// <para><b>Status:</b> Architecture implemented, delegates to standard fusion.
    /// Adaptive fusion will intelligently select which operations to fuse based on
    /// graph structure, tensor sizes, and hardware characteristics.
    /// </para>
    /// </remarks>
    public bool EnableAdaptiveFusion { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether to enable auto-tuning of optimizations.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Auto-tuning automatically determines the best optimization configuration for
    /// each graph based on graph analysis, tensor sizes, and operation types. It selects
    /// the optimal combination of fusion, unrolling, and vectorization strategies.
    /// </para>
    /// </remarks>
    public bool EnableAutoTuning { get; set; } = true;

    /// <summary>
    /// Gets or sets a value indicating whether to enable SIMD vectorization hints.
    /// Default: false (not yet fully implemented).
    /// </summary>
    /// <remarks>
    /// <para><b>Status:</b> Architecture planned, implementation pending.
    /// SIMD hints guide the code generator to use vector instructions (AVX, AVX-512)
    /// for better performance on element-wise operations.
    /// </para>
    /// </remarks>
    public bool EnableSIMDHints { get; set; } = false;

    /// <summary>
    /// Gets or sets a value indicating whether to enable memory pooling for tensors.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Reuses tensor memory to reduce allocations.
    ///
    /// Memory pooling improves performance by:
    /// - Reducing garbage collection pauses
    /// - Avoiding repeated memory allocations
    /// - Improving cache locality
    ///
    /// This is especially beneficial for training loops that create many temporary tensors.
    /// </para>
    /// </remarks>
    public bool EnableMemoryPooling { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum number of tensor buffers to keep per shape.
    /// Default: 10.
    /// </summary>
    public int MaxPoolSizePerShape { get; set; } = 10;

    /// <summary>
    /// Gets or sets the maximum total elements in a tensor to pool.
    /// Tensors larger than this will not be pooled.
    /// Default: 10,000,000 (about 40MB for float32).
    /// </summary>
    public int MaxElementsToPool { get; set; } = 10_000_000;

    /// <summary>
    /// Gets or sets how the JIT compiler handles unsupported operations.
    /// Default: Fallback (use interpreted execution for entire graph if any op is unsupported).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When your model has operations the JIT can't compile,
    /// this setting controls what happens:
    ///
    /// - <b>Throw:</b> Stop with an error - use when you need all ops compiled
    /// - <b>Fallback:</b> (Default) Run the whole graph interpreted - always works
    /// - <b>Hybrid:</b> JIT the supported ops, interpret the rest - best performance
    /// - <b>Skip:</b> Ignore unsupported ops - dangerous, may give wrong results
    ///
    /// Hybrid mode is recommended for production when you have mixed-support graphs.
    /// It gives you JIT speed for supported operations while still handling all ops correctly.
    /// </para>
    /// </remarks>
    public UnsupportedLayerHandling UnsupportedLayerHandling { get; set; } = UnsupportedLayerHandling.Fallback;

    /// <summary>
    /// Gets or sets whether to log warnings for unsupported operations.
    /// Default: true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, you'll see warnings in logs when
    /// operations can't be JIT compiled. This helps you:
    /// - Identify which operations need fallback
    /// - Understand performance implications
    /// - Know when to request JIT support for new operation types
    /// </para>
    /// </remarks>
    public bool LogUnsupportedOperations { get; set; } = true;
}
