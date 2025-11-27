using AiDotNet.JitCompiler;

namespace AiDotNet.Configuration;

/// <summary>
/// Configuration for JIT (Just-In-Time) compilation of models for accelerated inference.
/// </summary>
/// <remarks>
/// <para>
/// JIT compilation converts your model's computation graph into optimized native code,
/// providing significant performance improvements for inference. This configuration allows
/// you to control whether and how JIT compilation is applied.
/// </para>
/// <para><b>For Beginners:</b> JIT compilation is like translating your model into a faster language
/// before using it. This can make predictions 5-10x faster, especially for complex models.
///
/// Key benefits:
/// - <b>Performance:</b> 2-3x faster for simple operations, 5-10x for complex models
/// - <b>Optimization:</b> Automatic operation fusion, dead code elimination
/// - <b>Caching:</b> Compiled once, reused many times
///
/// When to enable JIT:
/// - Production inference (maximize speed)
/// - Batch processing (repeated predictions)
/// - Large or complex models (more optimization opportunities)
///
/// When NOT to enable JIT:
/// - Training (JIT is for inference only)
/// - Models that change structure dynamically
/// - Very simple models (compilation overhead exceeds benefits)
///
/// <b>Note:</b> Your model must implement IJitCompilable to support JIT compilation.
/// Currently, this works with models built using TensorOperations computation graphs.
/// Neural networks using layer-based architecture will be supported in a future update.
/// </para>
/// </remarks>
public class JitCompilationConfig
{
    /// <summary>
    /// Gets or sets whether JIT compilation is enabled.
    /// </summary>
    /// <value>True to enable JIT compilation, false to disable (default: false).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Turn this on to make your model's predictions faster.
    ///
    /// When enabled:
    /// - The model's computation graph is compiled during BuildAsync()
    /// - Predictions use the compiled version (5-10x faster)
    /// - Compilation happens once, then results are cached
    ///
    /// When disabled:
    /// - The model runs normally without JIT acceleration
    /// - No compilation overhead during build
    /// - Predictions use the standard execution path
    ///
    /// The compilation adds 10-50ms during model building, but makes every subsequent
    /// prediction much faster. For production deployment, this is almost always worth it.
    /// </para>
    /// </remarks>
    public bool Enabled { get; set; } = false;

    /// <summary>
    /// Gets or sets the JIT compiler options for optimization and performance tuning.
    /// </summary>
    /// <value>Compiler options controlling optimization passes (default: all optimizations enabled).</value>
    /// <remarks>
    /// <para>
    /// These options control how the JIT compiler optimizes your model's computation graph.
    /// The default configuration enables all optimizations, which works well for most cases.
    /// </para>
    /// <para><b>For Beginners:</b> These settings control HOW the JIT compiler optimizes your model.
    ///
    /// Available optimizations:
    /// - <b>Constant Folding:</b> Pre-computes constant values
    /// - <b>Dead Code Elimination:</b> Removes unused operations
    /// - <b>Operation Fusion:</b> Combines multiple operations into one (biggest speedup!)
    /// - <b>Caching:</b> Reuses compiled graphs with same structure
    ///
    /// Default settings (all enabled) work well for 99% of cases. You might customize if:
    /// - Debugging: Disable optimizations to see original graph structure
    /// - Memory constrained: Disable caching to reduce memory usage
    /// - Experimental: Test impact of specific optimizations
    ///
    /// Example:
    /// <code>
    /// var config = new JitCompilationConfig
    /// {
    ///     Enabled = true,
    ///     CompilerOptions = new JitCompilerOptions
    ///     {
    ///         EnableOperationFusion = true,  // Biggest perf gain
    ///         EnableDeadCodeElimination = true,
    ///         EnableConstantFolding = true,
    ///         EnableCaching = true
    ///     }
    /// };
    /// </code>
    /// </para>
    /// </remarks>
    public JitCompilerOptions CompilerOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets whether to throw an exception if JIT compilation fails.
    /// </summary>
    /// <value>True to throw on failure, false to fall back to normal execution (default: false).</value>
    /// <remarks>
    /// <para>
    /// When JIT compilation fails (e.g., model doesn't support it, unsupported operations),
    /// this setting determines whether to throw an exception or silently fall back to normal execution.
    /// </para>
    /// <para><b>For Beginners:</b> This controls what happens if JIT compilation can't be done.
    ///
    /// When true (ThrowOnFailure = true):
    /// - If JIT fails, an exception is thrown immediately
    /// - Build process stops
    /// - You're notified of the problem right away
    /// - Good for debugging or when JIT is critical
    ///
    /// When false (ThrowOnFailure = false, default):
    /// - If JIT fails, a warning is logged but build continues
    /// - Model works normally without JIT acceleration
    /// - Graceful degradation
    /// - Good for production where availability > performance
    ///
    /// Common reasons JIT might fail:
    /// - Model doesn't implement IJitCompilable
    /// - Model has dynamic graph structure
    /// - Operation types not yet supported by JIT compiler
    ///
    /// Example:
    /// <code>
    /// // Development: Fail fast to catch issues
    /// var devConfig = new JitCompilationConfig { Enabled = true, ThrowOnFailure = true };
    ///
    /// // Production: Graceful fallback
    /// var prodConfig = new JitCompilationConfig { Enabled = true, ThrowOnFailure = false };
    /// </code>
    /// </para>
    /// </remarks>
    public bool ThrowOnFailure { get; set; } = false;
}
