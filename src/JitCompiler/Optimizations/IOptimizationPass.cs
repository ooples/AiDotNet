using AiDotNet.JitCompiler.IR;

namespace AiDotNet.JitCompiler.Optimizations;

/// <summary>
/// Interface for optimization passes that transform IR graphs.
/// </summary>
/// <remarks>
/// <para>
/// An optimization pass takes an IR graph as input and returns a transformed
/// (optimized) IR graph as output. Passes should preserve the semantic meaning
/// of the computation while improving performance characteristics such as
/// execution time, memory usage, or code size.
/// </para>
/// <para><b>For Beginners:</b> This defines what an optimization pass must do.
///
/// Think of optimization passes as filters in a pipeline:
/// - Input: IR graph (description of computation)
/// - Process: Apply optimizations (make it better)
/// - Output: Optimized IR graph (same computation, faster execution)
///
/// Each optimization pass:
/// - Has a name (for logging and debugging)
/// - Takes a graph and returns an optimized version
/// - Preserves correctness (same results, just faster)
///
/// Example passes:
/// - Constant folding: Pre-compute constant expressions
/// - Dead code elimination: Remove unused operations
/// - Operation fusion: Combine multiple ops into one
///
/// By implementing this interface, you can create custom optimizations
/// and plug them into the JIT compiler's optimization pipeline.
/// </para>
/// </remarks>
public interface IOptimizationPass
{
    /// <summary>
    /// Gets the name of this optimization pass.
    /// </summary>
    /// <remarks>
    /// The name is used for logging, debugging, and reporting which
    /// optimizations were applied during compilation.
    /// </remarks>
    string Name { get; }

    /// <summary>
    /// Applies this optimization to an IR graph.
    /// </summary>
    /// <param name="graph">The IR graph to optimize.</param>
    /// <returns>An optimized IR graph.</returns>
    /// <remarks>
    /// <para>
    /// This method should return a new optimized graph. It should not modify
    /// the input graph (functional programming style). The returned graph
    /// must be semantically equivalent to the input (same computation),
    /// but can have different structure for better performance.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the magic happens!
    ///
    /// Your implementation should:
    /// 1. Analyze the input graph
    /// 2. Identify optimization opportunities
    /// 3. Transform the graph to be more efficient
    /// 4. Return the optimized graph
    ///
    /// Important rules:
    /// - Don't change what the graph computes (correctness!)
    /// - Don't modify the input graph (return a new one)
    /// - The optimized graph should produce identical results
    ///
    /// Example:
    /// Input:  t1 = Add(Const(2), Const(3)); t2 = Mul(t1, x)
    /// Output: t1 = Const(5); t2 = Mul(t1, x)
    /// (We pre-computed 2+3=5 at compile time!)
    /// </para>
    /// </remarks>
    IRGraph Optimize(IRGraph graph);
}
