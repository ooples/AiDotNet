using AiDotNet.Autodiff;

namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that can expose their computation graph for JIT compilation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Models implementing this interface can be JIT compiled for significantly faster inference.
/// JIT compilation converts the model's computation graph into optimized native code, providing
/// 5-10x speedup for complex models.
/// </para>
/// <para><b>For Beginners:</b> JIT (Just-In-Time) compilation is like translating your model's
/// calculations into a faster language. This interface lets models opt-in to this optimization.
///
/// Benefits of JIT compilation:
/// - 2-3x faster for simple operations
/// - 5-10x faster for complex models
/// - Near-zero overhead for cached compilations
/// - Automatic operation fusion and optimization
///
/// Requirements:
/// - Model must use ComputationNode-based computation graphs
/// - Graph structure must be deterministic (same structure for different inputs)
///
/// <b>Note:</b> Currently, neural networks using layer-based architecture need to be enhanced
/// to export their forward pass as a computation graph to support JIT compilation.
/// This is planned for a future update.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("JitCompilable")]
public interface IJitCompilable<T>
{
    /// <summary>
    /// Exports the model's computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes (parameters).</param>
    /// <returns>The output computation node representing the model's prediction.</returns>
    /// <remarks>
    /// <para>
    /// This method should construct a computation graph representing the model's forward pass.
    /// The graph should use placeholder input nodes that will be filled with actual data during execution.
    /// </para>
    /// <para><b>For Beginners:</b> This method creates a "recipe" of your model's calculations
    /// that the JIT compiler can optimize.
    ///
    /// The method should:
    /// 1. Create placeholder nodes for inputs (features, parameters)
    /// 2. Build the computation graph using TensorOperations
    /// 3. Return the final output node
    /// 4. Add all input nodes to the inputNodes list (in order)
    ///
    /// Example for a simple linear model (y = Wx + b):
    /// <code>
    /// public ComputationNode&lt;T&gt; ExportComputationGraph(List&lt;ComputationNode&lt;T&gt;&gt; inputNodes)
    /// {
    ///     // Create placeholder inputs
    ///     var x = TensorOperations&lt;T&gt;.Variable(new Tensor&lt;T&gt;(InputShape), "x");
    ///     var W = TensorOperations&lt;T&gt;.Variable(Weights, "W");
    ///     var b = TensorOperations&lt;T&gt;.Variable(Bias, "b");
    ///
    ///     // Add inputs in order
    ///     inputNodes.Add(x);
    ///     inputNodes.Add(W);
    ///     inputNodes.Add(b);
    ///
    ///     // Build graph: y = Wx + b
    ///     var matmul = TensorOperations&lt;T&gt;.MatMul(x, W);
    ///     var output = TensorOperations&lt;T&gt;.Add(matmul, b);
    ///
    ///     return output;
    /// }
    /// </code>
    ///
    /// The JIT compiler will then:
    /// - Optimize the graph (fuse operations, eliminate dead code)
    /// - Compile it to fast native code
    /// - Cache the compiled version for reuse
    /// </para>
    /// </remarks>
    ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes);

    /// <summary>
    /// Gets whether this model currently supports JIT compilation.
    /// </summary>
    /// <value>True if the model can be JIT compiled, false otherwise.</value>
    /// <remarks>
    /// <para>
    /// Some models may not support JIT compilation due to:
    /// - Dynamic graph structure (changes based on input)
    /// - Lack of computation graph representation
    /// - Use of operations not yet supported by the JIT compiler
    /// </para>
    /// <para><b>For Beginners:</b> This tells you whether this specific model can benefit from JIT compilation.
    ///
    /// Models return false if they:
    /// - Use layer-based architecture without graph export (e.g., current neural networks)
    /// - Have control flow that changes based on input data
    /// - Use operations the JIT compiler doesn't understand yet
    ///
    /// In these cases, the model will still work normally, just without JIT acceleration.
    /// </para>
    /// </remarks>
    bool SupportsJitCompilation { get; }
}
