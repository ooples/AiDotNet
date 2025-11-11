namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for components that provide diagnostic information for monitoring and debugging.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface enables neural network components (layers, networks, loss functions, etc.)
/// to provide detailed diagnostic information about their internal state and behavior.
/// This is particularly useful for:
/// <list type="bullet">
/// <item><description>Monitoring training progress</description></item>
/// <item><description>Debugging model behavior</description></item>
/// <item><description>Performance analysis and optimization</description></item>
/// <item><description>Understanding model decisions (explainability)</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as a "health report" interface for neural network components.
///
/// Just like you might want to check various health metrics for your body (heart rate, blood pressure, etc.),
/// you want to monitor various metrics for your neural network components during training and inference.
///
/// Real-world analogy:
/// Imagine you're driving a car. Your dashboard shows:
/// - Speed (how fast you're going)
/// - RPM (engine revolutions)
/// - Fuel level (remaining energy)
/// - Temperature (engine heat)
///
/// Similarly, a neural network layer might report:
/// - Activation statistics (min, max, mean values)
/// - Gradient flow (how well training signals propagate)
/// - Resource utilization (memory usage, computation time)
/// - Layer-specific metrics (attention weights, expert usage, etc.)
///
/// This information helps you understand:
/// - Is my model training properly?
/// - Are there any bottlenecks or issues?
/// - Which parts of the model are most active?
/// - Is the model behaving as expected?
/// </para>
/// <para>
/// <b>Industry Best Practices:</b>
/// <list type="bullet">
/// <item><description><b>Consistent Keys:</b> Use standardized key names across similar components</description></item>
/// <item><description><b>Meaningful Values:</b> Provide human-readable string representations</description></item>
/// <item><description><b>Hierarchical Organization:</b> Use prefixes to group related metrics (e.g., "activation.mean", "activation.std")</description></item>
/// <item><description><b>Efficient Computation:</b> Diagnostics should be cheap to compute or cached</description></item>
/// <item><description><b>Optional Depth:</b> Consider providing basic and detailed diagnostic modes</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Implementation Example:</b>
/// <code>
/// public class DenseLayer&lt;T&gt; : LayerBase&lt;T&gt;, IDiagnosticsProvider&lt;T&gt;
/// {
///     private Tensor&lt;T&gt;? _lastActivations;
///
///     public Dictionary&lt;string, string&gt; GetDiagnostics()
///     {
///         var diagnostics = new Dictionary&lt;string, string&gt;();
///
///         if (_lastActivations != null)
///         {
///             diagnostics["activation.mean"] = ComputeMean(_lastActivations).ToString();
///             diagnostics["activation.std"] = ComputeStd(_lastActivations).ToString();
///             diagnostics["activation.sparsity"] = ComputeSparsity(_lastActivations).ToString();
///         }
///
///         diagnostics["parameter.count"] = ParameterCount.ToString();
///         diagnostics["layer.type"] = "Dense";
///
///         return diagnostics;
///     }
/// }
///
/// // In monitoring code:
/// foreach (var layer in network.Layers)
/// {
///     if (layer is IDiagnosticsProvider&lt;T&gt; diagnosticLayer)
///     {
///         var metrics = diagnosticLayer.GetDiagnostics();
///         LogMetrics(metrics);
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public interface IDiagnosticsProvider<T>
{
    /// <summary>
    /// Gets diagnostic information about this component's state and behavior.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics. Keys should be descriptive and use
    /// consistent naming conventions (e.g., "activation.mean", "gradient.norm").
    /// Values should be human-readable string representations of the metrics.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method should return diagnostic information that is useful for understanding
    /// the component's current state. The specific metrics returned depend on the component type:
    /// <list type="bullet">
    /// <item><description><b>Layers:</b> Activation statistics, gradient flow, sparsity, etc.</description></item>
    /// <item><description><b>Networks:</b> Aggregate metrics, layer-by-layer summaries</description></item>
    /// <item><description><b>Loss Functions:</b> Loss components, regularization terms</description></item>
    /// <item><description><b>Optimizers:</b> Learning rates, momentum values, update statistics</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method returns a report card with various metrics.
    ///
    /// The returned dictionary is like a set of labeled measurements:
    /// - Keys: What you're measuring (e.g., "mean_activation", "sparsity")
    /// - Values: The measurement results as strings (e.g., "0.42", "85% sparse")
    ///
    /// Example for a Dense layer:
    /// <code>
    /// {
    ///     "activation.mean": "0.342",
    ///     "activation.std": "0.156",
    ///     "activation.min": "-0.82",
    ///     "activation.max": "1.24",
    ///     "activation.sparsity": "0.23",
    ///     "gradient.norm": "0.042",
    ///     "weights.norm": "15.6",
    ///     "layer.output_size": "256"
    /// }
    /// </code>
    ///
    /// You can use this information to:
    /// 1. <b>Detect training issues:</b> If activations are all zero, something might be wrong
    /// 2. <b>Tune hyperparameters:</b> If gradients are too large/small, adjust learning rate
    /// 3. <b>Monitor convergence:</b> Track metrics over time to see if training is progressing
    /// 4. <b>Compare experiments:</b> See how different configurations affect internal behavior
    ///
    /// Common patterns:
    /// <code>
    /// // Log diagnostics periodically during training
    /// if (epoch % 10 == 0)
    /// {
    ///     foreach (var layer in network.Layers)
    ///     {
    ///         if (layer is IDiagnosticsProvider&lt;T&gt; diag)
    ///         {
    ///             Console.WriteLine($"Layer {layer.Name}:");
    ///             foreach (var (key, value) in diag.GetDiagnostics())
    ///             {
    ///                 Console.WriteLine($"  {key}: {value}");
    ///             }
    ///         }
    ///     }
    /// }
    /// </code>
    /// </para>
    /// <para>
    /// <b>Performance Note:</b>
    /// Diagnostic computation should be efficient. If expensive calculations are needed,
    /// consider caching results or computing them only when diagnostics are requested.
    /// Diagnostics should not significantly impact training performance.
    /// </para>
    /// </remarks>
    Dictionary<string, string> GetDiagnostics();
}
