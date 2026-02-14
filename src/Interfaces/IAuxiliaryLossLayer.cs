namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for neural network layers that report auxiliary losses in addition to the primary task loss.
/// Extends <see cref="IDiagnosticsProvider{T}"/> to provide diagnostic information about auxiliary loss computation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Auxiliary losses are additional loss terms that help guide training beyond the primary task objective.
/// They are particularly useful in complex architectures where certain desirable properties (like
/// balanced resource utilization or regularization) need explicit encouragement during training.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of auxiliary losses as "side goals" for training a neural network.
///
/// While the primary loss tells the network "make accurate predictions," auxiliary losses add
/// additional objectives like:
/// - "Use all experts equally" (load balancing in Mixture-of-Experts)
/// - "Keep activations small" (regularization)
/// - "Learn similar representations" (similarity objectives)
///
/// Real-world analogy:
/// Imagine you're training to be a chef (primary goal: make delicious food). But you also have
/// auxiliary goals:
/// - Keep your workspace clean (regularization)
/// - Use all your tools equally (load balancing)
/// - Work efficiently (computational constraints)
///
/// These auxiliary goals don't directly make the food taste better, but they help you become
/// a better, more well-rounded chef.
///
/// In the training loop, auxiliary losses are typically combined with the primary loss:
/// <code>
/// total_loss = primary_loss + (alpha * auxiliary_loss)
/// </code>
///
/// Where alpha is a weight that balances the importance of the auxiliary objective.
/// </para>
/// <para>
/// <b>Common Use Cases:</b>
/// <list type="bullet">
/// <item><description><b>Load Balancing (MoE):</b> Encourage balanced expert usage to prevent some experts from being underutilized</description></item>
/// <item><description><b>Sparsity Regularization:</b> Encourage sparse activations to improve efficiency</description></item>
/// <item><description><b>Contrastive Learning:</b> Encourage similar inputs to have similar representations</description></item>
/// <item><description><b>Multi-Task Learning:</b> Additional task objectives that share representations</description></item>
/// </list>
/// </para>
/// <para>
/// <b>Implementation Example:</b>
/// <code>
/// public class MixtureOfExpertsLayer&lt;T&gt; : LayerBase&lt;T&gt;, IAuxiliaryLossLayer&lt;T&gt;
/// {
///     public T ComputeAuxiliaryLoss()
///     {
///         // Compute load balancing loss
///         return CalculateLoadBalancingLoss();
///     }
/// }
///
/// // In training loop:
/// var primaryLoss = lossFunction.CalculateLoss(predictions, targets);
/// var auxiliaryLoss = NumOps.Zero;
/// if (layer is IAuxiliaryLossLayer&lt;T&gt; auxLayer)
/// {
///     auxiliaryLoss = auxLayer.ComputeAuxiliaryLoss();
/// }
/// var totalLoss = NumOps.Add(primaryLoss, NumOps.Multiply(alpha, auxiliaryLoss));
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("AuxiliaryLossLayer")]
public interface IAuxiliaryLossLayer<T> : IDiagnosticsProvider
{
    /// <summary>
    /// Computes the auxiliary loss for this layer based on the most recent forward pass.
    /// </summary>
    /// <returns>The auxiliary loss value.</returns>
    /// <remarks>
    /// <para>
    /// This method calculates an additional loss term that is added to the primary task loss during training.
    /// The auxiliary loss typically encourages desirable properties like balanced resource usage,
    /// sparsity, or other architectural constraints.
    /// </para>
    /// <para>
    /// The auxiliary loss should be computed based on cached values from the most recent forward pass.
    /// It is typically called after the forward pass but before the backward pass, and its value is
    /// added to the primary loss before computing gradients.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method calculates the "side goal" loss for the layer.
    ///
    /// When this method is called:
    /// - The layer has just finished its forward pass
    /// - It has cached information about what happened (e.g., which experts were used)
    /// - It uses this information to compute an auxiliary loss
    ///
    /// For example, in a Mixture-of-Experts layer with load balancing:
    /// 1. During forward pass, track which experts were selected
    /// 2. When ComputeAuxiliaryLoss() is called, calculate how imbalanced the usage was
    /// 3. Return a loss value that's higher when usage is more imbalanced
    /// 4. This encourages the training to use all experts more equally
    ///
    /// The returned value should be:
    /// - Zero or near-zero when the auxiliary objective is satisfied
    /// - Higher when the objective is violated
    /// - Always non-negative
    ///
    /// This loss gets added to the main loss, so the training process tries to minimize both.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when ComputeAuxiliaryLoss is called before a forward pass has been performed.
    /// </exception>
    T ComputeAuxiliaryLoss();

    /// <summary>
    /// Gets a value indicating whether the auxiliary loss should be included in training.
    /// </summary>
    /// <value>
    /// <c>true</c> if the auxiliary loss should be computed and added to the total loss; otherwise, <c>false</c>.
    /// </value>
    /// <remarks>
    /// <para>
    /// This property allows layers to dynamically enable or disable their auxiliary loss contribution.
    /// For example, load balancing loss might only be applied during training, not during inference.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> A switch to turn the auxiliary loss on or off.
    ///
    /// Why you might want to disable auxiliary loss:
    /// - During inference/testing: Auxiliary losses are typically only needed during training
    /// - During early training: Some auxiliary losses are only helpful after initial learning
    /// - For fine-tuning: You might want different auxiliary objectives
    ///
    /// The training loop should check this property:
    /// <code>
    /// if (layer is IAuxiliaryLossLayer&lt;T&gt; auxLayer && auxLayer.UseAuxiliaryLoss)
    /// {
    ///     auxiliaryLoss = auxLayer.ComputeAuxiliaryLoss();
    /// }
    /// </code>
    ///
    /// This gives you control over when auxiliary losses are applied without changing the layer's code.
    /// </para>
    /// </remarks>
    bool UseAuxiliaryLoss { get; }

    /// <summary>
    /// Gets or sets the weight (coefficient) for the auxiliary loss.
    /// </summary>
    /// <value>
    /// The weight to multiply the auxiliary loss by before adding it to the total loss.
    /// Typically a small value like 0.01 to 0.1.
    /// </value>
    /// <remarks>
    /// <para>
    /// The auxiliary loss weight (often denoted as alpha or lambda) controls how much the auxiliary
    /// objective influences training relative to the primary objective. A higher weight means the
    /// auxiliary loss has more influence.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Controls how important the auxiliary loss is relative to the main loss.
    ///
    /// The auxiliary loss weight balances two objectives:
    /// - Primary objective: Make accurate predictions (main loss)
    /// - Auxiliary objective: Satisfy the side goal (auxiliary loss)
    ///
    /// Total loss = primary_loss + (AuxiliaryLossWeight * auxiliary_loss)
    ///
    /// Choosing the right weight:
    /// - Too small (e.g., 0.001): Auxiliary loss has little effect, side goal ignored
    /// - Too large (e.g., 1.0): Auxiliary loss dominates, accuracy might suffer
    /// - Just right (e.g., 0.01-0.1): Balances both objectives
    ///
    /// Example:
    /// If AuxiliaryLossWeight = 0.01:
    /// - Primary loss of 2.5 contributes: 2.5
    /// - Auxiliary loss of 10.0 contributes: 0.1 (10.0 * 0.01)
    /// - Total loss: 2.6
    ///
    /// This way, the main task is still the priority, but the side goal provides some guidance.
    ///
    /// You often need to tune this value experimentally:
    /// 1. Start with a small value (e.g., 0.01)
    /// 2. Monitor both losses during training
    /// 3. Increase if the auxiliary objective isn't being achieved
    /// 4. Decrease if the primary task accuracy suffers
    /// </para>
    /// </remarks>
    T AuxiliaryLossWeight { get; set; }

    /// <summary>
    /// Gets diagnostic information about the auxiliary loss computation.
    /// This method delegates to <see cref="IDiagnosticsProvider{T}.GetDiagnostics"/> for implementation.
    /// </summary>
    /// <returns>
    /// A dictionary containing diagnostic metrics related to the auxiliary loss.
    /// For example, load balancing might return expert utilization statistics.
    /// </returns>
    /// <remarks>
    /// <para>
    /// This method provides additional information about the auxiliary loss that can be useful
    /// for monitoring, debugging, and understanding model behavior. The returned dictionary keys
    /// and values depend on the specific type of auxiliary loss.
    /// </para>
    /// <para>
    /// <b>Implementation Note:</b> Since this interface extends <see cref="IDiagnosticsProvider{T}"/>,
    /// implementations should provide the core diagnostic logic in <see cref="IDiagnosticsProvider{T}.GetDiagnostics"/>
    /// and have this method delegate to it. This follows the interface segregation principle and allows
    /// auxiliary loss diagnostics to be accessed through both the specific and general diagnostic interfaces.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Gets extra information to help you understand what's happening.
    ///
    /// This is like a detailed report card for the auxiliary loss. While ComputeAuxiliaryLoss()
    /// gives you a single number, GetAuxiliaryLossDiagnostics() tells you the story behind it.
    ///
    /// For example, with load balancing in MoE:
    /// - ComputeAuxiliaryLoss() might return: 0.05 (the loss value)
    /// - GetAuxiliaryLossDiagnostics() might return:
    ///   {
    ///     "expert_0_usage": "0.35",   // Expert 0 was used 35% of the time
    ///     "expert_1_usage": "0.25",   // Expert 1 was used 25% of the time
    ///     "expert_2_usage": "0.30",   // Expert 2 was used 30% of the time
    ///     "expert_3_usage": "0.10",   // Expert 3 was used only 10% of the time
    ///     "balance_variance": "0.012" // Variance in usage (lower is better)
    ///   }
    ///
    /// This diagnostic information helps you:
    /// - Understand if the auxiliary loss is working
    /// - Debug training issues
    /// - Tune the auxiliary loss weight
    /// - Monitor model health over time
    ///
    /// You might log these diagnostics periodically:
    /// <code>
    /// if (iteration % 100 == 0 && layer is IAuxiliaryLossLayer&lt;T&gt; auxLayer)
    /// {
    ///     var diagnostics = auxLayer.GetAuxiliaryLossDiagnostics();
    ///     foreach (var (key, value) in diagnostics)
    ///     {
    ///         Console.WriteLine($"{key}: {value}");
    ///     }
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    Dictionary<string, string> GetAuxiliaryLossDiagnostics();
}
