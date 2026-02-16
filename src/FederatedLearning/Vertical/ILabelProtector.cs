using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Protects label holder information from being inferred by feature-holding parties.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In vertical FL, the label holder (e.g., a hospital that knows
/// patient outcomes) computes the loss and sends gradients back to feature parties (e.g., a bank
/// that knows income). Without protection, the bank could analyze these gradients to figure out
/// which patients had bad outcomes.</para>
///
/// <para>Label protection adds noise or other protections to the gradients before they're sent
/// to feature parties, preventing this kind of inference attack. The trade-off is between
/// privacy (more noise = more protection) and accuracy (more noise = slower learning).</para>
///
/// <para><b>Common attacks prevented:</b></para>
/// <list type="bullet">
/// <item><description>Label inference from gradient magnitude (large gradients suggest misclassification)</description></item>
/// <item><description>Batch-level label distribution estimation from gradient statistics</description></item>
/// <item><description>Model inversion attacks reconstructing labels from embedding gradients</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public interface ILabelProtector<T>
{
    /// <summary>
    /// Adds privacy protection to gradients before sending them to feature-holding parties.
    /// </summary>
    /// <param name="gradients">The raw gradients computed by the label holder.</param>
    /// <returns>Protected gradients safe to share with feature parties.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method takes the true gradients and adds noise or
    /// other transformations so that feature parties cannot reverse-engineer the labels
    /// from the gradients they receive.</para>
    /// </remarks>
    Tensor<T> ProtectGradients(Tensor<T> gradients);

    /// <summary>
    /// Adds privacy protection to the loss value before sharing it.
    /// </summary>
    /// <param name="loss">The raw loss value.</param>
    /// <returns>A protected loss value that can be shared without revealing individual labels.</returns>
    T ProtectLoss(T loss);

    /// <summary>
    /// Gets the cumulative privacy budget consumed so far.
    /// </summary>
    /// <returns>
    /// A tuple of (epsilon, delta) representing the total privacy cost.
    /// Smaller values mean more privacy has been spent.
    /// </returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each time gradients are protected with noise, some "privacy budget"
    /// is consumed. Once the budget runs out, the model must stop training to maintain
    /// the privacy guarantee. Epsilon measures privacy loss (lower = more private).</para>
    /// </remarks>
    (double Epsilon, double Delta) GetPrivacyBudgetSpent();
}
