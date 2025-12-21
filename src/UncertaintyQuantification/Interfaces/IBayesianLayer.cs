namespace AiDotNet.UncertaintyQuantification.Interfaces;

/// <summary>
/// Defines the contract for Bayesian neural network layers that support probabilistic inference.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Bayesian layer is different from a regular neural network layer because
/// instead of having fixed weights, it has distributions over weights.
///
/// Think of regular weights as saying "the connection strength is exactly 2.5", while Bayesian weights
/// say "the connection strength is probably around 2.5, but could be anywhere from 2.0 to 3.0".
///
/// This probabilistic approach allows the network to express uncertainty in its predictions,
/// which is crucial for safety-critical applications.
/// </para>
/// </remarks>
public interface IBayesianLayer<T>
{
    /// <summary>
    /// Samples from the weight distribution for stochastic forward passes.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This method randomly selects weights from the probability distribution.
    /// Each time you call this, you might get slightly different weights, representing the
    /// model's uncertainty about what the true weights should be.
    /// </remarks>
    void SampleWeights();

    /// <summary>
    /// Gets the KL divergence term for variational inference.
    /// </summary>
    /// <returns>The KL divergence value.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> KL divergence measures how different the learned weight distribution
    /// is from a simple baseline distribution (called the prior).
    ///
    /// This is used during training to prevent the model from becoming too confident or too uncertain.
    /// Think of it as a "regularization penalty" that keeps the weight distributions reasonable.
    /// </remarks>
    T GetKLDivergence();

    /// <summary>
    /// Adds the KL divergence gradients (regularization term) into the layer's accumulated gradients.
    /// </summary>
    /// <param name="klScale">Scaling applied to the KL term (e.g., 1/N for dataset size).</param>
    void AddKLDivergenceGradients(T klScale);
}
