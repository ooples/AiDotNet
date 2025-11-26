namespace AiDotNet.Interfaces;

/// <summary>
/// Defines privacy-preserving mechanisms for federated learning to protect client data.
/// </summary>
/// <remarks>
/// This interface represents techniques to ensure that model updates don't leak sensitive
/// information about individual data points in clients' local datasets.
///
/// <b>For Beginners:</b> Privacy mechanisms are like filters that protect sensitive information
/// while still allowing useful knowledge to be shared.
///
/// Think of privacy mechanisms as protective measures:
/// - Differential Privacy: Adds carefully calibrated noise to make individual data unidentifiable
/// - Secure Aggregation: Encrypts updates so the server only sees the combined result
/// - Homomorphic Encryption: Allows computation on encrypted data
///
/// For example, in a hospital scenario:
/// - Without privacy: Model updates might reveal information about specific patients
/// - With differential privacy: Random noise is added so you can't identify individual patients
/// - The noise is calibrated so the overall patterns remain accurate
///
/// Privacy mechanisms provide mathematical guarantees:
/// - Epsilon (ε): Privacy budget - lower values mean stronger privacy
/// - Delta (δ): Probability that privacy guarantee fails
/// - Common setting: ε=1.0, δ=1e-5 means strong privacy with high confidence
/// </remarks>
/// <typeparam name="TModel">The type of model to apply privacy mechanisms to.</typeparam>
public interface IPrivacyMechanism<TModel>
{
    /// <summary>
    /// Applies privacy-preserving techniques to a model update before sharing it.
    /// </summary>
    /// <remarks>
    /// This method transforms model updates to provide privacy guarantees while maintaining utility.
    ///
    /// <b>For Beginners:</b> This is like redacting sensitive parts of a document before sharing it.
    /// You remove or obscure information that could identify individuals while keeping the
    /// useful content intact.
    ///
    /// Common techniques:
    /// - Differential Privacy: Adds random noise proportional to sensitivity
    /// - Gradient Clipping: Limits the magnitude of updates to prevent outliers
    /// - Local DP: Each client adds noise before sending updates
    /// - Central DP: Server adds noise after aggregation
    ///
    /// For example with differential privacy:
    /// 1. Client trains model and computes weight updates
    /// 2. Applies gradient clipping to limit maximum change
    /// 3. Adds calibrated Gaussian noise to each weight
    /// 4. Sends noisy update to server
    /// 5. Even if server is compromised, individual data remains private
    /// </remarks>
    /// <param name="model">The model update to apply privacy to.</param>
    /// <param name="epsilon">Privacy budget parameter - smaller values provide stronger privacy.</param>
    /// <param name="delta">Probability of privacy guarantee failure - typically very small (e.g., 1e-5).</param>
    /// <returns>The model update with privacy mechanisms applied.</returns>
    TModel ApplyPrivacy(TModel model, double epsilon, double delta);

    /// <summary>
    /// Gets the current privacy budget consumed by this mechanism.
    /// </summary>
    /// <remarks>
    /// Privacy budget is a finite resource in differential privacy. Each time you share
    /// information, you "spend" some privacy budget. Once exhausted, you can no longer
    /// provide strong privacy guarantees.
    ///
    /// <b>For Beginners:</b> Think of privacy budget like a bank account for privacy.
    /// Each time you share data, you withdraw from this account. When the account is empty,
    /// you've used up your privacy guarantees and should stop sharing.
    ///
    /// For example:
    /// - Start with privacy budget ε=10
    /// - Round 1: Share update with ε=1, remaining budget = 9
    /// - Round 2: Share update with ε=1, remaining budget = 8
    /// - After 10 rounds, budget is exhausted
    /// </remarks>
    /// <returns>The amount of privacy budget consumed so far.</returns>
    double GetPrivacyBudgetConsumed();

    /// <summary>
    /// Gets the name of the privacy mechanism.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This identifies which privacy technique is being used,
    /// helpful for documentation and comparing different privacy approaches.
    /// </remarks>
    /// <returns>A string describing the privacy mechanism (e.g., "Gaussian Mechanism", "Laplace Mechanism").</returns>
    string GetMechanismName();
}
