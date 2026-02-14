namespace AiDotNet.Interfaces;

/// <summary>
/// Applies a server-side optimization step in federated learning (FedOpt family).
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> In classic FedAvg, the server simply replaces the global model with the averaged client model.
/// In FedOpt, the server treats the aggregated update like a "gradient" and applies an optimizer step (like Adam),
/// which can improve stability and convergence, especially with non-IID data.
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("FederatedServerOptimizer")]
public interface IFederatedServerOptimizer<T>
{
    /// <summary>
    /// Updates global parameters given the current parameters and an aggregated target.
    /// </summary>
    /// <param name="currentGlobalParameters">The current global parameter vector.</param>
    /// <param name="aggregatedTargetParameters">The aggregated target parameter vector (e.g., FedAvg output).</param>
    /// <returns>The updated global parameter vector.</returns>
    Vector<T> Step(Vector<T> currentGlobalParameters, Vector<T> aggregatedTargetParameters);

    /// <summary>
    /// Gets the name of the server optimizer.
    /// </summary>
    string GetOptimizerName();
}

