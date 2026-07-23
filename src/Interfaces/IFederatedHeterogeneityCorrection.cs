using AiDotNet.Models;

namespace AiDotNet.Interfaces;

/// <summary>
/// Applies a heterogeneity correction transform to client updates in federated learning.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Some federated algorithms change how client updates are interpreted before the server aggregates them.
/// This interface lets AiDotNet swap in different correction methods while keeping the public facade simple.
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("FederatedHeterogeneityCorrection")]
public interface IFederatedHeterogeneityCorrection<T>
{
    /// <summary>
    /// Returns corrected client parameters to be used for aggregation.
    /// </summary>
    /// <param name="clientId">Client identifier.</param>
    /// <param name="roundNumber">Round number (0-indexed).</param>
    /// <param name="globalParameters">Global parameter vector at the start of the round.</param>
    /// <param name="localParameters">Client-trained parameter vector.</param>
    /// <param name="localEpochs">Local epochs used for training (proxy for local steps in simulation).</param>
    /// <returns>Corrected parameters.</returns>
    Vector<T> Correct(int clientId, int roundNumber, Vector<T> globalParameters, Vector<T> localParameters, int localEpochs);

    /// <summary>
    /// Gets the name of the correction method.
    /// </summary>
    string GetCorrectionName();
}

