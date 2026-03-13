using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.FederatedLearning.Cryptography;

/// <summary>
/// Base class for homomorphic encryption providers.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class HomomorphicEncryptionProviderBase<T> : FederatedLearningComponentBase<T>, IHomomorphicEncryptionProvider<T>
{
    public abstract Vector<T> AggregateEncryptedWeightedAverage(
        Dictionary<int, Vector<T>> clientParameters,
        Dictionary<int, double> clientWeights,
        Vector<T> globalBaseline,
        IReadOnlyList<int> encryptedIndices,
        HomomorphicEncryptionOptions options);

    public abstract string GetProviderName();
}

