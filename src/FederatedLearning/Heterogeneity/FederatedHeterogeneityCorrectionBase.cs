using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.Heterogeneity;

/// <summary>
/// Base class for heterogeneity correction implementations.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class FederatedHeterogeneityCorrectionBase<T> : FederatedLearningComponentBase<T>, IFederatedHeterogeneityCorrection<T>
{
    public abstract Vector<T> Correct(int clientId, int roundNumber, Vector<T> globalParameters, Vector<T> localParameters, int localEpochs);

    public abstract string GetCorrectionName();

    protected static int SafeLocalSteps(int localEpochs) => Math.Max(1, localEpochs);
}

