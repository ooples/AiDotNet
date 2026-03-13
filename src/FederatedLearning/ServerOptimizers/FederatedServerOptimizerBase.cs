using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Interfaces;

namespace AiDotNet.FederatedLearning.ServerOptimizers;

/// <summary>
/// Base class for server-side federated optimizers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> for provides AI safety functionality. Default values follow the original paper settings.</para>
/// </remarks>
public abstract class FederatedServerOptimizerBase<T> : FederatedLearningComponentBase<T>, IFederatedServerOptimizer<T>
{
    public abstract Vector<T> Step(Vector<T> currentGlobalParameters, Vector<T> aggregatedTargetParameters);

    public abstract string GetOptimizerName();

    protected static void ValidateVectors(Vector<T> currentGlobalParameters, Vector<T> aggregatedTargetParameters)
    {
        if (currentGlobalParameters == null)
        {
            throw new ArgumentNullException(nameof(currentGlobalParameters));
        }

        if (aggregatedTargetParameters == null)
        {
            throw new ArgumentNullException(nameof(aggregatedTargetParameters));
        }

        if (currentGlobalParameters.Length != aggregatedTargetParameters.Length)
        {
            throw new ArgumentException("Parameter length mismatch between current and aggregated parameters.");
        }
    }
}

