using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Heterogeneity;

/// <summary>
/// FedNova-style normalization of client updates by local steps.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> FedNova reduces bias from clients doing different amounts of local work by
/// normalizing their updates before aggregation.
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public sealed class FedNovaHeterogeneityCorrection<T> : FederatedHeterogeneityCorrectionBase<T>
{
    /// <inheritdoc />
    public override Vector<T> Correct(int clientId, int roundNumber, Vector<T> globalParameters, Vector<T> localParameters, int localEpochs)
    {
        if (globalParameters.Length != localParameters.Length)
        {
            throw new ArgumentException("Global and local parameter vectors must have the same length for FedNova correction.");
        }

        int steps = SafeLocalSteps(localEpochs);
        var invSteps = NumOps.FromDouble(1.0 / steps);
        var delta = Engine.Subtract(localParameters, globalParameters);
        return Engine.Add(globalParameters, Engine.Multiply(delta, invSteps));
    }

    /// <inheritdoc />
    public override string GetCorrectionName() => "FedNova";
}

