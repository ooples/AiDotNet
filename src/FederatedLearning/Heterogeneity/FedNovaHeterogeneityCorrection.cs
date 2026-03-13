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
public sealed class FedNovaHeterogeneityCorrection<T> : FederatedHeterogeneityCorrectionBase<T>
{
    public override Vector<T> Correct(int clientId, int roundNumber, Vector<T> globalParameters, Vector<T> localParameters, int localEpochs)
    {
        if (globalParameters.Length != localParameters.Length)
        {
            throw new ArgumentException("Global and local parameter vectors must have the same length for FedNova correction.");
        }

        int steps = SafeLocalSteps(localEpochs);
        int n = globalParameters.Length;
        var corrected = new Vector<T>(n);
        var invSteps = NumOps.FromDouble(1.0 / steps);
        for (int i = 0; i < n; i++)
        {
            var delta = NumOps.Subtract(localParameters[i], globalParameters[i]);
            var normalized = NumOps.Multiply(invSteps, delta);
            corrected[i] = NumOps.Add(globalParameters[i], normalized);
        }

        return corrected;
    }

    public override string GetCorrectionName() => "FedNova";
}

