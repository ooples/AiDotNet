using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Heterogeneity;

/// <summary>
/// FedDyn-style dynamic regularization using a per-client drift accumulator.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> FedDyn reduces client drift by maintaining an extra per-client state that
/// accumulates how the client tends to move away from the global model.
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
public sealed class FedDynHeterogeneityCorrection<T> : FederatedHeterogeneityCorrectionBase<T>
{
    private readonly double _alpha;
    private readonly Dictionary<int, Vector<T>> _clientAccumulators = new();

    public FedDynHeterogeneityCorrection(double alpha = 0.01)
    {
        if (alpha <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha must be positive.");
        }

        _alpha = alpha;
    }

    public override Vector<T> Correct(int clientId, int roundNumber, Vector<T> globalParameters, Vector<T> localParameters, int localEpochs)
    {
        if (globalParameters.Length != localParameters.Length)
        {
            throw new ArgumentException("Global and local parameter vectors must have the same length for FedDyn correction.");
        }

        int n = globalParameters.Length;
        if (!_clientAccumulators.TryGetValue(clientId, out var acc) || acc.Length != n)
        {
            acc = new Vector<T>(n);
            _clientAccumulators[clientId] = acc;
        }

        // Update accumulator using the observed drift (local - global).
        for (int i = 0; i < n; i++)
        {
            var drift = NumOps.Subtract(localParameters[i], globalParameters[i]);
            acc[i] = NumOps.Add(acc[i], drift);
        }

        // Apply correction: local + alpha * accumulator.
        var corrected = new Vector<T>(n);
        var alphaT = NumOps.FromDouble(_alpha);
        for (int i = 0; i < n; i++)
        {
            corrected[i] = NumOps.Add(localParameters[i], NumOps.Multiply(alphaT, acc[i]));
        }

        return corrected;
    }

    public override string GetCorrectionName() => "FedDyn";
}

