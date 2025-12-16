using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Heterogeneity;

/// <summary>
/// SCAFFOLD-style heterogeneity correction using control variates.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> SCAFFOLD reduces client drift by tracking "control variates" that estimate
/// how each client's local training differs from the global direction.
/// </remarks>
/// <typeparam name="T">Numeric type.</typeparam>
public sealed class ScaffoldHeterogeneityCorrection<T> : FederatedHeterogeneityCorrectionBase<T>
{
    private readonly double _clientLearningRate;
    private readonly Dictionary<int, Vector<T>> _clientControlVariates = new();
    private Vector<T>? _serverControlVariate;

    public ScaffoldHeterogeneityCorrection(double clientLearningRate = 1.0)
    {
        if (clientLearningRate <= 0.0)
        {
            throw new ArgumentOutOfRangeException(nameof(clientLearningRate), "Client learning rate must be positive.");
        }

        _clientLearningRate = clientLearningRate;
    }

    public override Vector<T> Correct(int clientId, int roundNumber, Vector<T> globalParameters, Vector<T> localParameters, int localEpochs)
    {
        if (globalParameters.Length != localParameters.Length)
        {
            throw new ArgumentException("Global and local parameter vectors must have the same length for SCAFFOLD correction.");
        }

        int n = globalParameters.Length;
        _serverControlVariate ??= new Vector<T>(n);

        if (!_clientControlVariates.TryGetValue(clientId, out var clientC) || clientC.Length != n)
        {
            clientC = new Vector<T>(n);
            _clientControlVariates[clientId] = clientC;
        }

        int steps = SafeLocalSteps(localEpochs);
        double lrSteps = _clientLearningRate * steps;
        if (lrSteps <= 0.0)
        {
            lrSteps = 1.0;
        }

        // Parameter-delta proxy: localParameters = globalParameters + delta
        // SCAFFOLD client update correction approximated by adding lr * (c - c_i) to the delta.
        var corrected = new Vector<T>(n);
        var lrStepsT = NumOps.FromDouble(lrSteps);
        for (int i = 0; i < n; i++)
        {
            var delta = NumOps.Subtract(localParameters[i], globalParameters[i]);
            var control = NumOps.Subtract(_serverControlVariate[i], clientC[i]);
            corrected[i] = NumOps.Add(globalParameters[i], NumOps.Add(delta, NumOps.Multiply(lrStepsT, control)));
        }

        // Update client control variate:
        // c_i <- c_i - c + (1/(lr*steps)) * (global - local)
        var inv = 1.0 / lrSteps;
        var invT = NumOps.FromDouble(inv);
        var oldClientC = clientC.Clone();
        for (int i = 0; i < n; i++)
        {
            var diff = NumOps.Subtract(globalParameters[i], localParameters[i]);
            var term = NumOps.Multiply(invT, diff);
            clientC[i] = NumOps.Add(NumOps.Subtract(clientC[i], _serverControlVariate[i]), term);
        }

        // Update server control variate using the average delta in control variates over participating clients.
        // In the in-memory trainer, we approximate this online per call.
        for (int i = 0; i < n; i++)
        {
            var deltaC = NumOps.Subtract(clientC[i], oldClientC[i]);
            _serverControlVariate[i] = NumOps.Add(_serverControlVariate[i], deltaC);
        }

        return corrected;
    }

    public override string GetCorrectionName() => "SCAFFOLD";
}

