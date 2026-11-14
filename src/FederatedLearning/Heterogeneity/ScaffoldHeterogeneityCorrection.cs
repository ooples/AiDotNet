using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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

    /// <inheritdoc />
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
        var lrStepsT = NumOps.FromDouble(lrSteps);
        var delta = Engine.Subtract(localParameters, globalParameters);
        var control = Engine.Subtract(_serverControlVariate, clientC);
        var corrected = Engine.Add(globalParameters, Engine.Add(delta, Engine.Multiply(control, lrStepsT)));

        // Update client control variate:
        // c_i <- c_i - c + (1/(lr*steps)) * (global - local)
        var invT = NumOps.FromDouble(1.0 / lrSteps);
        var oldClientC = clientC;
        var term = Engine.Multiply(Engine.Subtract(globalParameters, localParameters), invT);
        clientC = Engine.Add(Engine.Subtract(clientC, _serverControlVariate), term);
        _clientControlVariates[clientId] = clientC;

        // Update server control variate using the average delta in control variates over participating clients.
        // In the in-memory trainer, we approximate this online per call.
        _serverControlVariate = Engine.Add(_serverControlVariate, Engine.Subtract(clientC, oldClientC));

        return corrected;
    }

    /// <inheritdoc />
    public override string GetCorrectionName() => "SCAFFOLD";
}

