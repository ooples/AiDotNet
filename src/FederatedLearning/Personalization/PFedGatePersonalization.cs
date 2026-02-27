namespace AiDotNet.FederatedLearning.Personalization;

/// <summary>
/// Implements pFedGate — gated layer-wise mixture of local and global parameters.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> pFedGate learns a small "gate" per layer that decides how much
/// to use the global model vs. the client's local model for that layer. Gates are numbers
/// between 0 and 1: gate=0 means "use fully global" and gate=1 means "use fully local."
/// Each client learns different gate values based on their data distribution. The gates
/// are lightweight (one scalar per layer) and personalized (not aggregated).</para>
///
/// <para>Per-layer mixing:</para>
/// <code>
/// effective_params_l = gate_l * local_params_l + (1 - gate_l) * global_params_l
/// </code>
///
/// <para>Reference: Chen, S., et al. (2023). "pFedGate: Data-Driven Expert Gating for
/// Personalized Federated Learning." NeurIPS 2023.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for model parameters.</typeparam>
public class PFedGatePersonalization<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly double _gateInitValue;
    private readonly double _gateLearningRate;
    private Dictionary<string, double>? _gates;

    /// <summary>
    /// Creates a new pFedGate personalization strategy.
    /// </summary>
    /// <param name="gateInitValue">Initial gate value (bias towards global model). Default: 0.1.</param>
    /// <param name="gateLearningRate">Learning rate for gate optimization. Default: 0.01.</param>
    public PFedGatePersonalization(double gateInitValue = 0.1, double gateLearningRate = 0.01)
    {
        if (gateInitValue < 0 || gateInitValue > 1)
        {
            throw new ArgumentOutOfRangeException(nameof(gateInitValue), "Gate init must be in [0, 1].");
        }

        if (gateLearningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(gateLearningRate), "Gate LR must be positive.");
        }

        _gateInitValue = gateInitValue;
        _gateLearningRate = gateLearningRate;
    }

    /// <summary>
    /// Initializes gates for a model structure.
    /// </summary>
    public void InitializeGates(Dictionary<string, T[]> modelStructure)
    {
        _gates = new Dictionary<string, double>(modelStructure.Count);
        foreach (var layerName in modelStructure.Keys)
        {
            _gates[layerName] = _gateInitValue;
        }
    }

    /// <summary>
    /// Applies gate mixing to produce effective parameters per layer.
    /// </summary>
    /// <param name="globalParams">Global model parameters from server.</param>
    /// <param name="localParams">Client's local model parameters.</param>
    /// <returns>Mixed parameters using learned gates.</returns>
    public Dictionary<string, T[]> ApplyGates(
        Dictionary<string, T[]> globalParams,
        Dictionary<string, T[]> localParams)
    {
        if (_gates == null)
        {
            InitializeGates(globalParams);
        }

        var mixed = new Dictionary<string, T[]>(globalParams.Count);
        foreach (var layerName in globalParams.Keys)
        {
            double gate = _gates!.GetValueOrDefault(layerName, _gateInitValue);
            var gT = NumOps.FromDouble(gate);
            var oneMinusG = NumOps.FromDouble(1.0 - gate);

            var gp = globalParams[layerName];
            var lp = localParams.GetValueOrDefault(layerName, gp);
            var result = new T[gp.Length];

            for (int i = 0; i < gp.Length; i++)
            {
                result[i] = NumOps.Add(
                    NumOps.Multiply(lp[i], gT),
                    NumOps.Multiply(gp[i], oneMinusG));
            }

            mixed[layerName] = result;
        }

        return mixed;
    }

    /// <summary>
    /// Updates gate values based on validation performance.
    /// </summary>
    /// <param name="layerName">Layer to update gate for.</param>
    /// <param name="gradient">Gradient of validation loss w.r.t. the gate.</param>
    public void UpdateGate(string layerName, double gradient)
    {
        if (_gates == null)
        {
            return;
        }

        if (_gates.TryGetValue(layerName, out var currentGate))
        {
            double newGate = currentGate - _gateLearningRate * gradient;
            _gates[layerName] = Math.Max(0, Math.Min(1, newGate));
        }
    }

    /// <summary>
    /// Updates all gates simultaneously by comparing validation loss with global-only vs local-only parameters.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For each layer, we compare how well the global model and the
    /// local model perform. If the local model is better for that layer (lower loss), the gate
    /// should increase (use more local). If the global model is better, the gate should decrease.
    /// The gate moves proportionally to the loss difference, clipped to [0, 1].</para>
    /// </remarks>
    /// <param name="globalLossPerLayer">Validation loss when using global params for each layer.</param>
    /// <param name="localLossPerLayer">Validation loss when using local params for each layer.</param>
    public void UpdateGatesFromLosses(
        Dictionary<string, double> globalLossPerLayer,
        Dictionary<string, double> localLossPerLayer)
    {
        if (_gates == null)
        {
            return;
        }

        foreach (var layerName in _gates.Keys.ToArray())
        {
            if (globalLossPerLayer.TryGetValue(layerName, out double globalLoss) &&
                localLossPerLayer.TryGetValue(layerName, out double localLoss))
            {
                // Gradient: if local is better (lower loss), increase gate towards 1.
                // Normalized by the mean loss to make gradient scale-invariant.
                double meanLoss = (globalLoss + localLoss) / 2.0;
                double normalizedDiff = meanLoss > 1e-10
                    ? (globalLoss - localLoss) / meanLoss
                    : 0;

                double newGate = _gates[layerName] + _gateLearningRate * normalizedDiff;
                _gates[layerName] = Math.Max(0, Math.Min(1, newGate));
            }
        }
    }

    /// <summary>
    /// Computes the total gate regularization loss (L2 on gates to prevent extreme values).
    /// </summary>
    /// <param name="regularizationStrength">L2 penalty coefficient. Default: 0.01.</param>
    /// <returns>Gate regularization loss.</returns>
    public double ComputeGateRegularizationLoss(double regularizationStrength = 0.01)
    {
        if (_gates == null)
        {
            return 0;
        }

        double loss = 0;
        foreach (var gate in _gates.Values)
        {
            // Penalize deviation from 0.5 (neutral mixing). This prevents gates from collapsing
            // to 0 (fully global) or 1 (fully local), maintaining the benefit of mixing.
            double dev = gate - 0.5;
            loss += dev * dev;
        }

        return regularizationStrength * loss / Math.Max(1, _gates.Count);
    }

    /// <summary>Gets the gate values for all layers.</summary>
    public IReadOnlyDictionary<string, double>? Gates => _gates;

    /// <summary>Gets the initial gate value.</summary>
    public double GateInitValue => _gateInitValue;

    /// <summary>Gets the gate learning rate.</summary>
    public double GateLearningRate => _gateLearningRate;
}
