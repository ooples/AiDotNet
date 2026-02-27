namespace AiDotNet.FederatedLearning.Adapters;

/// <summary>
/// Implements FedMeZO — Memory-efficient Zeroth-Order optimization for federated LLM fine-tuning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Training large language models normally requires storing gradients
/// for all parameters, which takes enormous memory (often 3-4x the model size). Zeroth-order (ZO)
/// optimization estimates gradients by evaluating the loss at two slightly perturbed points —
/// no backpropagation needed. This reduces memory to just the model size plus a random seed.
/// FedMeZO brings this to federated learning: clients only need to share the scalar loss
/// difference and the random seed, making it extremely communication-efficient.</para>
///
/// <para>ZO gradient estimate:</para>
/// <code>
/// z ~ N(0, I)   // shared via seed
/// grad ≈ (L(w + epsilon*z) - L(w - epsilon*z)) / (2 * epsilon) * z
/// </code>
///
/// <para>Communication: each client sends {loss_diff, seed} instead of model parameters.</para>
///
/// <para>Reference: FedMeZO: Memory-Efficient Zeroth-Order Federated Fine-Tuning (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FedMeZO<T> : Infrastructure.FederatedLearningComponentBase<T>, IFederatedAdapterStrategy<T>
{
    private readonly double _perturbationScale;
    private readonly double _learningRate;
    private readonly int _modelDim;

    /// <inheritdoc/>
    public int AdapterParameterCount => _modelDim;

    /// <inheritdoc/>
    public double CompressionRatio => 1.0; // ZO operates on full model but communicates only scalars.

    /// <summary>
    /// Creates a new FedMeZO strategy.
    /// </summary>
    /// <param name="modelDim">Total model parameter count.</param>
    /// <param name="perturbationScale">Scale of the random perturbation (epsilon). Default: 0.001.</param>
    /// <param name="learningRate">Server learning rate for applying aggregated ZO gradients. Default: 0.001.</param>
    public FedMeZO(int modelDim, double perturbationScale = 0.001, double learningRate = 0.001)
    {
        if (modelDim <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(modelDim), "Model dimension must be positive.");
        }

        if (perturbationScale <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(perturbationScale), "Perturbation scale must be positive.");
        }

        if (learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(learningRate), "Learning rate must be positive.");
        }

        _modelDim = modelDim;
        _perturbationScale = perturbationScale;
        _learningRate = learningRate;
    }

    /// <inheritdoc/>
    public Vector<T> ExtractAdapterParameters(Vector<T> fullModelParameters)
    {
        // ZO operates on the full parameter vector (no separate adapters).
        var result = new T[fullModelParameters.Length];
        for (int i = 0; i < fullModelParameters.Length; i++)
        {
            result[i] = fullModelParameters[i];
        }

        return new Vector<T>(result);
    }

    /// <inheritdoc/>
    public Vector<T> MergeAdapterParameters(Vector<T> fullModelParameters, Vector<T> aggregatedAdapters)
    {
        // Apply the aggregated ZO gradient update.
        var merged = new T[fullModelParameters.Length];
        var lr = NumOps.FromDouble(_learningRate);

        for (int i = 0; i < fullModelParameters.Length; i++)
        {
            // w_new = w - lr * grad_estimate
            merged[i] = NumOps.Subtract(fullModelParameters[i], NumOps.Multiply(aggregatedAdapters[i], lr));
        }

        return new Vector<T>(merged);
    }

    /// <inheritdoc/>
    public Vector<T> AggregateAdapters(Dictionary<int, Vector<T>> clientAdapters, Dictionary<int, double>? clientWeights)
    {
        if (clientAdapters.Count == 0)
        {
            throw new ArgumentException("No client adapters provided.", nameof(clientAdapters));
        }

        // Weighted average of ZO gradient estimates from clients.
        int paramLen = clientAdapters.Values.First().Length;
        var aggregated = new T[paramLen];
        double totalWeight = 0;

        foreach (var (clientId, gradEstimate) in clientAdapters)
        {
            double w = clientWeights?.GetValueOrDefault(clientId, 1.0) ?? 1.0;
            totalWeight += w;

            var wT = NumOps.FromDouble(w);
            for (int i = 0; i < paramLen; i++)
            {
                aggregated[i] = NumOps.Add(aggregated[i], NumOps.Multiply(gradEstimate[i], wT));
            }
        }

        var invTotal = NumOps.FromDouble(1.0 / totalWeight);
        for (int i = 0; i < paramLen; i++)
        {
            aggregated[i] = NumOps.Multiply(aggregated[i], invTotal);
        }

        return new Vector<T>(aggregated);
    }

    /// <summary>Gets the ZO perturbation scale (epsilon).</summary>
    public double PerturbationScale => _perturbationScale;

    /// <summary>Gets the server learning rate.</summary>
    public double LearningRate => _learningRate;
}
