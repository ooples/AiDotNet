namespace AiDotNet.FederatedLearning.Alignment;

/// <summary>
/// Implements Federated DPO (Direct Preference Optimization) for reward-model-free LLM alignment.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> DPO is a simpler alternative to RLHF that skips the reward model
/// entirely. Instead of training a separate reward model and then using RL, DPO directly
/// optimizes the LLM to prefer good responses over bad ones using a binary cross-entropy loss
/// on preference pairs. Federated DPO lets each organization keep their preference data private
/// while collaboratively aligning the model.</para>
///
/// <para>DPO loss per preference pair (w, l):</para>
/// <code>
/// L = -log(sigmoid(beta * (log(pi(w)/ref(w)) - log(pi(l)/ref(l)))))
/// </code>
/// <para>where w is the preferred response, l is the dispreferred, and beta controls sharpness.</para>
///
/// <para>Reference: FedDPO: Federated Direct Preference Optimization (2024).
/// https://arxiv.org/abs/2404.18567</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FederatedDPO<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly FederatedDPOOptions _options;

    /// <summary>
    /// Creates a new Federated DPO instance.
    /// </summary>
    /// <param name="options">Configuration options. Uses defaults if null.</param>
    public FederatedDPO(FederatedDPOOptions? options = null)
    {
        _options = options ?? new FederatedDPOOptions();
    }

    /// <summary>
    /// Computes the DPO loss for a batch of preference pairs.
    /// </summary>
    /// <param name="policyChosenLogProbs">Log probs of chosen responses under current policy.</param>
    /// <param name="policyRejectedLogProbs">Log probs of rejected responses under current policy.</param>
    /// <param name="referenceChosenLogProbs">Log probs of chosen responses under reference model.</param>
    /// <param name="referenceRejectedLogProbs">Log probs of rejected responses under reference model.</param>
    /// <returns>Average DPO loss over the batch.</returns>
    public double ComputeDPOLoss(
        double[] policyChosenLogProbs,
        double[] policyRejectedLogProbs,
        double[] referenceChosenLogProbs,
        double[] referenceRejectedLogProbs)
    {
        ValidateLogProbArrays(policyChosenLogProbs, policyRejectedLogProbs,
            referenceChosenLogProbs, referenceRejectedLogProbs);
        int batchSize = policyChosenLogProbs.Length;
        if (batchSize == 0)
        {
            throw new ArgumentException("Batch cannot be empty.", nameof(policyChosenLogProbs));
        }

        double totalLoss = 0;
        for (int i = 0; i < batchSize; i++)
        {
            double chosenReward = policyChosenLogProbs[i] - referenceChosenLogProbs[i];
            double rejectedReward = policyRejectedLogProbs[i] - referenceRejectedLogProbs[i];
            double logit = _options.Beta * (chosenReward - rejectedReward);

            // -log(sigmoid(x)) = log(1 + exp(-x)) (numerically stable)
            totalLoss += logit >= 0
                ? Math.Log(1 + Math.Exp(-logit))
                : -logit + Math.Log(1 + Math.Exp(logit));
        }

        return totalLoss / batchSize;
    }

    /// <summary>
    /// Aggregates DPO-trained model updates from multiple clients.
    /// </summary>
    /// <param name="clientModels">Client model parameter dictionaries after local DPO training.</param>
    /// <param name="clientWeights">Per-client weights (proportional to preference pair count).</param>
    /// <returns>Aggregated model parameters.</returns>
    public Dictionary<string, T[]> AggregateModels(
        Dictionary<int, Dictionary<string, T[]>> clientModels,
        Dictionary<int, double> clientWeights)
    {
        Guard.NotNull(clientModels);
        Guard.NotNull(clientWeights);
        if (clientModels.Count == 0)
        {
            throw new ArgumentException("Client models cannot be empty.", nameof(clientModels));
        }

        var referenceModel = clientModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();
        double totalWeight = clientWeights.Values.Sum();
        if (totalWeight <= 0)
        {
            totalWeight = clientModels.Count;
        }

        var aggregated = new Dictionary<string, T[]>(referenceModel.Count, referenceModel.Comparer);
        foreach (var layerName in layerNames)
        {
            var result = new T[referenceModel[layerName].Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = NumOps.Zero;
            }

            aggregated[layerName] = result;
        }

        foreach (var (clientId, clientModel) in clientModels)
        {
            double w = clientWeights.GetValueOrDefault(clientId, 1.0);
            var normalizedWeight = NumOps.FromDouble(w / totalWeight);

            foreach (var layerName in layerNames)
            {
                var cp = clientModel[layerName];
                var rp = aggregated[layerName];
                for (int i = 0; i < rp.Length; i++)
                {
                    rp[i] = NumOps.Add(rp[i], NumOps.Multiply(cp[i], normalizedWeight));
                }
            }
        }

        return aggregated;
    }

    /// <summary>
    /// Computes per-example DPO gradients (as implicit reward margins) for parameter updates.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The DPO gradient for each preference pair depends on how
    /// "surprised" the model is by the correct preference ordering. If the model already
    /// strongly prefers the chosen response, the gradient is small (sigmoid is near 1). If it
    /// incorrectly prefers the rejected response, the gradient is large. This returns the
    /// per-example gradient weights that should multiply the log-prob gradients during backprop.</para>
    /// </remarks>
    /// <param name="policyChosenLogProbs">Log probs of chosen responses under current policy.</param>
    /// <param name="policyRejectedLogProbs">Log probs of rejected responses under current policy.</param>
    /// <param name="referenceChosenLogProbs">Log probs of chosen responses under reference model.</param>
    /// <param name="referenceRejectedLogProbs">Log probs of rejected responses under reference model.</param>
    /// <returns>Per-example gradient weights (negative sigmoid values for the loss gradient).</returns>
    public double[] ComputeDPOGradientWeights(
        double[] policyChosenLogProbs,
        double[] policyRejectedLogProbs,
        double[] referenceChosenLogProbs,
        double[] referenceRejectedLogProbs)
    {
        ValidateLogProbArrays(policyChosenLogProbs, policyRejectedLogProbs,
            referenceChosenLogProbs, referenceRejectedLogProbs);
        int batchSize = policyChosenLogProbs.Length;
        var weights = new double[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            double chosenReward = policyChosenLogProbs[i] - referenceChosenLogProbs[i];
            double rejectedReward = policyRejectedLogProbs[i] - referenceRejectedLogProbs[i];
            double logit = _options.Beta * (chosenReward - rejectedReward);

            // Gradient of DPO loss = -beta * sigmoid(-logit)
            // sigmoid(-x) = 1 / (1 + exp(x)) computed stably
            double sigmoidNeg = logit >= 0
                ? Math.Exp(-logit) / (1 + Math.Exp(-logit))
                : 1.0 / (1 + Math.Exp(logit));

            weights[i] = -_options.Beta * sigmoidNeg;
        }

        return weights;
    }

    /// <summary>
    /// Computes the implicit reward margin for each preference pair. Positive means the model
    /// correctly prefers the chosen response.
    /// </summary>
    /// <param name="policyChosenLogProbs">Log probs of chosen responses under current policy.</param>
    /// <param name="policyRejectedLogProbs">Log probs of rejected responses under current policy.</param>
    /// <param name="referenceChosenLogProbs">Log probs of chosen responses under reference model.</param>
    /// <param name="referenceRejectedLogProbs">Log probs of rejected responses under reference model.</param>
    /// <returns>Per-example reward margins.</returns>
    public double[] ComputeRewardMargins(
        double[] policyChosenLogProbs,
        double[] policyRejectedLogProbs,
        double[] referenceChosenLogProbs,
        double[] referenceRejectedLogProbs)
    {
        ValidateLogProbArrays(policyChosenLogProbs, policyRejectedLogProbs,
            referenceChosenLogProbs, referenceRejectedLogProbs);
        int batchSize = policyChosenLogProbs.Length;
        var margins = new double[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            double chosenReward = _options.Beta * (policyChosenLogProbs[i] - referenceChosenLogProbs[i]);
            double rejectedReward = _options.Beta * (policyRejectedLogProbs[i] - referenceRejectedLogProbs[i]);
            margins[i] = chosenReward - rejectedReward;
        }

        return margins;
    }

    /// <summary>
    /// Computes the accuracy of the model's implicit preference on a batch.
    /// Returns the fraction of pairs where the model assigns higher reward to the chosen response.
    /// </summary>
    /// <param name="policyChosenLogProbs">Log probs of chosen responses under current policy.</param>
    /// <param name="policyRejectedLogProbs">Log probs of rejected responses under current policy.</param>
    /// <param name="referenceChosenLogProbs">Log probs of chosen responses under reference model.</param>
    /// <param name="referenceRejectedLogProbs">Log probs of rejected responses under reference model.</param>
    /// <returns>Preference accuracy in [0, 1].</returns>
    public double ComputePreferenceAccuracy(
        double[] policyChosenLogProbs,
        double[] policyRejectedLogProbs,
        double[] referenceChosenLogProbs,
        double[] referenceRejectedLogProbs)
    {
        var margins = ComputeRewardMargins(
            policyChosenLogProbs, policyRejectedLogProbs,
            referenceChosenLogProbs, referenceRejectedLogProbs);

        int correct = 0;
        for (int i = 0; i < margins.Length; i++)
        {
            if (margins[i] > 0)
            {
                correct++;
            }
        }

        return margins.Length > 0 ? (double)correct / margins.Length : 0;
    }

    /// <summary>Gets the DPO configuration options.</summary>
    public FederatedDPOOptions Options => _options;

    private static void ValidateLogProbArrays(
        double[] policyChosen, double[] policyRejected,
        double[] referenceChosen, double[] referenceRejected)
    {
        Guard.NotNull(policyChosen);
        Guard.NotNull(policyRejected);
        Guard.NotNull(referenceChosen);
        Guard.NotNull(referenceRejected);

        if (policyChosen.Length != policyRejected.Length ||
            policyChosen.Length != referenceChosen.Length ||
            policyChosen.Length != referenceRejected.Length)
        {
            throw new ArgumentException(
                $"All log probability arrays must have equal length. Got chosen={policyChosen.Length}, " +
                $"rejected={policyRejected.Length}, refChosen={referenceChosen.Length}, refRejected={referenceRejected.Length}.");
        }
    }
}

/// <summary>
/// Configuration options for Federated DPO.
/// </summary>
public class FederatedDPOOptions
{
    /// <summary>
    /// Gets or sets the DPO temperature (beta). Higher values = sharper preference.
    /// Default: 0.1 per Rafailov et al.
    /// </summary>
    private double _beta = 0.1;
    private double _learningRate = 5e-7;
    private int _localEpochs = 1;
    private int _loraRank = 8;

    /// <summary>
    /// Gets or sets the DPO temperature (beta). Must be positive. Default: 0.1.
    /// </summary>
    public double Beta
    {
        get => _beta;
        set => _beta = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(value), "Beta must be positive.");
    }

    /// <summary>
    /// Gets or sets the learning rate for DPO training. Must be positive. Default: 5e-7.
    /// </summary>
    public double LearningRate
    {
        get => _learningRate;
        set => _learningRate = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(value), "Learning rate must be positive.");
    }

    /// <summary>
    /// Gets or sets the number of local DPO epochs per round. Must be at least 1. Default: 1.
    /// </summary>
    public int LocalEpochs
    {
        get => _localEpochs;
        set => _localEpochs = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(value), "LocalEpochs must be at least 1.");
    }

    /// <summary>
    /// Gets or sets whether to use LoRA for DPO training. Default: true.
    /// </summary>
    public bool UseLoRA { get; set; } = true;

    /// <summary>
    /// Gets or sets the LoRA rank if UseLoRA is true. Must be at least 1. Default: 8.
    /// </summary>
    public int LoRARank
    {
        get => _loraRank;
        set => _loraRank = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(value), "LoRA rank must be at least 1.");
    }
}
