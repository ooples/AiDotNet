namespace AiDotNet.FederatedLearning.Alignment;

/// <summary>
/// Configuration and orchestration for Federated RLHF (Reinforcement Learning from Human Feedback).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> RLHF is how modern LLMs learn to be helpful, harmless, and honest.
/// Normally, human feedback data is centralized. Federated RLHF keeps the feedback private at each
/// organization: each client trains a local reward model on their preference data, and the server
/// aggregates these reward models. The policy (LLM) is then fine-tuned using the aggregated reward
/// signal via PPO or similar RL algorithms.</para>
///
/// <para>Pipeline:</para>
/// <list type="number">
/// <item>Each client collects human preference data (chosen/rejected response pairs)</item>
/// <item>Clients train local reward models on their preferences</item>
/// <item>Server aggregates reward models via FedAvg</item>
/// <item>Server fine-tunes the LLM policy using aggregated reward (PPO)</item>
/// <item>Updated policy is distributed to clients</item>
/// </list>
///
/// <para>Reference: Federated RLHF for Privacy-Preserving LLM Alignment (2024).</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class FederatedRLHF<T> : Infrastructure.FederatedLearningComponentBase<T>
{
    private readonly FederatedRLHFOptions _options;

    /// <summary>
    /// Creates a new Federated RLHF orchestrator.
    /// </summary>
    /// <param name="options">Configuration options. Uses defaults if null.</param>
    public FederatedRLHF(FederatedRLHFOptions? options = null)
    {
        _options = options ?? new FederatedRLHFOptions();
    }

    /// <summary>
    /// Aggregates reward model parameters from multiple clients.
    /// </summary>
    /// <param name="clientRewardModels">Client reward model parameter dictionaries.</param>
    /// <param name="clientWeights">Per-client weights (typically proportional to feedback count).</param>
    /// <returns>Aggregated reward model parameters.</returns>
    public Dictionary<string, T[]> AggregateRewardModels(
        Dictionary<int, Dictionary<string, T[]>> clientRewardModels,
        Dictionary<int, double> clientWeights)
    {
        if (clientRewardModels == null || clientRewardModels.Count == 0)
        {
            throw new ArgumentException("Client reward models cannot be null or empty.", nameof(clientRewardModels));
        }

        // Weighted average of reward model parameters.
        var referenceModel = clientRewardModels.First().Value;
        var layerNames = referenceModel.Keys.ToArray();
        double totalWeight = clientWeights.Values.Sum();

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

        foreach (var (clientId, rewardModel) in clientRewardModels)
        {
            double w = clientWeights.GetValueOrDefault(clientId, 1.0);
            var normalizedWeight = NumOps.FromDouble(w / totalWeight);

            foreach (var layerName in layerNames)
            {
                var cp = rewardModel[layerName];
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
    /// Computes the KL penalty for PPO-style policy updates.
    /// </summary>
    /// <param name="policyLogProbs">Log probabilities under the new policy.</param>
    /// <param name="referenceLogProbs">Log probabilities under the reference policy.</param>
    /// <returns>KL divergence penalty value.</returns>
    public double ComputeKLPenalty(double[] policyLogProbs, double[] referenceLogProbs)
    {
        if (policyLogProbs.Length != referenceLogProbs.Length)
        {
            throw new ArgumentException("Log probability arrays must have the same length.");
        }

        double kl = 0;
        for (int i = 0; i < policyLogProbs.Length; i++)
        {
            kl += Math.Exp(policyLogProbs[i]) * (policyLogProbs[i] - referenceLogProbs[i]);
        }

        return _options.KLCoefficient * kl / policyLogProbs.Length;
    }

    /// <summary>
    /// Computes GAE (Generalized Advantage Estimation) for PPO training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> In RL, the "advantage" tells us how much better an action was
    /// compared to the average. GAE smooths this estimate by combining multiple time steps with
    /// an exponential decay (lambda). This is critical for stable PPO training — without good
    /// advantage estimates, the policy update direction would be too noisy.</para>
    /// </remarks>
    /// <param name="rewards">Per-token rewards from the reward model (includes KL penalty).</param>
    /// <param name="values">Value function estimates for each token position.</param>
    /// <param name="gamma">Discount factor. Default: 1.0 (no discounting for single-response RLHF).</param>
    /// <param name="lambda">GAE smoothing parameter. Default: 0.95.</param>
    /// <returns>Advantage estimates for each token position.</returns>
    public double[] ComputeGAE(double[] rewards, double[] values, double gamma = 1.0, double lambda = 0.95)
    {
        int n = rewards.Length;
        if (n == 0)
        {
            return [];
        }

        if (values.Length != n + 1)
        {
            throw new ArgumentException(
                $"Values must have length {n + 1} (rewards.Length + 1 for bootstrap). Got {values.Length}.");
        }

        var advantages = new double[n];
        double gae = 0;

        for (int t = n - 1; t >= 0; t--)
        {
            double delta = rewards[t] + gamma * values[t + 1] - values[t];
            gae = delta + gamma * lambda * gae;
            advantages[t] = gae;
        }

        return advantages;
    }

    /// <summary>
    /// Computes the PPO clipped surrogate loss for a batch of tokens.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> PPO limits how much the policy changes in one step by "clipping"
    /// the probability ratio. If the new policy is too different from the old one (ratio outside
    /// [1-epsilon, 1+epsilon]), the gradient is zeroed out. This prevents catastrophic policy updates
    /// that could destabilize training.</para>
    /// </remarks>
    /// <param name="logProbsNew">Log probabilities under the new (updated) policy.</param>
    /// <param name="logProbsOld">Log probabilities under the old (data-collection) policy.</param>
    /// <param name="advantages">Advantage estimates from ComputeGAE.</param>
    /// <returns>The clipped surrogate loss (to be minimized).</returns>
    public double ComputePPOLoss(double[] logProbsNew, double[] logProbsOld, double[] advantages)
    {
        int n = logProbsNew.Length;
        if (n == 0)
        {
            return 0;
        }

        double totalLoss = 0;
        double clipRange = _options.ClipRange;

        for (int i = 0; i < n; i++)
        {
            double ratio = Math.Exp(logProbsNew[i] - logProbsOld[i]);
            double clippedRatio = Math.Max(1 - clipRange, Math.Min(1 + clipRange, ratio));

            // PPO objective: min(ratio * A, clip(ratio) * A)
            double surr1 = ratio * advantages[i];
            double surr2 = clippedRatio * advantages[i];
            totalLoss -= Math.Min(surr1, surr2); // Negate because we minimize loss.
        }

        return totalLoss / n;
    }

    /// <summary>
    /// Computes per-token rewards by combining reward model scores with KL penalty.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The final reward for each response combines two signals:
    /// (1) the reward model's score (how good is this response?) minus
    /// (2) a KL penalty (how far did we drift from the original model?).
    /// The KL penalty prevents "reward hacking" — where the model finds ways to exploit the
    /// reward model rather than genuinely improving.</para>
    /// </remarks>
    /// <param name="rewardModelScores">Scores from the reward model per token/response.</param>
    /// <param name="policyLogProbs">Log probs under current policy.</param>
    /// <param name="referenceLogProbs">Log probs under reference model.</param>
    /// <returns>Per-token rewards with KL penalty applied.</returns>
    public double[] ComputeRewardsWithKLPenalty(
        double[] rewardModelScores,
        double[] policyLogProbs,
        double[] referenceLogProbs)
    {
        int n = rewardModelScores.Length;
        var rewards = new double[n];
        double beta = _options.KLCoefficient;

        for (int i = 0; i < n; i++)
        {
            double klPerToken = policyLogProbs[i] - referenceLogProbs[i];
            rewards[i] = rewardModelScores[i] - beta * klPerToken;
        }

        return rewards;
    }

    /// <summary>Gets the RLHF configuration options.</summary>
    public FederatedRLHFOptions Options => _options;
}

/// <summary>
/// Configuration options for Federated RLHF.
/// </summary>
public class FederatedRLHFOptions
{
    /// <summary>
    /// Gets or sets the KL divergence penalty coefficient (beta). Default: 0.1.
    /// </summary>
    /// <remarks>Controls how far the policy can deviate from the reference model.</remarks>
    public double KLCoefficient { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the number of PPO epochs per round. Default: 4.
    /// </summary>
    public int PPOEpochs { get; set; } = 4;

    /// <summary>
    /// Gets or sets the PPO clip range. Default: 0.2.
    /// </summary>
    public double ClipRange { get; set; } = 0.2;

    /// <summary>
    /// Gets or sets the reward model learning rate. Default: 1e-5.
    /// </summary>
    public double RewardModelLearningRate { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether to use adapter-only reward models. Default: true.
    /// </summary>
    /// <remarks>When true, only LoRA adapters of the reward model are aggregated.</remarks>
    public bool AdapterOnlyRewardModel { get; set; } = true;
}
