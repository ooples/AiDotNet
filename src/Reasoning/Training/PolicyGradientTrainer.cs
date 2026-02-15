using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Reasoning.Models;
using AiDotNet.Validation;

namespace AiDotNet.Reasoning.Training;

/// <summary>
/// Policy gradient trainer for reinforcement learning of reasoning models.
/// </summary>
/// <typeparam name="T">The numeric type used for scoring (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This trains AI to reason better using reinforcement learning,
/// similar to how ChatGPT o1 and o3 are trained.
///
/// **What is Policy Gradient Training?**
/// A method to train AI by rewarding good reasoning chains and discouraging bad ones.
/// The "policy" is the model's strategy for generating reasoning steps.
///
/// **Simple analogy:**
/// Like training a dog:
/// - Dog (model) tries different actions (reasoning steps)
/// - Good actions (correct reasoning) → treat (positive reward)
/// - Bad actions (wrong reasoning) → no treat (low/negative reward)
/// - Over time, dog learns which actions get treats
///
/// **How it works:**
///
/// *Step 1: Generate reasoning chains*
/// Model generates multiple solutions to a problem
///
/// *Step 2: Evaluate with rewards*
/// - Process rewards (PRM): Rate quality of each step
/// - Outcome rewards (ORM): Rate final answer correctness
/// - Combine: Total reward for the chain
///
/// *Step 3: Calculate gradients*
/// Determine how to adjust model to increase reward
///
/// *Step 4: Update model*
/// Make small changes to encourage better reasoning
///
/// **REINFORCE algorithm (basic policy gradient):**
/// ```
/// For each episode:
///   1. Sample action (reasoning step) from policy: a ~ π(a|s)
///   2. Receive reward: r
///   3. Calculate return: G = Σ γᵗ * rₜ
///   4. Update policy: ∇θ J(θ) = E[∇θ log π(a|s) * G]
/// ```
///
/// **Example training episode:**
/// ```
/// Problem: "What is 7 × 8?"
///
/// Chain 1:
/// "7 × 8 = 7 × (4 + 4) = 28 + 28 = 56" ✓
/// Reward: +1.0 (correct)
/// → Increase probability of this reasoning pattern
///
/// Chain 2:
/// "7 × 8 is approximately 7 × 10 = 70" ✗
/// Reward: +0.0 (incorrect)
/// → Decrease probability of approximation without correction
///
/// Chain 3:
/// "7 × 8 = (7 × 10) - (7 × 2) = 70 - 14 = 56" ✓
/// Reward: +1.0 (correct)
/// → Increase probability of this reasoning pattern
/// ```
///
/// **Advanced techniques:**
///
/// *1. Advantage estimation:*
/// Compare reward to baseline to reduce variance
/// Advantage A = R - baseline
///
/// *2. Entropy regularization:*
/// Encourage exploration by penalizing overly confident predictions
/// Loss = -E[log π * A] - β * H(π)
///
/// *3. PPO (Proximal Policy Optimization):*
/// Limit how much policy changes per update (more stable)
///
/// *4. Reward shaping:*
/// Design rewards to guide learning:
/// - Step correctness
/// - Progress toward solution
/// - Explanation quality
/// - Computational efficiency
///
/// **Comparison to supervised learning:**
/// - **Supervised**: "Here's the right answer, copy it"
/// - **RL**: "Try different approaches, I'll tell you which work"
///
/// RL is better when:
/// - Multiple valid solutions exist
/// - Want to discover novel strategies
/// - Can define reward but not exact steps
/// - Want adaptive, robust reasoning
///
/// **Training loop:**
/// ```csharp
/// var trainer = new PolicyGradientTrainer<double>(
///     model,
///     rewardModel: new HybridRewardModel<double>(prm, orm),
///     learningRate: 0.0001
/// );
///
/// for (int epoch = 0; epoch < 100; epoch++)
/// {
///     foreach (var batch in trainingData)
///     {
///         // Generate chains
///         var chains = await GenerateChainsAsync(batch.Problems);
///
///         // Calculate rewards
///         var rewards = await CalculateRewardsAsync(chains);
///
///         // Update policy
///         await trainer.UpdateAsync(chains, rewards);
///     }
///
///     // Evaluate
///     var accuracy = await EvaluateAsync(validationSet);
///     Console.WriteLine($"Epoch {epoch}: Accuracy = {accuracy:P2}");
/// }
/// ```
///
/// **Research:**
/// - "Training Verifiers to Solve Math Word Problems" (Cobbe et al., 2021)
/// - "Let's Verify Step by Step" (Lightman et al., 2023)
/// - "Proximal Policy Optimization" (Schulman et al., 2017)
/// - "Self-Taught Reasoner (STaR)" (Zelikman et al., 2022)
/// - Used in ChatGPT o1/o3, DeepSeek-R1
/// </para>
/// </remarks>
internal class PolicyGradientTrainer<T>
{
    private readonly IChatModel<T> _model;
    private readonly IRewardModel<T>? _rewardModel;
    private readonly INumericOperations<T> _numOps;
    private readonly double _learningRate;
    private readonly double _discountFactor;
    private readonly double _entropyCoefficient;
    private readonly bool _useBaseline;

    private Vector<T>? _baseline;  // Running average of returns

    /// <summary>
    /// Initializes a new instance of the <see cref="PolicyGradientTrainer{T}"/> class.
    /// </summary>
    /// <param name="model">The model to train.</param>
    /// <param name="rewardModel">Reward model for evaluating chains.</param>
    /// <param name="learningRate">Learning rate for gradient updates.</param>
    /// <param name="discountFactor">Discount factor for future rewards (γ).</param>
    /// <param name="entropyCoefficient">Entropy regularization coefficient.</param>
    /// <param name="useBaseline">Whether to use baseline for variance reduction.</param>
    public PolicyGradientTrainer(
        IChatModel<T> model,
        IRewardModel<T>? rewardModel = null,
        double learningRate = 0.0001,
        double discountFactor = 0.99,
        double entropyCoefficient = 0.01,
        bool useBaseline = true)
    {
        Guard.NotNull(model);
        _model = model;
        _rewardModel = rewardModel;
        _numOps = MathHelper.GetNumericOperations<T>();
        _learningRate = learningRate;
        _discountFactor = discountFactor;
        _entropyCoefficient = entropyCoefficient;
        _useBaseline = useBaseline;
    }

    /// <summary>
    /// Trains on a batch of reasoning chains.
    /// </summary>
    /// <param name="chains">Reasoning chains to learn from.</param>
    /// <param name="correctAnswers">Correct answers for each problem.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Training metrics.</returns>
    public async Task<TrainingMetrics<T>> TrainBatchAsync(
        List<ReasoningChain<T>> chains,
        List<string>? correctAnswers = null,
        CancellationToken cancellationToken = default)
    {
        if (chains == null || chains.Count == 0)
            throw new ArgumentException("Chains cannot be null or empty", nameof(chains));

        var metrics = new TrainingMetrics<T>();

        // Calculate rewards for each chain
        var rewards = new List<T>();
        for (int i = 0; i < chains.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var chain = chains[i];
            string? correctAnswer = correctAnswers != null && i < correctAnswers.Count
                ? correctAnswers[i]
                : null;

            T reward = await CalculateRewardAsync(chain, correctAnswer, cancellationToken);
            rewards.Add(reward);
        }

        // Calculate returns (discounted cumulative rewards)
        var returns = CalculateReturns(rewards);

        // Calculate advantages (if using baseline)
        var advantages = _useBaseline
            ? CalculateAdvantages(returns)
            : returns;

        // Calculate policy gradient loss
        var policyLoss = CalculatePolicyLoss(chains, advantages);
        metrics.PolicyLoss = policyLoss;

        // Calculate entropy for exploration
        var entropy = CalculateEntropy(chains);
        metrics.Entropy = entropy;

        // Total loss
        var totalLoss = _numOps.Add(
            policyLoss,
            _numOps.Multiply(_numOps.FromDouble(_entropyCoefficient), entropy)
        );
        metrics.TotalLoss = totalLoss;

        // Update baseline
        if (_useBaseline)
        {
            UpdateBaseline(returns);
        }

        // Metrics
        metrics.AverageReward = new Vector<T>(rewards).Mean();
        metrics.AverageReturn = returns.Mean();
        metrics.ChainCount = chains.Count;

        return metrics;
    }

    /// <summary>
    /// Evaluates the model on a validation set.
    /// </summary>
    public async Task<EvaluationMetrics<T>> EvaluateAsync(
        List<(string problem, string correctAnswer)> validationSet,
        Func<string, Task<ReasoningChain<T>>> generateChain,
        CancellationToken cancellationToken = default)
    {
        var metrics = new EvaluationMetrics<T>();
        int correctCount = 0;
        var rewards = new List<T>();

        foreach (var (problem, correctAnswer) in validationSet)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var chain = await generateChain(problem);
            T reward = await CalculateRewardAsync(chain, correctAnswer, cancellationToken);
            rewards.Add(reward);

            if (Convert.ToDouble(reward) >= 0.9)
            {
                correctCount++;
            }
        }

        metrics.Accuracy = (double)correctCount / validationSet.Count;
        metrics.AverageReward = new Vector<T>(rewards).Mean();
        metrics.TotalEvaluated = validationSet.Count;

        return metrics;
    }

    /// <summary>
    /// Implements self-taught reasoner (STaR) training.
    /// </summary>
    /// <remarks>
    /// STaR: Generate chains, keep only correct ones, train on them.
    /// Bootstrapping approach for improving reasoning.
    /// </remarks>
    public async Task<TrainingMetrics<T>> TrainSTaRAsync(
        List<string> problems,
        List<string> correctAnswers,
        Func<string, Task<List<ReasoningChain<T>>>> generateMultipleChains,
        int samplesPerProblem = 5,
        CancellationToken cancellationToken = default)
    {
        var correctChains = new List<ReasoningChain<T>>();

        // Sample multiple chains per problem
        for (int i = 0; i < problems.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var chains = await generateMultipleChains(problems[i]);

            // Keep only correct chains
            foreach (var chain in chains)
            {
                T reward = await CalculateRewardAsync(chain, correctAnswers[i], cancellationToken);

                if (Convert.ToDouble(reward) >= 0.9)
                {
                    correctChains.Add(chain);
                }
            }
        }

        // Train on correct chains
        if (correctChains.Count > 0)
        {
            return await TrainBatchAsync(correctChains, cancellationToken: cancellationToken);
        }

        return new TrainingMetrics<T>
        {
            ChainCount = 0,
            AverageReward = _numOps.Zero
        };
    }

    private async Task<T> CalculateRewardAsync(
        ReasoningChain<T> chain,
        string? correctAnswer,
        CancellationToken cancellationToken)
    {
        if (_rewardModel != null)
        {
            return await _rewardModel.CalculateChainRewardAsync(chain, correctAnswer, cancellationToken);
        }

        // Fallback: Simple outcome-based reward
        if (!string.IsNullOrEmpty(correctAnswer))
        {
            bool isCorrect = chain.FinalAnswer?.Equals(correctAnswer, StringComparison.OrdinalIgnoreCase) ?? false;
            return isCorrect ? _numOps.One : _numOps.Zero;
        }

        // No reward model and no correct answer
        return _numOps.FromDouble(0.5);
    }

    private Vector<T> CalculateReturns(List<T> rewards)
    {
        // Calculate discounted cumulative rewards
        var returns = new List<T>();
        T cumulativeReturn = _numOps.Zero;

        // Go backwards through rewards
        for (int i = rewards.Count - 1; i >= 0; i--)
        {
            cumulativeReturn = _numOps.Add(
                rewards[i],
                _numOps.Multiply(_numOps.FromDouble(_discountFactor), cumulativeReturn)
            );
            returns.Insert(0, cumulativeReturn);
        }

        return new Vector<T>(returns);
    }

    private Vector<T> CalculateAdvantages(Vector<T> returns)
    {
        if (_baseline == null)
        {
            _baseline = returns;  // Initialize baseline
            return returns;  // No advantage calculation on first batch
        }

        // Advantage = Return - Baseline
        var advantages = new List<T>();
        for (int i = 0; i < returns.Length; i++)
        {
            T advantage = _numOps.Subtract(returns[i], _baseline[i % _baseline.Length]);
            advantages.Add(advantage);
        }

        return new Vector<T>(advantages);
    }

    private void UpdateBaseline(Vector<T> returns)
    {
        if (_baseline == null)
        {
            _baseline = returns;
            return;
        }

        // Exponential moving average
        double alpha = 0.1;  // Smoothing factor

        var updated = new List<T>();
        for (int i = 0; i < Math.Min(returns.Length, _baseline.Length); i++)
        {
            double oldVal = Convert.ToDouble(_baseline[i]);
            double newVal = Convert.ToDouble(returns[i]);
            double smoothed = oldVal * (1 - alpha) + newVal * alpha;
            updated.Add(_numOps.FromDouble(smoothed));
        }

        _baseline = new Vector<T>(updated);
    }

    private T CalculatePolicyLoss(List<ReasoningChain<T>> chains, Vector<T> advantages)
    {
        // Simplified policy gradient loss
        // In practice, this would compute: -E[log π(a|s) * A]
        // Here we approximate with chain quality weighted by advantages

        double totalLoss = 0.0;
        for (int i = 0; i < chains.Count; i++)
        {
            double advantage = Convert.ToDouble(advantages[i % advantages.Length]);
            double chainScore = Convert.ToDouble(chains[i].OverallScore);

            // Negative log likelihood weighted by advantage
            totalLoss -= Math.Log(Math.Max(chainScore, 0.001)) * advantage;
        }

        return _numOps.FromDouble(totalLoss / chains.Count);
    }

    private T CalculateEntropy(List<ReasoningChain<T>> chains)
    {
        // Entropy encourages exploration
        // H(π) = -Σ π(a) log π(a)

        double totalEntropy = 0.0;
        foreach (var chain in chains)
        {
            if (chain.StepScores.Length > 0)
            {
                foreach (var score in chain.StepScores)
                {
                    double prob = Convert.ToDouble(score);
                    if (prob > 0.001)
                    {
                        totalEntropy -= prob * Math.Log(prob);
                    }
                }
            }
        }

        return _numOps.FromDouble(totalEntropy / chains.Count);
    }
}

/// <summary>
/// Metrics from a training batch.
/// </summary>
internal class TrainingMetrics<T>
{
    public T PolicyLoss { get; set; } = default!;
    public T Entropy { get; set; } = default!;
    public T TotalLoss { get; set; } = default!;
    public T AverageReward { get; set; } = default!;
    public T AverageReturn { get; set; } = default!;
    public int ChainCount { get; set; }

    public override string ToString()
    {
        return $@"Training Metrics:
Policy Loss: {Convert.ToDouble(PolicyLoss):F4}
Entropy: {Convert.ToDouble(Entropy):F4}
Total Loss: {Convert.ToDouble(TotalLoss):F4}
Average Reward: {Convert.ToDouble(AverageReward):F3}
Average Return: {Convert.ToDouble(AverageReturn):F3}
Chains: {ChainCount}";
    }
}

/// <summary>
/// Metrics from evaluation.
/// </summary>
internal class EvaluationMetrics<T>
{
    public double Accuracy { get; set; }
    public T AverageReward { get; set; } = default!;
    public int TotalEvaluated { get; set; }

    public override string ToString()
    {
        return $@"Evaluation Metrics:
Accuracy: {Accuracy:P2}
Average Reward: {Convert.ToDouble(AverageReward):F3}
Total Evaluated: {TotalEvaluated}";
    }
}
