using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Validation;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Group Relative Policy Optimization (GRPO) for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// GRPO is a memory-efficient reinforcement learning algorithm developed by DeepSeek
/// that eliminates the need for a separate critic model, reducing memory requirements by ~50%.
/// </para>
/// <para><b>For Beginners:</b> GRPO is like PPO but more efficient. Instead of training a
/// separate critic model to estimate value, GRPO generates multiple responses for each prompt
/// and uses the group's average reward as the baseline. This makes training faster and
/// uses less GPU memory.
/// </para>
/// <para>
/// Key features of GRPO:
/// 1. No critic model needed (saves ~50% memory)
/// 2. Group-based advantage estimation
/// 3. Works well with verifiable rewards (RLVR)
/// 4. Used in DeepSeekMath and DeepSeek-R1
/// </para>
/// <para>
/// The GRPO advantage is computed as:
/// A_i = (r_i - mean(r_group)) / std(r_group)
/// where r_group are the rewards for all responses in the group.
/// </para>
/// <para>
/// Original paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
/// by Shao et al. (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class GroupRelativePolicyOptimization<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    private IFullModel<T, TInput, TOutput>? _referenceModel;
    private Func<TInput, TOutput, double>? _rewardFunction;

    /// <summary>
    /// Initializes a new instance of GRPO fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public GroupRelativePolicyOptimization(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.GRPO)
        {
            options.MethodType = FineTuningMethodType.GRPO;
        }
    }

    /// <summary>
    /// Sets the reward function for evaluating responses.
    /// </summary>
    /// <param name="rewardFunction">A function that takes (input, output) and returns a reward score.</param>
    public void SetRewardFunction(Func<TInput, TOutput, double> rewardFunction)
    {
        Guard.NotNull(rewardFunction);
        _rewardFunction = rewardFunction;
    }

    /// <inheritdoc/>
    public override string MethodName => "GRPO";

    /// <inheritdoc/>
    public override FineTuningCategory Category => FineTuningCategory.ReinforcementLearning;

    /// <inheritdoc/>
    public override bool RequiresRewardModel => true;

    /// <inheritdoc/>
    public override bool RequiresReferenceModel => true;

    /// <inheritdoc/>
    public override async Task<IFullModel<T, TInput, TOutput>> FineTuneAsync(
        IFullModel<T, TInput, TOutput> baseModel,
        FineTuningData<T, TInput, TOutput> trainingData,
        CancellationToken cancellationToken = default)
    {
        if (baseModel == null)
        {
            throw new ArgumentNullException(nameof(baseModel));
        }

        ValidateTrainingData(trainingData);

        if (_rewardFunction == null && !trainingData.HasRLData)
        {
            throw new InvalidOperationException(
                "GRPO requires either a reward function (call SetRewardFunction) or pre-computed rewards in training data.");
        }

        // Clone the base model as the reference (frozen during training)
        _referenceModel = baseModel.Clone();

        // Clone the base model as the policy (will be trained)
        var policyModel = baseModel.Clone();

        CurrentMetrics = new FineTuningMetrics<T>
        {
            MethodName = MethodName,
            TrainingStartTime = DateTime.UtcNow
        };

        var groupSize = Options.GRPOGroupSize;
        var klCoeff = Options.KLCoefficient;
        var clipRange = Options.PPOClipRange;
        var totalSteps = Options.Epochs * (trainingData.Count / Options.BatchSize);
        var currentStep = 0;

        for (int epoch = 0; epoch < Options.Epochs; epoch++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            foreach (var batch in CreateBatches(trainingData, Options.BatchSize))
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                var (batchLoss, avgReward, avgKL) = await ComputeGRPOLossAndUpdateAsync(
                    policyModel, batch, groupSize, klCoeff, clipRange, Options.LearningRate, cancellationToken);
                currentStep++;

                UpdateMetrics(batchLoss, currentStep);
                CurrentMetrics.AverageReward = avgReward;
                CurrentMetrics.KLDivergence = avgKL;

                LogProgress(currentStep, totalSteps, batchLoss,
                    $"reward={avgReward:F3}, KL={avgKL:F4}");
            }
        }

        CurrentMetrics.TrainingEndTime = DateTime.UtcNow;
        CurrentMetrics.TrainingTimeSeconds = (CurrentMetrics.TrainingEndTime - CurrentMetrics.TrainingStartTime).TotalSeconds;

        return policyModel;
    }

    /// <inheritdoc/>
    public override async Task<FineTuningMetrics<T>> EvaluateAsync(
        IFullModel<T, TInput, TOutput> model,
        FineTuningData<T, TInput, TOutput> evaluationData,
        CancellationToken cancellationToken = default)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (evaluationData == null)
        {
            throw new ArgumentNullException(nameof(evaluationData));
        }

        var metrics = new FineTuningMetrics<T>
        {
            MethodName = MethodName
        };

        if (_rewardFunction == null && !evaluationData.HasRLData)
        {
            return metrics;
        }

        double totalReward = 0.0;
        var rewards = new List<double>();

        for (int i = 0; i < evaluationData.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = evaluationData.Inputs[i];
            var output = model.Predict(input);

            double reward = _rewardFunction != null
                ? _rewardFunction(input, output)
                : evaluationData.Rewards[i];

            rewards.Add(reward);
            totalReward += reward;
        }

        metrics.AverageReward = evaluationData.Count > 0 ? totalReward / evaluationData.Count : 0.0;
        metrics.RewardStd = rewards.Count > 1
            ? Math.Sqrt(rewards.Select(r => Math.Pow(r - metrics.AverageReward, 2)).Sum() / (rewards.Count - 1))
            : 0.0;

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Computes the GRPO loss for a batch and updates model parameters.
    /// </summary>
    /// <returns>A tuple of (loss, average reward, average KL divergence).</returns>
    private async Task<(double Loss, double AvgReward, double AvgKL)> ComputeGRPOLossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        int groupSize,
        double klCoeff,
        double clipRange,
        double learningRate,
        CancellationToken cancellationToken)
    {
        if (_referenceModel == null)
        {
            throw new InvalidOperationException("Reference model not initialized.");
        }

        double totalLoss = 0.0;
        double totalReward = 0.0;
        double totalKL = 0.0;
        double totalGroupVariance = 0.0;
        int sampleCount = 0;
        int groupCount = 0;

        for (int i = 0; i < batch.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = batch.Inputs[i];

            // Generate group of responses
            var groupRewards = new List<double>();
            var groupLogProbs = new List<double>();
            var groupRefLogProbs = new List<double>();
            var groupResponses = new List<TOutput>();

            for (int g = 0; g < groupSize; g++)
            {
                // Generate response - use batch outputs if available for diversity,
                // otherwise fall back to model prediction (note: deterministic prediction
                // produces identical responses; proper GRPO requires temperature sampling)
                TOutput response = batch.HasPairwisePreferenceData && g < 2
                    ? (g == 0 ? batch.ChosenOutputs[i] : batch.RejectedOutputs[i])
                    : policyModel.Predict(input);

                groupResponses.Add(response);

                // Compute reward
                double reward = _rewardFunction != null
                    ? _rewardFunction(input, response)
                    : (batch.HasRLData && i < batch.Rewards.Length)
                        ? batch.Rewards[i]
                        : 0.0;
                groupRewards.Add(reward);

                // Compute log probabilities
                var logProb = ComputeLogProbability(policyModel, input, response);
                var refLogProb = ComputeLogProbability(_referenceModel, input, response);
                groupLogProbs.Add(logProb);
                groupRefLogProbs.Add(refLogProb);
            }

            // Compute group statistics for advantage estimation
            var meanReward = groupRewards.Average();
            var stdReward = groupRewards.Count > 1
                ? Math.Sqrt(groupRewards.Select(r => Math.Pow(r - meanReward, 2)).Sum() / (groupRewards.Count - 1))
                : 1.0;

            // Avoid division by zero
            stdReward = stdReward < 1e-8 ? 1.0 : stdReward;

            // Compute GRPO loss for each response in the group
            for (int g = 0; g < groupSize; g++)
            {
                // Normalized advantage
                var advantage = (groupRewards[g] - meanReward) / stdReward;

                // Log probability ratio
                var logRatio = groupLogProbs[g] - groupRefLogProbs[g];
                var ratio = Math.Exp(logRatio);

                // Clipped surrogate objective (PPO-style)
                var clippedRatio = Math.Max(Math.Min(ratio, 1.0 + clipRange), 1.0 - clipRange);
                var surrogateObjective = Math.Min(ratio * advantage, clippedRatio * advantage);

                // KL penalty
                var kl = logRatio;
                totalKL += kl;

                // GRPO loss: -surrogate + kl_coeff * kl
                var loss = -surrogateObjective + klCoeff * kl;
                totalLoss += loss;

                // Compute and apply gradients scaled by capped advantage
                var cappedAdvantage = Math.Max(-1.0, Math.Min(1.0, advantage));
                var scaledLearningRate = learningRate * cappedAdvantage;
                var gradients = policyModel.ComputeGradients(input, groupResponses[g]);
                policyModel.ApplyGradients(gradients, NumOps.FromDouble(scaledLearningRate));

                totalReward += groupRewards[g];
                sampleCount++;
            }

            // Accumulate group variance
            totalGroupVariance += stdReward * stdReward;
            groupCount++;
        }

        // Set average group variance for metrics
        CurrentMetrics.GroupRewardVariance = groupCount > 0 ? totalGroupVariance / groupCount : 0.0;

        var avgLoss = sampleCount > 0 ? totalLoss / sampleCount : 0.0;
        var avgReward = sampleCount > 0 ? totalReward / sampleCount : 0.0;
        var avgKL = sampleCount > 0 ? totalKL / sampleCount : 0.0;

        return await Task.FromResult((avgLoss, avgReward, avgKL));
    }
}
