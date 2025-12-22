using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Reinforcement Learning from Human Feedback (RLHF) with PPO for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// RLHF is the foundational approach for aligning language models with human preferences.
/// It uses a reward model trained on human feedback to guide policy optimization via PPO.
/// </para>
/// <para><b>For Beginners:</b> RLHF is like training a model with a coach (reward model)
/// that tells it how good its responses are. The model learns to generate responses
/// that the coach rates highly, while not straying too far from its original behavior.</para>
/// <para>
/// RLHF pipeline:
/// 1. Train a reward model on human preference data
/// 2. Use PPO to optimize the policy against the reward model
/// 3. Add KL penalty to prevent reward hacking
/// </para>
/// <para>
/// Original paper: "Training language models to follow instructions with human feedback"
/// by Ouyang et al. (2022) - InstructGPT
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class ReinforcementLearningHumanFeedback<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    private IFullModel<T, TInput, TOutput>? _referenceModel;
    private IFullModel<T, TInput, TOutput>? _valueModel;
    private Func<TInput, TOutput, double>? _rewardFunction;

    /// <summary>
    /// Initializes a new instance of RLHF fine-tuning with PPO.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public ReinforcementLearningHumanFeedback(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.RLHF &&
            options.MethodType != FineTuningMethodType.PPO)
        {
            options.MethodType = FineTuningMethodType.RLHF;
        }
    }

    /// <summary>
    /// Sets the reward function (or reward model) for evaluating responses.
    /// </summary>
    /// <param name="rewardFunction">A function that takes (input, output) and returns a reward score.</param>
    public void SetRewardFunction(Func<TInput, TOutput, double> rewardFunction)
    {
        _rewardFunction = rewardFunction ?? throw new ArgumentNullException(nameof(rewardFunction));
    }

    /// <summary>
    /// Sets the value model for PPO advantage estimation.
    /// </summary>
    /// <param name="valueModel">The value model for critic.</param>
    public void SetValueModel(IFullModel<T, TInput, TOutput> valueModel)
    {
        _valueModel = valueModel ?? throw new ArgumentNullException(nameof(valueModel));
    }

    /// <inheritdoc/>
    public override string MethodName => "RLHF-PPO";

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
                "RLHF requires either a reward function (call SetRewardFunction) or pre-computed rewards in training data.");
        }

        // Clone the base model as the reference (frozen during training)
        _referenceModel = baseModel.Clone();

        // Clone the base model as the policy (will be trained)
        var policyModel = baseModel.Clone();

        // Initialize value model if not provided
        _valueModel ??= baseModel.Clone();

        CurrentMetrics = new FineTuningMetrics<T>
        {
            MethodName = MethodName,
            TrainingStartTime = DateTime.UtcNow
        };

        var klCoeff = Options.KLCoefficient;
        var clipRange = Options.PPOClipRange;
        var valueCoeff = Options.ValueCoefficient;
        var entropyCoeff = Options.EntropyCoefficient;
        var gaeLambda = Options.GAELambda;
        var gamma = Options.Gamma;
        var ppoEpochs = Options.PPOEpochsPerBatch;

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

                // Collect experience
                var experience = CollectExperience(policyModel, batch);

                // Compute advantages using GAE
                ComputeAdvantages(experience, gamma, gaeLambda);

                // PPO update for multiple epochs per batch
                double batchLoss = 0.0;
                for (int ppoEpoch = 0; ppoEpoch < ppoEpochs; ppoEpoch++)
                {
                    batchLoss = await ComputePPOLossAndUpdateAsync(
                        policyModel, batch, experience, clipRange, klCoeff, valueCoeff, entropyCoeff, cancellationToken);
                }

                currentStep++;
                UpdateMetrics(batchLoss, currentStep);

                var avgReward = experience.Sum(e => e.Reward) / Math.Max(experience.Count, 1);
                var avgKL = experience.Sum(e => e.KL) / Math.Max(experience.Count, 1);

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
    /// Collects experience (states, actions, rewards) from the policy.
    /// </summary>
    private List<PPOExperience> CollectExperience(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch)
    {
        var experience = new List<PPOExperience>();

        for (int i = 0; i < batch.Count; i++)
        {
            var input = batch.Inputs[i];
            var output = policyModel.Predict(input);

            // Compute reward
            double reward = _rewardFunction != null
                ? _rewardFunction(input, output)
                : (batch.HasRLData && i < batch.Rewards.Length)
                    ? batch.Rewards[i]
                    : 0.0;

            // Compute log probabilities
            var logProb = ComputeLogProbability(policyModel, input, output);
            var refLogProb = _referenceModel != null
                ? ComputeLogProbability(_referenceModel, input, output)
                : logProb;

            // Compute value estimate
            double value = 0.0;
            if (_valueModel != null)
            {
                // Use value model to estimate state value
                // Convert the value model's output to a scalar value estimate
                var valueOutput = _valueModel.Predict(input);
                var valueVector = ConversionsHelper.ConvertToVector<T, TOutput>(valueOutput);
                if (valueVector.Length > 0)
                {
                    // Use mean of output vector as scalar value estimate
                    value = NumOps.ToDouble(valueVector.Average());
                }
            }

            // KL divergence from reference
            var kl = logProb - refLogProb;

            experience.Add(new PPOExperience
            {
                InputIndex = i,
                Output = output,
                Reward = reward,
                LogProb = logProb,
                RefLogProb = refLogProb,
                Value = value,
                KL = kl,
                Advantage = 0.0 // Will be computed later
            });
        }

        return experience;
    }

    /// <summary>
    /// Computes advantages using Generalized Advantage Estimation (GAE).
    /// </summary>
    private void ComputeAdvantages(List<PPOExperience> experience, double gamma, double gaeLambda)
    {
        // For language model fine-tuning, we typically have single-step episodes
        // So GAE simplifies to: A = R - V
        for (int i = experience.Count - 1; i >= 0; i--)
        {
            var exp = experience[i];

            // Simple advantage: reward - value
            // In multi-step, we would compute temporal difference errors
            exp.Advantage = exp.Reward - exp.Value;

            // Normalize advantages for stability
            experience[i] = exp;
        }

        // Normalize advantages across batch
        if (experience.Count > 1)
        {
            var mean = experience.Average(e => e.Advantage);
            var std = Math.Sqrt(experience.Select(e => Math.Pow(e.Advantage - mean, 2)).Sum() / (experience.Count - 1));

            if (std > 1e-8)
            {
                for (int i = 0; i < experience.Count; i++)
                {
                    var exp = experience[i];
                    exp.Advantage = (exp.Advantage - mean) / std;
                    experience[i] = exp;
                }
            }
        }
    }

    /// <summary>
    /// Computes PPO loss and updates model parameters.
    /// </summary>
    private async Task<double> ComputePPOLossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        List<PPOExperience> experience,
        double clipRange,
        double klCoeff,
        double valueCoeff,
        double entropyCoeff,
        CancellationToken cancellationToken)
    {
        double totalPolicyLoss = 0.0;
        double totalValueLoss = 0.0;
        double totalEntropyLoss = 0.0;

        foreach (var exp in experience)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            // Recompute current log probability of the SAME ACTION (exp.Output) under the updated policy
            // This is essential for PPO - we compare probability of the same action under old vs current policy
            var input = batch.Inputs[exp.InputIndex];
            var currentLogProb = ComputeLogProbability(policyModel, input, exp.Output);

            // Probability ratio between current and old policy for the SAME action
            var logRatio = currentLogProb - exp.LogProb;
            var ratio = Math.Exp(logRatio);

            // Clipped surrogate objective
            var clippedRatio = Math.Max(Math.Min(ratio, 1.0 + clipRange), 1.0 - clipRange);
            var surrogateObjective = Math.Min(ratio * exp.Advantage, clippedRatio * exp.Advantage);

            // Policy loss (negative because we want to maximize)
            var policyLoss = -surrogateObjective;

            // KL penalty
            policyLoss += klCoeff * exp.KL;

            totalPolicyLoss += policyLoss;

            // Compute and apply gradients scaled by capped advantage
            var cappedAdvantage = Math.Max(-1.0, Math.Min(1.0, exp.Advantage));
            var scaledLearningRate = Options.LearningRate * cappedAdvantage;
            var gradients = policyModel.ComputeGradients(input, exp.Output);
            policyModel.ApplyGradients(gradients, NumOps.FromDouble(scaledLearningRate));

            // Value loss: MSE between predicted value and observed reward
            var valueLoss = Math.Pow(exp.Value - exp.Reward, 2);
            totalValueLoss += valueLoss;

            // Update value model toward better value estimates
            if (_valueModel != null)
            {
                var valueError = exp.Reward - exp.Value;
                var cappedValueError = Math.Max(-1.0, Math.Min(1.0, valueError));
                var valueGradients = _valueModel.ComputeGradients(input, exp.Output);
                _valueModel.ApplyGradients(valueGradients, NumOps.FromDouble(Options.LearningRate * valueCoeff * cappedValueError));
            }

            // Entropy loss from policy distribution
            var entropyLoss = ComputeEntropyLoss(policyModel, input);
            totalEntropyLoss += entropyLoss;
        }

        int count = experience.Count;
        if (count == 0)
        {
            return 0.0;
        }

        var avgPolicyLoss = totalPolicyLoss / count;
        var avgValueLoss = totalValueLoss / count;
        var avgEntropyLoss = totalEntropyLoss / count;

        CurrentMetrics.PolicyLoss = avgPolicyLoss;
        CurrentMetrics.ValueLoss = avgValueLoss;
        CurrentMetrics.PolicyEntropy = -avgEntropyLoss;

        // Combined loss
        var totalLoss = avgPolicyLoss + valueCoeff * avgValueLoss + entropyCoeff * avgEntropyLoss;

        return await Task.FromResult(totalLoss);
    }

    /// <summary>
    /// Computes entropy loss to encourage exploration.
    /// </summary>
    /// <param name="model">The policy model.</param>
    /// <param name="input">The input state.</param>
    /// <returns>The negative entropy (as a loss to minimize).</returns>
    private double ComputeEntropyLoss(IFullModel<T, TInput, TOutput> model, TInput input)
    {
        var output = model.Predict(input);
        var outputVector = ConversionsHelper.ConvertToVector<T, TOutput>(output);

        if (outputVector.Length == 0)
        {
            return 0.0;
        }

        // Convert to probabilities using softmax
        var maxVal = outputVector.Max();
        var expValues = new double[outputVector.Length];
        double sumExp = 0.0;

        for (int i = 0; i < outputVector.Length; i++)
        {
            var val = NumOps.ToDouble(outputVector[i]) - NumOps.ToDouble(maxVal);
            expValues[i] = Math.Exp(val);
            sumExp += expValues[i];
        }

        // Compute entropy: -sum(p * log(p))
        double entropy = 0.0;
        for (int i = 0; i < expValues.Length; i++)
        {
            var prob = expValues[i] / sumExp;
            if (prob > 1e-10)
            {
                entropy -= prob * Math.Log(prob);
            }
        }

        // Return negative entropy as loss (we want to maximize entropy)
        return -entropy;
    }

    /// <summary>
    /// Represents a single experience sample for PPO training.
    /// </summary>
    private struct PPOExperience
    {
        public int InputIndex;
        public TOutput Output;
        public double Reward;
        public double LogProb;
        public double RefLogProb;
        public double Value;
        public double KL;
        public double Advantage;
    }
}
