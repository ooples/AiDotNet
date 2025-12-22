using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Odds Ratio Preference Optimization (ORPO) for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// ORPO combines SFT and preference optimization in a single training objective.
/// It doesn't require a reference model, making it simpler and more memory efficient.
/// </para>
/// <para><b>For Beginners:</b> ORPO is a clever method that learns both to produce
/// good outputs (like SFT) and to prefer good outputs over bad ones (like DPO)
/// at the same time. It's simpler because it doesn't need a frozen reference model.</para>
/// <para>
/// The ORPO loss combines:
/// 1. SFT loss on chosen outputs: -log P(y_w|x)
/// 2. Odds ratio loss: -λ * log(odds(y_w) / odds(y_l))
/// where odds(y) = P(y|x) / (1 - P(y|x))
/// </para>
/// <para>
/// Original paper: "ORPO: Monolithic Preference Optimization without Reference Model"
/// by Hong et al. (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class OddsRatioPreferenceOptimization<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of ORPO fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public OddsRatioPreferenceOptimization(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.ORPO)
        {
            options.MethodType = FineTuningMethodType.ORPO;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "ORPO";

    /// <inheritdoc/>
    public override FineTuningCategory Category => FineTuningCategory.DirectPreference;

    /// <inheritdoc/>
    public override bool RequiresRewardModel => false;

    /// <inheritdoc/>
    public override bool RequiresReferenceModel => false;

    /// <inheritdoc/>
    protected override void ValidateTrainingData(FineTuningData<T, TInput, TOutput> data)
    {
        base.ValidateTrainingData(data);

        if (!data.HasPairwisePreferenceData)
        {
            throw new ArgumentException(
                "ORPO requires pairwise preference data. Ensure ChosenOutputs and RejectedOutputs are populated.",
                nameof(data));
        }
    }

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

        // Clone the base model as the policy (will be trained)
        // ORPO doesn't need a reference model
        var policyModel = baseModel.Clone();

        CurrentMetrics = new FineTuningMetrics<T>
        {
            MethodName = MethodName,
            TrainingStartTime = DateTime.UtcNow
        };

        var lambda = Options.ORPOLambda;
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

                var batchLoss = await ComputeORPOLossAndUpdateAsync(policyModel, batch, lambda, cancellationToken);
                currentStep++;

                UpdateMetrics(batchLoss, currentStep);
                LogProgress(currentStep, totalSteps, batchLoss, $"λ={lambda:F3}");
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

        if (!evaluationData.HasPairwisePreferenceData)
        {
            return metrics;
        }

        int wins = 0;
        int correct = 0;
        double totalChosenLogProb = 0.0;
        double totalRejectedLogProb = 0.0;
        double totalOddsRatio = 0.0;

        for (int i = 0; i < evaluationData.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = evaluationData.Inputs[i];
            var chosen = evaluationData.ChosenOutputs[i];
            var rejected = evaluationData.RejectedOutputs[i];

            var chosenLogProb = ComputeLogProbability(model, input, chosen);
            var rejectedLogProb = ComputeLogProbability(model, input, rejected);

            totalChosenLogProb += chosenLogProb;
            totalRejectedLogProb += rejectedLogProb;

            // Compute odds ratio
            var chosenOdds = ComputeOdds(chosenLogProb);
            var rejectedOdds = ComputeOdds(rejectedLogProb);
            if (rejectedOdds > 1e-10)
            {
                totalOddsRatio += chosenOdds / rejectedOdds;
            }

            if (chosenLogProb > rejectedLogProb)
            {
                wins++;
                correct++;
            }
        }

        metrics.WinRate = evaluationData.Count > 0 ? (double)wins / evaluationData.Count : 0.0;
        metrics.PreferenceAccuracy = evaluationData.Count > 0 ? (double)correct / evaluationData.Count : 0.0;
        metrics.ChosenLogProb = evaluationData.Count > 0 ? totalChosenLogProb / evaluationData.Count : 0.0;
        metrics.RejectedLogProb = evaluationData.Count > 0 ? totalRejectedLogProb / evaluationData.Count : 0.0;
        metrics.LogProbMargin = metrics.ChosenLogProb - metrics.RejectedLogProb;
        metrics.CustomMetrics["average_odds_ratio"] = evaluationData.Count > 0 ? totalOddsRatio / evaluationData.Count : 0.0;

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Computes the ORPO loss for a batch and updates model parameters.
    /// </summary>
    private async Task<double> ComputeORPOLossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        double lambda,
        CancellationToken cancellationToken)
    {
        double totalLoss = 0.0;
        double totalSFTLoss = 0.0;
        double totalORLoss = 0.0;

        for (int i = 0; i < batch.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = batch.Inputs[i];
            var chosen = batch.ChosenOutputs[i];
            var rejected = batch.RejectedOutputs[i];

            // Compute log probabilities
            var chosenLogProb = ComputeLogProbability(policyModel, input, chosen);
            var rejectedLogProb = ComputeLogProbability(policyModel, input, rejected);

            // SFT loss on chosen output: -log P(y_w|x)
            var sftLoss = -chosenLogProb;
            totalSFTLoss += sftLoss;

            // Odds ratio loss: -log(odds(y_w) / odds(y_l))
            // where odds(y) = P(y|x) / (1 - P(y|x)) = exp(log_p) / (1 - exp(log_p))
            var chosenLogOdds = ComputeLogOdds(chosenLogProb);
            var rejectedLogOdds = ComputeLogOdds(rejectedLogProb);
            var orLoss = -LogSigmoid(chosenLogOdds - rejectedLogOdds);
            totalORLoss += orLoss;

            // Combined loss: SFT + lambda * OR
            var loss = sftLoss + lambda * orLoss;
            totalLoss += loss;
        }

        // Store additional metrics
        if (batch.Count > 0)
        {
            CurrentMetrics.CustomMetrics["sft_loss"] = totalSFTLoss / batch.Count;
            CurrentMetrics.CustomMetrics["or_loss"] = totalORLoss / batch.Count;
        }

        return await Task.FromResult(batch.Count > 0 ? totalLoss / batch.Count : 0.0);
    }

    /// <summary>
    /// Computes the odds from log probability: P/(1-P).
    /// </summary>
    private static double ComputeOdds(double logProb)
    {
        var prob = Math.Exp(logProb);
        prob = Math.Min(prob, 1.0 - 1e-10); // Prevent division by zero
        return prob / (1.0 - prob);
    }

    /// <summary>
    /// Computes the log odds from log probability: log(P/(1-P)).
    /// </summary>
    private static double ComputeLogOdds(double logProb)
    {
        // log(P/(1-P)) = log(P) - log(1-P)
        // For numerical stability when P is close to 1:
        // log(1-P) = log(1 - exp(log_p))
        var prob = Math.Exp(logProb);
        if (prob > 0.999)
        {
            // Use approximation for high probabilities
            return logProb - Math.Log(1e-10);
        }
        return logProb - Math.Log(1.0 - prob);
    }
}
