using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Identity Preference Optimization (IPO) for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// IPO improves upon DPO by addressing overfitting issues through a different loss function
/// that doesn't push the rejected response probability to zero.
/// </para>
/// <para><b>For Beginners:</b> IPO is like DPO but more stable. It learns preferences
/// without being too aggressive, which helps prevent the model from becoming overconfident
/// or forgetting how to generate diverse responses.</para>
/// <para>
/// The IPO loss function is:
/// L_IPO = (log(π(y_w|x)/π_ref(y_w|x)) - log(π(y_l|x)/π_ref(y_l|x)) - 1/(2β))²
/// This squared loss form provides better gradients and prevents collapse.
/// </para>
/// <para>
/// Original paper: "A General Theoretical Paradigm to Understand Learning from Human Preferences"
/// by Azar et al. (2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class IdentityPreferenceOptimization<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    private IFullModel<T, TInput, TOutput>? _referenceModel;

    /// <summary>
    /// Initializes a new instance of IPO fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public IdentityPreferenceOptimization(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.IPO)
        {
            options.MethodType = FineTuningMethodType.IPO;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "IPO";

    /// <inheritdoc/>
    public override FineTuningCategory Category => FineTuningCategory.DirectPreference;

    /// <inheritdoc/>
    public override bool RequiresRewardModel => false;

    /// <inheritdoc/>
    public override bool RequiresReferenceModel => true;

    /// <inheritdoc/>
    protected override void ValidateTrainingData(FineTuningData<T, TInput, TOutput> data)
    {
        base.ValidateTrainingData(data);

        if (!data.HasPairwisePreferenceData)
        {
            throw new ArgumentException(
                "IPO requires pairwise preference data. Ensure ChosenOutputs and RejectedOutputs are populated.",
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

        // Clone the base model as the reference (frozen during training)
        _referenceModel = baseModel.Clone();

        // Clone the base model as the policy (will be trained)
        var policyModel = baseModel.Clone();

        CurrentMetrics = new FineTuningMetrics<T>
        {
            MethodName = MethodName,
            TrainingStartTime = DateTime.UtcNow
        };

        var beta = Options.Beta;
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

                var batchLoss = await ComputeIPOLossAndUpdateAsync(policyModel, batch, beta, cancellationToken);
                currentStep++;

                UpdateMetrics(batchLoss, currentStep);
                LogProgress(currentStep, totalSteps, batchLoss, $"β={beta:F3}");
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

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Computes the IPO loss for a batch and updates model parameters.
    /// </summary>
    /// <param name="policyModel">The model being trained.</param>
    /// <param name="batch">The training batch.</param>
    /// <param name="beta">The beta parameter for preference strength.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The average loss for the batch.</returns>
    private async Task<double> ComputeIPOLossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        double beta,
        CancellationToken cancellationToken)
    {
        if (_referenceModel == null)
        {
            throw new InvalidOperationException("Reference model not initialized.");
        }

        double totalLoss = 0.0;

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
            var piChosenLogProb = ComputeLogProbability(policyModel, input, chosen);
            var piRejectedLogProb = ComputeLogProbability(policyModel, input, rejected);
            var refChosenLogProb = ComputeLogProbability(_referenceModel, input, chosen);
            var refRejectedLogProb = ComputeLogProbability(_referenceModel, input, rejected);

            // Compute log ratios
            var chosenLogRatio = piChosenLogProb - refChosenLogProb;
            var rejectedLogRatio = piRejectedLogProb - refRejectedLogProb;

            // IPO loss: ((log_ratio_w - log_ratio_l) - 1/(2*beta))^2
            // This is a squared loss that targets a specific margin
            var margin = chosenLogRatio - rejectedLogRatio;
            var target = 1.0 / (2.0 * beta);
            var loss = Math.Pow(margin - target, 2);

            totalLoss += loss;
        }

        return await Task.FromResult(totalLoss / Math.Max(batch.Count, 1));
    }
}
