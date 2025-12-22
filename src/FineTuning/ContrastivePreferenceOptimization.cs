using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Contrastive Preference Optimization (CPO) for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// CPO is a variant of DPO that focuses on contrastive learning between chosen
/// and rejected responses without requiring a reference model, similar to ORPO
/// but with a different loss formulation.
/// </para>
/// <para><b>For Beginners:</b> CPO trains the model by directly comparing preferred
/// and rejected responses. It's simpler than DPO because it doesn't need to keep
/// a frozen copy of the original model.</para>
/// <para>
/// The CPO loss directly maximizes the gap between chosen and rejected outputs:
/// L_CPO = -log σ(β * (log π(y_w|x) - log π(y_l|x)))
/// </para>
/// <para>
/// Key advantages:
/// - No reference model needed (saves memory)
/// - Simpler implementation
/// - Good for scenarios where you want the model to deviate from its initial behavior
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class ContrastivePreferenceOptimization<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of CPO fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public ContrastivePreferenceOptimization(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.CPO)
        {
            options.MethodType = FineTuningMethodType.CPO;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "CPO";

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
                "CPO requires pairwise preference data. Ensure ChosenOutputs and RejectedOutputs are populated.",
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

                var batchLoss = await ComputeCPOLossAndUpdateAsync(policyModel, batch, beta, cancellationToken);
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
            }
        }

        metrics.WinRate = evaluationData.Count > 0 ? (double)wins / evaluationData.Count : 0.0;
        metrics.PreferenceAccuracy = metrics.WinRate;
        metrics.ChosenLogProb = evaluationData.Count > 0 ? totalChosenLogProb / evaluationData.Count : 0.0;
        metrics.RejectedLogProb = evaluationData.Count > 0 ? totalRejectedLogProb / evaluationData.Count : 0.0;
        metrics.LogProbMargin = metrics.ChosenLogProb - metrics.RejectedLogProb;

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Computes the CPO loss for a batch and updates model parameters.
    /// </summary>
    private async Task<double> ComputeCPOLossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        double beta,
        CancellationToken cancellationToken)
    {
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

            // Compute log probabilities directly (no reference model)
            var chosenLogProb = ComputeLogProbability(policyModel, input, chosen);
            var rejectedLogProb = ComputeLogProbability(policyModel, input, rejected);

            // CPO loss: -log(sigmoid(beta * (log_pi(y_w|x) - log_pi(y_l|x))))
            var margin = beta * (chosenLogProb - rejectedLogProb);
            var loss = -LogSigmoid(margin);

            // Apply label smoothing if configured
            if (Options.LabelSmoothing > 0)
            {
                var lossRejectedPreferred = -LogSigmoid(-margin);
                loss = (1 - Options.LabelSmoothing) * loss + Options.LabelSmoothing * lossRejectedPreferred;
            }

            totalLoss += loss;
        }

        return await Task.FromResult(batch.Count > 0 ? totalLoss / batch.Count : 0.0);
    }
}
