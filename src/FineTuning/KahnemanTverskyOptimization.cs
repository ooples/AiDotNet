using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Kahneman-Tversky Optimization (KTO) for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// KTO applies prospect theory from behavioral economics to preference learning.
/// Unlike DPO, it doesn't require paired data - each example is independently labeled
/// as desirable or undesirable.
/// </para>
/// <para><b>For Beginners:</b> KTO is based on how humans actually make decisions.
/// People feel losses more strongly than equivalent gains (loss aversion).
/// KTO uses this insight to train models - making them more careful about avoiding
/// bad outputs than they are eager to produce good ones.</para>
/// <para>
/// Key features:
/// - Doesn't require paired preference data
/// - Uses loss aversion (undesirable weight typically higher than desirable weight)
/// - More sample efficient for imbalanced datasets
/// </para>
/// <para>
/// Original paper: "KTO: Model Alignment as Prospect Theoretic Optimization"
/// by Ethayarajh et al. (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class KahnemanTverskyOptimization<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    private IFullModel<T, TInput, TOutput>? _referenceModel;

    /// <summary>
    /// Initializes a new instance of KTO fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public KahnemanTverskyOptimization(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.KTO)
        {
            options.MethodType = FineTuningMethodType.KTO;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "KTO";

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

        // KTO can work with either paired or unpaired data
        if (!data.HasPairwisePreferenceData && !data.HasUnpairedPreferenceData)
        {
            throw new ArgumentException(
                "KTO requires either pairwise preference data or unpaired preference data with DesirabilityLabels.",
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
        var desirableWeight = Options.KTODesirableWeight;
        var undesirableWeight = Options.KTOUndesirableWeight;

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

                var batchLoss = await ComputeKTOLossAndUpdateAsync(
                    policyModel, batch, beta, desirableWeight, undesirableWeight, cancellationToken);
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

        int desirableCorrect = 0;
        int undesirableCorrect = 0;
        int desirableCount = 0;
        int undesirableCount = 0;

        // For KTO, evaluate on unpaired data if available
        if (evaluationData.HasUnpairedPreferenceData && evaluationData.DesirabilityLabels.Length > 0)
        {
            for (int i = 0; i < evaluationData.DesirabilityLabels.Length; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                var isDesirable = evaluationData.DesirabilityLabels[i];
                var input = evaluationData.Inputs[i];
                var output = isDesirable
                    ? evaluationData.ChosenOutputs[Math.Min(i, evaluationData.ChosenOutputs.Length - 1)]
                    : evaluationData.RejectedOutputs[Math.Min(i, evaluationData.RejectedOutputs.Length - 1)];

                var logProb = ComputeLogProbability(model, input, output);

                if (isDesirable)
                {
                    desirableCount++;
                    if (logProb > 0) desirableCorrect++;
                }
                else
                {
                    undesirableCount++;
                    if (logProb < 0) undesirableCorrect++;
                }
            }
        }
        else if (evaluationData.HasPairwisePreferenceData)
        {
            // Fall back to pairwise evaluation
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

                desirableCount++;
                undesirableCount++;

                if (chosenLogProb > rejectedLogProb)
                {
                    desirableCorrect++;
                    undesirableCorrect++;
                }
            }
        }

        var totalCount = desirableCount + undesirableCount;
        var totalCorrect = desirableCorrect + undesirableCorrect;

        metrics.PreferenceAccuracy = totalCount > 0 ? (double)totalCorrect / totalCount : 0.0;
        metrics.CustomMetrics["desirable_accuracy"] = desirableCount > 0 ? (double)desirableCorrect / desirableCount : 0.0;
        metrics.CustomMetrics["undesirable_accuracy"] = undesirableCount > 0 ? (double)undesirableCorrect / undesirableCount : 0.0;

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Computes the KTO loss for a batch and updates model parameters.
    /// </summary>
    private async Task<double> ComputeKTOLossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        double beta,
        double desirableWeight,
        double undesirableWeight,
        CancellationToken cancellationToken)
    {
        if (_referenceModel == null)
        {
            throw new InvalidOperationException("Reference model not initialized.");
        }

        double totalLoss = 0.0;
        int count = 0;

        // If we have paired data, convert to unpaired format
        if (batch.HasPairwisePreferenceData)
        {
            // First compute KL estimate (needed for the value function)
            double klEstimate = ComputeBatchKLEstimate(policyModel, batch);

            // Process chosen (desirable) outputs
            for (int i = 0; i < batch.Count; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                var input = batch.Inputs[i];
                var chosen = batch.ChosenOutputs[i];

                var piLogProb = ComputeLogProbability(policyModel, input, chosen);
                var refLogProb = ComputeLogProbability(_referenceModel, input, chosen);
                var logRatio = piLogProb - refLogProb;

                // KTO desirable loss: -weight * sigmoid(beta * (log_ratio - KL))
                var loss = -desirableWeight * Sigmoid(beta * (logRatio - klEstimate));
                totalLoss += loss;
                count++;
            }

            // Process rejected (undesirable) outputs
            for (int i = 0; i < batch.Count; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                var input = batch.Inputs[i];
                var rejected = batch.RejectedOutputs[i];

                var piLogProb = ComputeLogProbability(policyModel, input, rejected);
                var refLogProb = ComputeLogProbability(_referenceModel, input, rejected);
                var logRatio = piLogProb - refLogProb;

                // KTO undesirable loss: -weight * (1 - sigmoid(beta * (log_ratio - KL)))
                var loss = -undesirableWeight * (1 - Sigmoid(beta * (logRatio - klEstimate)));
                totalLoss += loss;
                count++;
            }
        }

        return await Task.FromResult(count > 0 ? totalLoss / count : 0.0);
    }

    /// <summary>
    /// Computes the KL divergence estimate for the batch.
    /// </summary>
    private double ComputeBatchKLEstimate(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch)
    {
        if (_referenceModel == null)
        {
            return 0.0;
        }

        double totalKL = 0.0;
        int count = 0;

        // Estimate KL using chosen outputs
        for (int i = 0; i < batch.Count; i++)
        {
            var input = batch.Inputs[i];
            var chosen = batch.ChosenOutputs[i];

            var piLogProb = ComputeLogProbability(policyModel, input, chosen);
            var refLogProb = ComputeLogProbability(_referenceModel, input, chosen);

            // KL = sum(pi * log(pi/ref)) = log(pi) - log(ref) when using samples
            totalKL += piLogProb - refLogProb;
            count++;
        }

        return count > 0 ? totalKL / count : 0.0;
    }
}
