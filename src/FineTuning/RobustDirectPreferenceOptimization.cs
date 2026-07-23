using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Robust Direct Preference Optimization (RDPO) for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// RDPO extends DPO to handle noisy or inconsistent preference labels by incorporating
/// label confidence and noise-aware training.
/// </para>
/// <para><b>For Beginners:</b> Real-world preference data often contains mistakes -
/// annotators sometimes disagree or make errors. RDPO handles this by being more
/// cautious about uncertain preferences and focusing on high-confidence examples.</para>
/// <para>
/// Key features:
/// - Confidence-weighted preference learning
/// - Robust to label noise and annotator disagreement
/// - Uses sample weights to down-weight uncertain examples
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class RobustDirectPreferenceOptimization<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    private IFullModel<T, TInput, TOutput>? _referenceModel;

    /// <summary>
    /// Initializes a new instance of RDPO fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public RobustDirectPreferenceOptimization(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.RDPO)
        {
            options.MethodType = FineTuningMethodType.RDPO;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "RDPO";

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
                "RDPO requires pairwise preference data. Ensure ChosenOutputs and RejectedOutputs are populated.",
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

        _referenceModel = baseModel.Clone();
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

                var batchLoss = await ComputeRDPOLossAndUpdateAsync(policyModel, batch, beta, cancellationToken);
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
    /// Computes the RDPO loss for a batch with robustness to label noise.
    /// </summary>
    private async Task<double> ComputeRDPOLossAndUpdateAsync(
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
        double totalWeight = 0.0;

        for (int i = 0; i < batch.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = batch.Inputs[i];
            var chosen = batch.ChosenOutputs[i];
            var rejected = batch.RejectedOutputs[i];

            // Get sample weight (confidence) if available
            double weight = 1.0;
            if (batch.SampleWeights.Length > i)
            {
                weight = batch.SampleWeights[i];
            }

            // Compute log probabilities
            var piChosenLogProb = ComputeLogProbability(policyModel, input, chosen);
            var piRejectedLogProb = ComputeLogProbability(policyModel, input, rejected);
            var refChosenLogProb = ComputeLogProbability(_referenceModel, input, chosen);
            var refRejectedLogProb = ComputeLogProbability(_referenceModel, input, rejected);

            // Compute log ratios
            var chosenLogRatio = piChosenLogProb - refChosenLogProb;
            var rejectedLogRatio = piRejectedLogProb - refRejectedLogProb;

            // Standard DPO margin
            var margin = beta * (chosenLogRatio - rejectedLogRatio);

            // Robust loss with confidence weighting
            // For uncertain samples (low weight), use a softer loss
            var baseLoss = -LogSigmoid(margin);

            // Add robustness term: reduce impact of samples where model strongly disagrees
            // This helps when the preference label might be wrong
            var modelConfidence = Sigmoid(margin);
            var robustnessWeight = ComputeRobustnessWeight(modelConfidence, weight);

            var loss = robustnessWeight * baseLoss;

            totalLoss += loss;
            totalWeight += robustnessWeight;
        }

        return await Task.FromResult(totalWeight > 0 ? totalLoss / totalWeight : 0.0);
    }

    /// <summary>
    /// Computes the robustness weight for a sample based on model confidence and label confidence.
    /// </summary>
    private static double ComputeRobustnessWeight(double modelConfidence, double labelConfidence)
    {
        // If model strongly disagrees with label (modelConfidence < 0.5) and
        // label confidence is low, reduce the sample weight
        // This helps handle noisy labels
        if (modelConfidence < 0.3 && labelConfidence < 0.7)
        {
            // Down-weight this sample
            return labelConfidence * 0.5;
        }

        return labelConfidence;
    }
}
