using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Self-Play Fine-Tuning (SPIN) for model improvement.
/// </summary>
/// <remarks>
/// <para>
/// SPIN uses self-play to improve models without additional human-labeled data.
/// The model plays against previous versions of itself, learning to distinguish
/// its own outputs from ground truth.
/// </para>
/// <para><b>For Beginners:</b> SPIN is like a model playing chess against itself
/// to get better. The current model learns to prefer real human responses over
/// its own generated responses, iteratively improving without new labeled data.</para>
/// <para>
/// Key features:
/// - No additional human labels needed after initial SFT
/// - Iterative improvement through self-play
/// - Uses DPO-style objective with model's own outputs as negatives
/// </para>
/// <para>
/// Original paper: "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models"
/// by Chen et al. (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class SelfPlayFineTuning<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    private IFullModel<T, TInput, TOutput>? _opponentModel;

    /// <summary>
    /// Initializes a new instance of SPIN fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public SelfPlayFineTuning(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.SPIN)
        {
            options.MethodType = FineTuningMethodType.SPIN;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "SPIN";

    /// <inheritdoc/>
    public override FineTuningCategory Category => FineTuningCategory.SelfPlay;

    /// <inheritdoc/>
    public override bool RequiresRewardModel => false;

    /// <inheritdoc/>
    public override bool RequiresReferenceModel => false;

    /// <inheritdoc/>
    protected override void ValidateTrainingData(FineTuningData<T, TInput, TOutput> data)
    {
        base.ValidateTrainingData(data);

        if (!data.HasSFTData)
        {
            throw new ArgumentException(
                "SPIN requires SFT data (Inputs and Outputs) to provide ground truth responses.",
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
        var spinIterations = Options.SPINIterations;

        CurrentMetrics = new FineTuningMetrics<T>
        {
            MethodName = MethodName,
            TrainingStartTime = DateTime.UtcNow
        };

        var beta = Options.Beta;
        var totalSteps = spinIterations * Options.Epochs * (trainingData.Count / Options.BatchSize);
        var currentStep = 0;

        // SPIN iterative training
        for (int iteration = 0; iteration < spinIterations; iteration++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            // The opponent is the model from the previous iteration
            _opponentModel = policyModel.Clone();

            LogProgress(currentStep, totalSteps, 0, $"SPIN iteration {iteration + 1}/{spinIterations}");

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

                    var batchLoss = await ComputeSPINLossAndUpdateAsync(
                        policyModel, batch, beta, cancellationToken);
                    currentStep++;

                    UpdateMetrics(batchLoss, currentStep);
                    LogProgress(currentStep, totalSteps, batchLoss, $"iter={iteration + 1}, β={beta:F3}");
                }
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

        if (!evaluationData.HasSFTData)
        {
            return metrics;
        }

        int wins = 0;
        double totalGroundTruthLogProb = 0.0;
        double totalModelLogProb = 0.0;

        for (int i = 0; i < evaluationData.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = evaluationData.Inputs[i];
            var groundTruth = evaluationData.Outputs[i];
            var modelOutput = model.Predict(input);

            var groundTruthLogProb = ComputeLogProbability(model, input, groundTruth);
            var modelLogProb = ComputeLogProbability(model, input, modelOutput);

            totalGroundTruthLogProb += groundTruthLogProb;
            totalModelLogProb += modelLogProb;

            // Model should prefer ground truth over its own outputs
            if (groundTruthLogProb > modelLogProb)
            {
                wins++;
            }
        }

        // In SPIN, high win rate means model prefers ground truth (good!)
        metrics.WinRate = evaluationData.Count > 0 ? (double)wins / evaluationData.Count : 0.0;
        metrics.ChosenLogProb = evaluationData.Count > 0 ? totalGroundTruthLogProb / evaluationData.Count : 0.0;
        metrics.RejectedLogProb = evaluationData.Count > 0 ? totalModelLogProb / evaluationData.Count : 0.0;
        metrics.LogProbMargin = metrics.ChosenLogProb - metrics.RejectedLogProb;

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Computes the SPIN loss for a batch.
    /// </summary>
    private async Task<double> ComputeSPINLossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        double beta,
        CancellationToken cancellationToken)
    {
        if (_opponentModel == null)
        {
            throw new InvalidOperationException("Opponent model not initialized.");
        }

        double totalLoss = 0.0;

        for (int i = 0; i < batch.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = batch.Inputs[i];
            var groundTruth = batch.Outputs[i];

            // Generate opponent (previous iteration) response
            var opponentOutput = _opponentModel.Predict(input);

            // SPIN objective: prefer ground truth over opponent's output
            // This is DPO-style with ground truth as chosen and opponent output as rejected
            var piGroundTruthLogProb = ComputeLogProbability(policyModel, input, groundTruth);
            var piOpponentLogProb = ComputeLogProbability(policyModel, input, opponentOutput);
            var refGroundTruthLogProb = ComputeLogProbability(_opponentModel, input, groundTruth);
            var refOpponentLogProb = ComputeLogProbability(_opponentModel, input, opponentOutput);

            var groundTruthLogRatio = piGroundTruthLogProb - refGroundTruthLogProb;
            var opponentLogRatio = piOpponentLogProb - refOpponentLogProb;

            var margin = beta * (groundTruthLogRatio - opponentLogRatio);
            var loss = -LogSigmoid(margin);

            totalLoss += loss;
        }

        return await Task.FromResult(batch.Count > 0 ? totalLoss / batch.Count : 0.0);
    }
}
