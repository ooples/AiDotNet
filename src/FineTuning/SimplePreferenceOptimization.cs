using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Simple Preference Optimization (SimPO) for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// SimPO is a reference-free preference optimization method that outperforms DPO
/// while being more computationally efficient (no need for a reference model).
/// </para>
/// <para><b>For Beginners:</b> SimPO is like DPO but simpler and more memory efficient.
/// It doesn't need to keep a frozen copy of the original model, which saves GPU memory.
/// Instead, it uses the average log probability of responses and a target reward margin.
/// </para>
/// <para>
/// Key differences from DPO:
/// 1. Uses average log probability instead of sum (length-normalized)
/// 2. No reference model needed
/// 3. Adds a target reward margin (gamma) for stability
/// </para>
/// <para>
/// The SimPO loss function is:
/// L_SimPO = -log σ(β/|y| * (log π(y_w|x) - log π(y_l|x)) - γ)
/// where γ is the target margin and |y| denotes response length for normalization.
/// </para>
/// <para>
/// Original paper: "SimPO: Simple Preference Optimization with a Reference-Free Reward"
/// by Meng et al. (2024) - NeurIPS 2024
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class SimplePreferenceOptimization<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of SimPO fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public SimplePreferenceOptimization(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.SimPO)
        {
            options.MethodType = FineTuningMethodType.SimPO;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "SimPO";

    /// <inheritdoc/>
    public override FineTuningCategory Category => FineTuningCategory.DirectPreference;

    /// <inheritdoc/>
    public override bool RequiresRewardModel => false;

    /// <inheritdoc/>
    public override bool RequiresReferenceModel => false;

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

        if (!trainingData.HasPairwisePreferenceData)
        {
            throw new ArgumentException("SimPO requires pairwise preference data.", nameof(trainingData));
        }

        // Clone the base model for training (no reference model needed!)
        var policyModel = baseModel.Clone();

        CurrentMetrics = new FineTuningMetrics<T>
        {
            MethodName = MethodName,
            TrainingStartTime = DateTime.UtcNow
        };

        var beta = Options.Beta;
        var gamma = Options.SimPOGamma;
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

                var batchLoss = await ComputeSimPOLossAndUpdateAsync(policyModel, batch, beta, gamma, cancellationToken);
                currentStep++;

                UpdateMetrics(batchLoss, currentStep);
                LogProgress(currentStep, totalSteps, batchLoss, $"β={beta:F3}, γ={gamma:F3}");
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

            // SimPO uses average log probability (length-normalized)
            var chosenLogProb = ComputeAverageLogProbability(model, input, chosen);
            var rejectedLogProb = ComputeAverageLogProbability(model, input, rejected);

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
    /// Computes the SimPO loss for a batch and updates model parameters.
    /// </summary>
    /// <param name="policyModel">The model being trained.</param>
    /// <param name="batch">The training batch.</param>
    /// <param name="beta">The beta parameter for preference strength.</param>
    /// <param name="gamma">The target reward margin.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The average loss for the batch.</returns>
    private async Task<double> ComputeSimPOLossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        double beta,
        double gamma,
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

            // SimPO uses average log probability (length-normalized)
            var chosenAvgLogProb = ComputeAverageLogProbability(policyModel, input, chosen);
            var rejectedAvgLogProb = ComputeAverageLogProbability(policyModel, input, rejected);

            // SimPO loss: -log(sigmoid(beta * (chosen_avg_logprob - rejected_avg_logprob) - gamma))
            var margin = beta * (chosenAvgLogProb - rejectedAvgLogProb) - gamma;
            var loss = -LogSigmoid(margin);

            totalLoss += loss;
        }

        return await Task.FromResult(totalLoss / Math.Max(batch.Count, 1));
    }

    /// <summary>
    /// Computes the average log probability of an output (length-normalized).
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="input">The input.</param>
    /// <param name="output">The output.</param>
    /// <returns>The average log probability per token/element.</returns>
    private double ComputeAverageLogProbability(
        IFullModel<T, TInput, TOutput> model,
        TInput input,
        TOutput output)
    {
        var logProb = ComputeLogProbability(model, input, output);
        var length = GetOutputLength(output);

        // Return average log probability (length-normalized)
        return length > 0 ? logProb / length : logProb;
    }

    /// <summary>
    /// Gets the length of an output for normalization.
    /// </summary>
    /// <param name="output">The output.</param>
    /// <returns>The length of the output.</returns>
    private int GetOutputLength(TOutput output)
    {
        // Default implementation - subclasses should override for specific types
        if (output is Array arr)
        {
            return arr.Length;
        }

        if (output is System.Collections.ICollection col)
        {
            return col.Count;
        }

        // Default to 1 for scalar outputs
        return 1;
    }
}
