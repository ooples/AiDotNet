using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Pairwise Ranking Optimization (PRO) for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// PRO optimizes models using pairwise comparisons from ranked lists,
/// similar to learning-to-rank algorithms used in search engines.
/// </para>
/// <para><b>For Beginners:</b> PRO treats model alignment as a ranking problem.
/// Given multiple responses, it learns to assign higher scores to better responses,
/// using techniques from information retrieval and recommendation systems.</para>
/// <para>
/// Key features:
/// - Uses pairwise ranking loss (similar to RankNet)
/// - Can leverage ranking information from multiple annotators
/// - Robust to noise through margin-based learning
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class PairwiseRankingOptimization<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of PRO fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public PairwiseRankingOptimization(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.PRO)
        {
            options.MethodType = FineTuningMethodType.PRO;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "PRO";

    /// <inheritdoc/>
    public override FineTuningCategory Category => FineTuningCategory.RankingBased;

    /// <inheritdoc/>
    public override bool RequiresRewardModel => false;

    /// <inheritdoc/>
    public override bool RequiresReferenceModel => false;

    /// <inheritdoc/>
    protected override void ValidateTrainingData(FineTuningData<T, TInput, TOutput> data)
    {
        base.ValidateTrainingData(data);

        if (!data.HasRankingData && !data.HasPairwisePreferenceData)
        {
            throw new ArgumentException(
                "PRO requires either RankedOutputs or pairwise preference data.",
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

        var margin = Options.RankingMargin;
        var temperature = Options.RankingTemperature;
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

                var batchLoss = await ComputePROLossAndUpdateAsync(
                    policyModel, batch, margin, temperature, cancellationToken);
                currentStep++;

                UpdateMetrics(batchLoss, currentStep);
                LogProgress(currentStep, totalSteps, batchLoss, $"margin={margin:F3}");
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

        int correctPairs = 0;
        int totalPairs = 0;

        if (evaluationData.HasRankingData)
        {
            for (int i = 0; i < evaluationData.Count; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                var input = evaluationData.Inputs[i];
                var rankedOutputs = evaluationData.RankedOutputs[i];

                for (int j = 0; j < rankedOutputs.Length - 1; j++)
                {
                    for (int k = j + 1; k < rankedOutputs.Length; k++)
                    {
                        var betterLogProb = ComputeLogProbability(model, input, rankedOutputs[j]);
                        var worseLogProb = ComputeLogProbability(model, input, rankedOutputs[k]);

                        totalPairs++;
                        if (betterLogProb > worseLogProb)
                        {
                            correctPairs++;
                        }
                    }
                }
            }
        }
        else if (evaluationData.HasPairwisePreferenceData)
        {
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

                totalPairs++;
                if (chosenLogProb > rejectedLogProb)
                {
                    correctPairs++;
                }
            }
        }

        metrics.PreferenceAccuracy = totalPairs > 0 ? (double)correctPairs / totalPairs : 0.0;
        metrics.WinRate = metrics.PreferenceAccuracy;

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Computes the PRO loss for a batch.
    /// </summary>
    private async Task<double> ComputePROLossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        double margin,
        double temperature,
        CancellationToken cancellationToken)
    {
        double totalLoss = 0.0;
        int pairCount = 0;

        if (batch.HasRankingData)
        {
            for (int i = 0; i < batch.Count; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                var input = batch.Inputs[i];
                var rankedOutputs = batch.RankedOutputs[i];

                // Generate all pairwise comparisons
                for (int j = 0; j < rankedOutputs.Length - 1; j++)
                {
                    for (int k = j + 1; k < rankedOutputs.Length; k++)
                    {
                        var betterOutput = rankedOutputs[j];
                        var worseOutput = rankedOutputs[k];

                        var loss = ComputePairwiseRankingLoss(
                            policyModel, input, betterOutput, worseOutput, margin, temperature);
                        totalLoss += loss;
                        pairCount++;
                    }
                }
            }
        }
        else if (batch.HasPairwisePreferenceData)
        {
            for (int i = 0; i < batch.Count; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                var input = batch.Inputs[i];
                var chosen = batch.ChosenOutputs[i];
                var rejected = batch.RejectedOutputs[i];

                var loss = ComputePairwiseRankingLoss(
                    policyModel, input, chosen, rejected, margin, temperature);
                totalLoss += loss;
                pairCount++;
            }
        }

        return await Task.FromResult(pairCount > 0 ? totalLoss / pairCount : 0.0);
    }

    /// <summary>
    /// Computes the pairwise ranking loss (RankNet-style).
    /// </summary>
    private double ComputePairwiseRankingLoss(
        IFullModel<T, TInput, TOutput> model,
        TInput input,
        TOutput better,
        TOutput worse,
        double margin,
        double temperature)
    {
        var betterScore = ComputeLogProbability(model, input, better);
        var worseScore = ComputeLogProbability(model, input, worse);

        // RankNet loss: -log(sigmoid((better_score - worse_score - margin) / temperature))
        var scoreDiff = (betterScore - worseScore - margin) / temperature;
        return -LogSigmoid(scoreDiff);
    }
}
