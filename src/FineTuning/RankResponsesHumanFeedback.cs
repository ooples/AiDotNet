using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Rank Responses to Align Human Feedback (RRHF) for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// RRHF aligns language models by ranking multiple responses and learning from
/// the ranking order rather than just pairwise preferences.
/// </para>
/// <para><b>For Beginners:</b> RRHF learns from a full ranking of responses,
/// not just pairs. If you have 5 responses ranked from best to worst, RRHF
/// uses all that ranking information to train the model more efficiently.</para>
/// <para>
/// Key features:
/// - Uses full ranking information, not just pairs
/// - More efficient use of human feedback
/// - Combines SFT and ranking objectives
/// </para>
/// <para>
/// Original paper: "RRHF: Rank Responses to Align Language Models with Human Feedback
/// without tears" by Yuan et al. (2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class RankResponsesHumanFeedback<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    /// <summary>
    /// Initializes a new instance of RRHF fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public RankResponsesHumanFeedback(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.RRHF)
        {
            options.MethodType = FineTuningMethodType.RRHF;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "RRHF";

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

        if (!data.HasRankingData)
        {
            throw new ArgumentException(
                "RRHF requires RankedOutputs data with responses ranked from best to worst.",
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

                var batchLoss = await ComputeRRHFLossAndUpdateAsync(
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

        if (!evaluationData.HasRankingData)
        {
            return metrics;
        }

        int correctPairs = 0;
        int totalPairs = 0;
        double totalKendallTau = 0.0;

        for (int i = 0; i < evaluationData.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = evaluationData.Inputs[i];
            var rankedOutputs = evaluationData.RankedOutputs[i];

            // Compute model's ranking
            var scores = new double[rankedOutputs.Length];
            for (int j = 0; j < rankedOutputs.Length; j++)
            {
                scores[j] = ComputeLogProbability(model, input, rankedOutputs[j]);
            }

            // Check pairwise accuracy
            for (int j = 0; j < rankedOutputs.Length - 1; j++)
            {
                for (int k = j + 1; k < rankedOutputs.Length; k++)
                {
                    totalPairs++;
                    if (scores[j] > scores[k])
                    {
                        correctPairs++;
                    }
                }
            }

            // Compute Kendall's tau correlation
            totalKendallTau += ComputeKendallTau(scores);
        }

        metrics.PreferenceAccuracy = totalPairs > 0 ? (double)correctPairs / totalPairs : 0.0;
        metrics.WinRate = metrics.PreferenceAccuracy;
        metrics.CustomMetrics["kendall_tau"] = evaluationData.Count > 0 ? totalKendallTau / evaluationData.Count : 0.0;

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Computes the RRHF loss for a batch.
    /// </summary>
    private async Task<double> ComputeRRHFLossAndUpdateAsync(
        IFullModel<T, TInput, TOutput> policyModel,
        FineTuningData<T, TInput, TOutput> batch,
        double margin,
        double temperature,
        CancellationToken cancellationToken)
    {
        double totalLoss = 0.0;
        int count = 0;

        for (int i = 0; i < batch.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            var input = batch.Inputs[i];
            var rankedOutputs = batch.RankedOutputs[i];

            if (rankedOutputs.Length < 2)
            {
                continue;
            }

            // RRHF loss combines SFT on best response with ranking loss

            // 1. SFT loss on best response
            var bestOutput = rankedOutputs[0];
            var sftLoss = -ComputeLogProbability(policyModel, input, bestOutput);

            // 2. Ranking loss - ensure proper ordering with margin
            double rankingLoss = 0.0;
            int pairCount = 0;

            for (int j = 0; j < rankedOutputs.Length - 1; j++)
            {
                for (int k = j + 1; k < rankedOutputs.Length; k++)
                {
                    var betterLogProb = ComputeLogProbability(policyModel, input, rankedOutputs[j]);
                    var worseLogProb = ComputeLogProbability(policyModel, input, rankedOutputs[k]);

                    // Margin-based ranking loss
                    // We want: betterLogProb - worseLogProb > margin * (k - j)
                    var requiredMargin = margin * (k - j);
                    var actualMargin = betterLogProb - worseLogProb;

                    if (actualMargin < requiredMargin)
                    {
                        rankingLoss += Math.Pow(requiredMargin - actualMargin, 2);
                    }

                    pairCount++;
                }
            }

            if (pairCount > 0)
            {
                rankingLoss /= pairCount;
            }

            // Combined loss with temperature weighting
            var loss = sftLoss + temperature * rankingLoss;
            totalLoss += loss;
            count++;
        }

        return await Task.FromResult(count > 0 ? totalLoss / count : 0.0);
    }

    /// <summary>
    /// Computes Kendall's tau correlation coefficient for a ranking.
    /// </summary>
    private static double ComputeKendallTau(double[] scores)
    {
        int n = scores.Length;
        if (n < 2)
        {
            return 1.0;
        }

        int concordant = 0;
        int discordant = 0;

        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                // Scores should be in decreasing order for correct ranking
                if (scores[i] > scores[j])
                {
                    concordant++;
                }
                else if (scores[i] < scores[j])
                {
                    discordant++;
                }
            }
        }

        int totalPairs = n * (n - 1) / 2;
        return totalPairs > 0 ? (double)(concordant - discordant) / totalPairs : 0.0;
    }
}
