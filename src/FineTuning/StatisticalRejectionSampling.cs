using AiDotNet.Interfaces;
using AiDotNet.Models.Options;

namespace AiDotNet.FineTuning;

/// <summary>
/// Implements Statistical Rejection Sampling Optimization (RSO) for fine-tuning.
/// </summary>
/// <remarks>
/// <para>
/// RSO uses rejection sampling to create high-quality training pairs from ranked outputs.
/// It samples pairs based on the statistical properties of the ranking to create
/// better preference data.
/// </para>
/// <para><b>For Beginners:</b> RSO takes a list of responses ranked from best to worst
/// and intelligently picks pairs to learn from. Instead of just using best vs worst,
/// it uses statistical sampling to create more informative training examples.</para>
/// <para>
/// Original paper: "Statistical Rejection Sampling Improves Preference Optimization"
/// by Liu et al. (2024)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class StatisticalRejectionSampling<T, TInput, TOutput> : FineTuningBase<T, TInput, TOutput>
{
    private IFullModel<T, TInput, TOutput>? _referenceModel;

    /// <summary>
    /// Initializes a new instance of RSO fine-tuning.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    public StatisticalRejectionSampling(FineTuningOptions<T> options) : base(options)
    {
        if (options.MethodType != FineTuningMethodType.RSO)
        {
            options.MethodType = FineTuningMethodType.RSO;
        }
    }

    /// <inheritdoc/>
    public override string MethodName => "RSO";

    /// <inheritdoc/>
    public override FineTuningCategory Category => FineTuningCategory.RankingBased;

    /// <inheritdoc/>
    public override bool RequiresRewardModel => false;

    /// <inheritdoc/>
    public override bool RequiresReferenceModel => true;

    /// <inheritdoc/>
    protected override void ValidateTrainingData(FineTuningData<T, TInput, TOutput> data)
    {
        base.ValidateTrainingData(data);

        if (!data.HasRankingData && !data.HasPairwisePreferenceData)
        {
            throw new ArgumentException(
                "RSO requires either RankedOutputs or pairwise preference data.",
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

                var batchLoss = await ComputeRSOLossAndUpdateAsync(policyModel, batch, beta, cancellationToken);
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

        if (!evaluationData.HasRankingData && !evaluationData.HasPairwisePreferenceData)
        {
            return metrics;
        }

        int correctRankings = 0;
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

                // Check pairwise ranking accuracy
                for (int j = 0; j < rankedOutputs.Length - 1; j++)
                {
                    var betterOutput = rankedOutputs[j];
                    var worseOutput = rankedOutputs[j + 1];

                    var betterLogProb = ComputeLogProbability(model, input, betterOutput);
                    var worseLogProb = ComputeLogProbability(model, input, worseOutput);

                    if (betterLogProb > worseLogProb)
                    {
                        correctRankings++;
                    }
                    totalPairs++;
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

                if (chosenLogProb > rejectedLogProb)
                {
                    correctRankings++;
                }
                totalPairs++;
            }
        }

        metrics.PreferenceAccuracy = totalPairs > 0 ? (double)correctRankings / totalPairs : 0.0;
        metrics.WinRate = metrics.PreferenceAccuracy;

        return await Task.FromResult(metrics);
    }

    /// <summary>
    /// Computes the RSO loss for a batch.
    /// </summary>
    private async Task<double> ComputeRSOLossAndUpdateAsync(
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
        int pairCount = 0;

        if (batch.HasRankingData)
        {
            // Sample pairs from ranked outputs using rejection sampling
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

                // Statistical rejection sampling: sample pairs based on rank difference
                var pairs = SamplePairsWithRejection(rankedOutputs);

                foreach (var (better, worse) in pairs)
                {
                    var loss = ComputePairLoss(policyModel, input, better, worse, beta);
                    totalLoss += loss;
                    pairCount++;
                }
            }
        }
        else if (batch.HasPairwisePreferenceData)
        {
            // Use pairwise data directly
            for (int i = 0; i < batch.Count; i++)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                var input = batch.Inputs[i];
                var chosen = batch.ChosenOutputs[i];
                var rejected = batch.RejectedOutputs[i];

                var loss = ComputePairLoss(policyModel, input, chosen, rejected, beta);
                totalLoss += loss;
                pairCount++;
            }
        }

        return await Task.FromResult(pairCount > 0 ? totalLoss / pairCount : 0.0);
    }

    /// <summary>
    /// Samples pairs from ranked outputs using statistical rejection sampling.
    /// </summary>
    private List<(TOutput Better, TOutput Worse)> SamplePairsWithRejection(TOutput[] rankedOutputs)
    {
        var pairs = new List<(TOutput, TOutput)>();
        int n = rankedOutputs.Length;

        // Sample pairs with probability proportional to rank difference
        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                // Acceptance probability based on rank difference
                double rankDiff = j - i;
                double acceptProb = Math.Min(1.0, rankDiff / (n - 1));

                if (Random.NextDouble() < acceptProb)
                {
                    pairs.Add((rankedOutputs[i], rankedOutputs[j]));
                }
            }
        }

        // Ensure at least one pair
        if (pairs.Count == 0 && n >= 2)
        {
            pairs.Add((rankedOutputs[0], rankedOutputs[n - 1]));
        }

        return pairs;
    }

    /// <summary>
    /// Computes the DPO-style loss for a preference pair.
    /// </summary>
    private double ComputePairLoss(
        IFullModel<T, TInput, TOutput> policyModel,
        TInput input,
        TOutput better,
        TOutput worse,
        double beta)
    {
        if (_referenceModel == null)
        {
            return 0.0;
        }

        var piBetterLogProb = ComputeLogProbability(policyModel, input, better);
        var piWorseLogProb = ComputeLogProbability(policyModel, input, worse);
        var refBetterLogProb = ComputeLogProbability(_referenceModel, input, better);
        var refWorseLogProb = ComputeLogProbability(_referenceModel, input, worse);

        var betterLogRatio = piBetterLogProb - refBetterLogProb;
        var worseLogRatio = piWorseLogProb - refWorseLogProb;

        var margin = beta * (betterLogRatio - worseLogRatio);
        return -LogSigmoid(margin);
    }
}
