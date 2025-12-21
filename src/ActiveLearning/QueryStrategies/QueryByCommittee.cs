using AiDotNet.ActiveLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ActiveLearning.QueryStrategies;

/// <summary>
/// Query-by-Committee (QBC) strategy for active learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Query-by-Committee trains multiple models (the "committee")
/// and selects examples where these models disagree the most. The idea is that if
/// models trained on the same data make different predictions, the example must be
/// informative or difficult.</para>
///
/// <para><b>How it works:</b>
/// 1. Train multiple models on the current labeled data
///    - Can use different random initializations
///    - Can use different architectures
///    - Can use bootstrap sampling of training data
/// 2. For each unlabeled example, get predictions from all committee members
/// 3. Measure disagreement between committee members
/// 4. Select examples with highest disagreement
/// </para>
///
/// <para><b>Disagreement Measures:</b>
/// - <b>Vote Entropy:</b> Entropy of the vote distribution across classes
/// - <b>Consensus Entropy:</b> Average entropy of individual predictions
/// - <b>KL Divergence:</b> Average divergence from committee mean
/// </para>
///
/// <para><b>Advantages:</b>
/// - Doesn't rely on model uncertainty calibration
/// - Can be more robust than single-model uncertainty
/// - Captures epistemic (model) uncertainty, not just aleatoric (data) uncertainty
/// </para>
///
/// <para><b>Disadvantages:</b>
/// - More expensive (must train and run multiple models)
/// - May not help if all committee members make the same mistakes
/// </para>
///
/// <para><b>Reference:</b> Seung et al. "Query by committee" (1992)</para>
/// </remarks>
public class QueryByCommittee<T, TInput, TOutput> : IQueryStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Disagreement measure to use.
    /// </summary>
    public enum DisagreementMeasure
    {
        /// <summary>
        /// Vote entropy: entropy of vote distribution
        /// </summary>
        VoteEntropy,

        /// <summary>
        /// Consensus entropy: average entropy of predictions
        /// </summary>
        ConsensusEntropy,

        /// <summary>
        /// KL divergence: average KL from mean prediction
        /// </summary>
        KLDivergence
    }

    private readonly List<IFullModel<T, TInput, TOutput>> _committee;
    private readonly DisagreementMeasure _measure;

    /// <summary>
    /// Initializes a new Query-by-Committee strategy.
    /// </summary>
    /// <param name="committeeSize">Number of models in the committee.</param>
    /// <param name="measure">The disagreement measure to use.</param>
    public QueryByCommittee(
        int committeeSize = 5,
        DisagreementMeasure measure = DisagreementMeasure.VoteEntropy)
    {
        if (committeeSize < 2)
            throw new ArgumentException("Committee must have at least 2 members", nameof(committeeSize));

        _committee = new List<IFullModel<T, TInput, TOutput>>(committeeSize);
        _measure = measure;
    }

    /// <summary>
    /// Initializes with an existing committee of models.
    /// </summary>
    /// <param name="committee">The committee of models to use.</param>
    /// <param name="measure">The disagreement measure to use.</param>
    public QueryByCommittee(
        List<IFullModel<T, TInput, TOutput>> committee,
        DisagreementMeasure measure = DisagreementMeasure.VoteEntropy)
    {
        _committee = committee ?? throw new ArgumentNullException(nameof(committee));
        _measure = measure;

        if (_committee.Count < 2)
            throw new ArgumentException("Committee must have at least 2 members");
    }

    /// <inheritdoc/>
    public string Name => $"QueryByCommittee-{_measure}";

    /// <inheritdoc/>
    public Vector<T> ScoreExamples(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledData,
        IDataset<T, TInput, TOutput>? labeledData = null)
    {
        if (unlabeledData == null)
            throw new ArgumentNullException(nameof(unlabeledData));

        if (_committee.Count == 0)
        {
            throw new InvalidOperationException(
                "Committee has not been initialized. Use SetCommittee() to provide trained models before scoring examples.");
        }

        int numExamples = unlabeledData.Count;
        var scores = new T[numExamples];

        for (int i = 0; i < numExamples; i++)
        {
            var input = unlabeledData.GetInput(i);

            // Collect predictions from all committee members
            var predictions = _committee
                .Select(member => ConversionsHelper.ConvertToVector<T, TOutput>(member.Predict(input)))
                .ToList();

            // Compute disagreement score based on measure
            scores[i] = _measure switch
            {
                DisagreementMeasure.VoteEntropy => ComputeVoteEntropy(predictions),
                DisagreementMeasure.ConsensusEntropy => ComputeConsensusEntropy(predictions),
                DisagreementMeasure.KLDivergence => ComputeKLDivergence(predictions),
                _ => NumOps.Zero
            };
        }

        return new Vector<T>(scores);
    }

    /// <inheritdoc/>
    public Vector<int> SelectBatch(
        IFullModel<T, TInput, TOutput> model,
        IDataset<T, TInput, TOutput> unlabeledData,
        int k,
        IDataset<T, TInput, TOutput>? labeledData = null)
    {
        var scores = ScoreExamples(model, unlabeledData, labeledData);

        // Select top-k examples with highest disagreement
        var indexedScores = scores.ToArray()
            .Select((score, index) => (Score: Convert.ToDouble(score), Index: index))
            .OrderByDescending(x => x.Score)
            .Take(k)
            .Select(x => x.Index)
            .ToArray();

        return new Vector<int>(indexedScores);
    }

    /// <summary>
    /// Computes vote entropy disagreement measure.
    /// </summary>
    /// <param name="predictions">Predictions from all committee members.</param>
    /// <returns>Disagreement score (higher = more disagreement).</returns>
    /// <remarks>
    /// <para>Vote entropy measures disagreement by:
    /// 1. Count votes for each class across committee members
    /// 2. Normalize to get vote distribution
    /// 3. Compute entropy of vote distribution
    /// </para>
    /// <para>High entropy means committee is split, low entropy means consensus.</para>
    /// </remarks>
    private T ComputeVoteEntropy(List<Vector<T>> predictions)
    {
        if (predictions == null || predictions.Count == 0)
            return NumOps.Zero;

        int numClasses = predictions[0].Length;
        var votes = new int[numClasses];

        // Count votes for each class (argmax of each prediction)
        foreach (var pred in predictions)
        {
            int maxClass = 0;
            double maxProb = Convert.ToDouble(pred[0]);

            for (int c = 1; c < numClasses; c++)
            {
                double prob = Convert.ToDouble(pred[c]);
                if (prob > maxProb)
                {
                    maxProb = prob;
                    maxClass = c;
                }
            }

            votes[maxClass]++;
        }

        // Compute entropy of vote distribution
        T entropy = NumOps.Zero;
        int totalVotes = predictions.Count;

        for (int c = 0; c < numClasses; c++)
        {
            if (votes[c] > 0)
            {
                double prob = (double)votes[c] / totalVotes;
                double logProb = Math.Log(prob);
                T term = NumOps.FromDouble(-prob * logProb);
                entropy = NumOps.Add(entropy, term);
            }
        }

        return entropy;
    }

    /// <summary>
    /// Computes consensus entropy disagreement measure.
    /// </summary>
    /// <param name="predictions">Predictions from all committee members.</param>
    /// <returns>Disagreement score (higher = more disagreement).</returns>
    /// <remarks>
    /// <para>Consensus entropy averages the entropy of each committee member's prediction.
    /// This considers the uncertainty within each model, not just disagreement.</para>
    /// </remarks>
    private T ComputeConsensusEntropy(List<Vector<T>> predictions)
    {
        if (predictions == null || predictions.Count == 0)
            return NumOps.Zero;

        T totalEntropy = NumOps.Zero;

        foreach (var pred in predictions)
        {
            T entropy = NumOps.Zero;

            for (int c = 0; c < pred.Length; c++)
            {
                double prob = Convert.ToDouble(pred[c]);
                if (prob > 1e-10)
                {
                    double logProb = Math.Log(prob);
                    T term = NumOps.FromDouble(-prob * logProb);
                    entropy = NumOps.Add(entropy, term);
                }
            }

            totalEntropy = NumOps.Add(totalEntropy, entropy);
        }

        // Average entropy across committee
        return NumOps.Divide(totalEntropy, NumOps.FromDouble(predictions.Count));
    }

    /// <summary>
    /// Computes KL divergence disagreement measure.
    /// </summary>
    /// <param name="predictions">Predictions from all committee members.</param>
    /// <returns>Disagreement score (higher = more disagreement).</returns>
    /// <remarks>
    /// <para>KL divergence measures how much each committee member's prediction
    /// differs from the average prediction. Steps:
    /// 1. Compute mean prediction across committee
    /// 2. Compute KL(member || mean) for each member
    /// 3. Average KL divergences
    /// </para>
    /// </remarks>
    private T ComputeKLDivergence(List<Vector<T>> predictions)
    {
        if (predictions == null || predictions.Count == 0)
            return NumOps.Zero;

        int numClasses = predictions[0].Length;

        // Compute mean prediction
        var meanPred = new T[numClasses];
        for (int c = 0; c < numClasses; c++)
        {
            T sum = NumOps.Zero;
            foreach (var pred in predictions)
            {
                sum = NumOps.Add(sum, pred[c]);
            }
            meanPred[c] = NumOps.Divide(sum, NumOps.FromDouble(predictions.Count));
        }

        // Compute average KL divergence from mean
        T totalKL = NumOps.Zero;

        foreach (var pred in predictions)
        {
            T kl = NumOps.Zero;

            for (int c = 0; c < numClasses; c++)
            {
                double p = Convert.ToDouble(pred[c]);
                double q = Convert.ToDouble(meanPred[c]) + 1e-10; // Avoid division by zero

                if (p > 1e-10)
                {
                    double logRatio = Math.Log(p / q);
                    T term = NumOps.FromDouble(p * logRatio);
                    kl = NumOps.Add(kl, term);
                }
            }

            totalKL = NumOps.Add(totalKL, kl);
        }

        return NumOps.Divide(totalKL, NumOps.FromDouble(predictions.Count));
    }

    /// <summary>
    /// Gets the committee of models.
    /// </summary>
    public IReadOnlyList<IFullModel<T, TInput, TOutput>> Committee => _committee.AsReadOnly();

    /// <summary>
    /// Sets the committee of models.
    /// </summary>
    public void SetCommittee(List<IFullModel<T, TInput, TOutput>> committee)
    {
        if (committee == null || committee.Count < 2)
            throw new ArgumentException("Committee must have at least 2 members");

        _committee.Clear();
        _committee.AddRange(committee);
    }
}
