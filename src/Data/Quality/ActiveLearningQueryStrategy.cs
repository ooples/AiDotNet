using AiDotNet.Helpers;

namespace AiDotNet.Data.Quality;

/// <summary>
/// Selects the most informative unlabeled samples for annotation using active learning.
/// </summary>
/// <remarks>
/// <para>
/// Active learning reduces labeling costs by selecting samples that maximize model improvement.
/// Supports uncertainty sampling, margin sampling, least confidence, and BALD strategies.
/// Works on model prediction probabilities from the unlabeled pool.
/// </para>
/// </remarks>
public class ActiveLearningQueryStrategy
{
    private readonly ActiveLearningQueryStrategyOptions _options;
    private readonly Random _random;

    public ActiveLearningQueryStrategy(ActiveLearningQueryStrategyOptions? options = null)
    {
        _options = options ?? new ActiveLearningQueryStrategyOptions();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Selects the most informative samples from an unlabeled pool.
    /// </summary>
    /// <param name="predictions">Prediction probabilities for each unlabeled sample. Shape: [numSamples][numClasses].</param>
    /// <returns>Indices into the predictions array of selected samples.</returns>
    public List<int> Query(double[][] predictions)
    {
        if (predictions.Length == 0)
            return new List<int>();

        // Validate predictions are non-empty and have consistent shape
        int numClasses = predictions[0].Length;
        if (numClasses == 0)
            throw new ArgumentException("Prediction vectors must have at least one class.", nameof(predictions));
        for (int i = 1; i < predictions.Length; i++)
        {
            if (predictions[i].Length != numClasses)
                throw new ArgumentException(
                    $"Prediction at index {i} has {predictions[i].Length} classes, expected {numClasses}.",
                    nameof(predictions));
        }

        int querySize = Math.Min(_options.QueryBatchSize, predictions.Length);

        return _options.Strategy switch
        {
            QueryStrategy.Uncertainty => QueryByUncertainty(predictions, querySize),
            QueryStrategy.Margin => QueryByMargin(predictions, querySize),
            QueryStrategy.LeastConfidence => QueryByLeastConfidence(predictions, querySize),
            QueryStrategy.BALD => QueryByUncertainty(predictions, querySize), // BALD requires MC dropout passes
            QueryStrategy.Random => QueryRandom(predictions.Length, querySize),
            _ => QueryByUncertainty(predictions, querySize)
        };
    }

    /// <summary>
    /// Selects samples using BALD (Bayesian Active Learning by Disagreement).
    /// Requires multiple stochastic forward pass predictions.
    /// </summary>
    /// <param name="mcPredictions">MC Dropout predictions. Shape: [numPasses][numSamples][numClasses].</param>
    /// <returns>Indices of selected samples.</returns>
    public List<int> QueryBALD(double[][][] mcPredictions)
    {
        int numSamples = mcPredictions[0].Length;
        int numClasses = mcPredictions[0][0].Length;
        int numPasses = mcPredictions.Length;
        int querySize = Math.Min(_options.QueryBatchSize, numSamples);

        var baldScores = new double[numSamples];

        for (int s = 0; s < numSamples; s++)
        {
            // Mean prediction across MC passes
            var meanPred = new double[numClasses];
            for (int p = 0; p < numPasses; p++)
            {
                for (int c = 0; c < numClasses; c++)
                    meanPred[c] += mcPredictions[p][s][c];
            }
            for (int c = 0; c < numClasses; c++)
                meanPred[c] /= numPasses;

            // Total entropy H[y|x]
            double totalEntropy = ComputeEntropy(meanPred);

            // Average conditional entropy E[H[y|x,w]]
            double avgConditionalEntropy = 0;
            for (int p = 0; p < numPasses; p++)
            {
                avgConditionalEntropy += ComputeEntropy(mcPredictions[p][s]);
            }
            avgConditionalEntropy /= numPasses;

            // BALD = mutual information = total entropy - conditional entropy
            baldScores[s] = totalEntropy - avgConditionalEntropy;
        }

        return TopKIndices(baldScores, querySize, descending: true);
    }

    private List<int> QueryByUncertainty(double[][] predictions, int querySize)
    {
        var entropies = new double[predictions.Length];
        for (int i = 0; i < predictions.Length; i++)
        {
            entropies[i] = ComputeEntropy(predictions[i]);
        }
        return TopKIndices(entropies, querySize, descending: true);
    }

    private List<int> QueryByMargin(double[][] predictions, int querySize)
    {
        var margins = new double[predictions.Length];
        for (int i = 0; i < predictions.Length; i++)
        {
            var sorted = predictions[i].OrderByDescending(p => p).ToArray();
            margins[i] = sorted.Length >= 2 ? sorted[0] - sorted[1] : 1.0;
        }
        // Small margin = more informative
        return TopKIndices(margins, querySize, descending: false);
    }

    private List<int> QueryByLeastConfidence(double[][] predictions, int querySize)
    {
        var confidences = new double[predictions.Length];
        for (int i = 0; i < predictions.Length; i++)
        {
            confidences[i] = predictions[i].Max();
        }
        // Low confidence = more informative
        return TopKIndices(confidences, querySize, descending: false);
    }

    private List<int> QueryRandom(int numSamples, int querySize)
    {
        var indices = Enumerable.Range(0, numSamples).ToList();
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
        return indices.Take(querySize).ToList();
    }

    private static double ComputeEntropy(double[] probs)
    {
        double entropy = 0;
        foreach (double p in probs)
        {
            if (p > 1e-10)
                entropy -= p * Math.Log(p);
        }
        return entropy;
    }

    private static List<int> TopKIndices(double[] scores, int k, bool descending)
    {
        var indexed = scores
            .Select((score, idx) => (Score: score, Index: idx));

        var sorted = descending
            ? indexed.OrderByDescending(x => x.Score)
            : indexed.OrderBy(x => x.Score);

        return sorted.Take(k).Select(x => x.Index).ToList();
    }
}
