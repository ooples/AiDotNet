namespace AiDotNet.Data.Quality;

/// <summary>
/// Performs dataset distillation to synthesize a compact representative dataset.
/// </summary>
/// <remarks>
/// <para>
/// Dataset distillation creates a small synthetic dataset that captures the essential
/// patterns of the original training data. Uses a gradient-based optimization approach
/// to iteratively refine synthetic samples toward class centroids.
/// </para>
/// </remarks>
public class DatasetDistiller
{
    private readonly DatasetDistillerOptions _options;
    private readonly Random _random;

    public DatasetDistiller(DatasetDistillerOptions? options = null)
    {
        _options = options ?? new DatasetDistillerOptions();
        _options.Validate();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Distills a dataset by computing class-wise centroids and refining them.
    /// </summary>
    /// <param name="features">Feature vectors for each sample. Shape: [numSamples][featureDim].</param>
    /// <param name="labels">Integer class labels for each sample. Shape: [numSamples].</param>
    /// <returns>Tuple of (distilled features, distilled labels).</returns>
    public (double[][] Features, int[] Labels) Distill(double[][] features, int[] labels)
    {
        if (features == null || features.Length == 0)
            throw new ArgumentException("Features must not be null or empty.", nameof(features));
        if (labels == null || labels.Length == 0)
            throw new ArgumentException("Labels must not be null or empty.", nameof(labels));
        if (features.Length != labels.Length)
            throw new ArgumentException($"Features length ({features.Length}) must match labels length ({labels.Length}).");

        // Group samples by class
        var classSamples = new Dictionary<int, List<int>>();
        for (int i = 0; i < labels.Length; i++)
        {
            if (!classSamples.TryGetValue(labels[i], out var list))
            {
                list = new List<int>();
                classSamples[labels[i]] = list;
            }
            list.Add(i);
        }

        if (features[0] == null)
            throw new ArgumentException("First feature row must not be null.", nameof(features));
        int featureDim = features[0].Length;
        var distilledFeatures = new List<double[]>();
        var distilledLabels = new List<int>();

        foreach (var (classLabel, sampleIndices) in classSamples)
        {
            int numSynthetic = Math.Min(_options.SamplesPerClass, sampleIndices.Count);

            // Initialize synthetic samples as random class members
            var synthetic = new double[numSynthetic][];
            for (int s = 0; s < numSynthetic; s++)
            {
                int sourceIdx = sampleIndices[_random.Next(sampleIndices.Count)];
                synthetic[s] = (double[])features[sourceIdx].Clone();
            }

            // Compute class centroid
            var centroid = new double[featureDim];
            foreach (int idx in sampleIndices)
            {
                for (int d = 0; d < featureDim; d++)
                    centroid[d] += features[idx][d];
            }
            for (int d = 0; d < featureDim; d++)
                centroid[d] /= sampleIndices.Count;

            // Iteratively move synthetic samples toward centroid with diversity
            for (int step = 0; step < _options.NumSteps; step++)
            {
                for (int s = 0; s < numSynthetic; s++)
                {
                    // Gradient toward centroid
                    for (int d = 0; d < featureDim; d++)
                    {
                        double gradCentroid = centroid[d] - synthetic[s][d];

                        // Repulsion from other synthetic samples for diversity
                        double gradDiversity = 0;
                        for (int other = 0; other < numSynthetic; other++)
                        {
                            if (other == s) continue;
                            double diff = synthetic[s][d] - synthetic[other][d];
                            double dist = Math.Max(Math.Abs(diff), 1e-8);
                            gradDiversity += diff / (dist * dist);
                        }

                        double lr = _options.DistillLearningRate * (1.0 - (double)step / _options.NumSteps);
                        synthetic[s][d] += lr * (gradCentroid + 0.1 * gradDiversity);
                    }
                }
            }

            for (int s = 0; s < numSynthetic; s++)
            {
                distilledFeatures.Add(synthetic[s]);
                distilledLabels.Add(classLabel);
            }
        }

        return (distilledFeatures.ToArray(), distilledLabels.ToArray());
    }
}
