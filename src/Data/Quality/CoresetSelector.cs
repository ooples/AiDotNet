using AiDotNet.Helpers;

namespace AiDotNet.Data.Quality;

/// <summary>
/// Selects a representative coreset from a dataset using distance-based strategies.
/// </summary>
/// <remarks>
/// <para>
/// Coreset selection finds a small representative subset that approximates the full dataset.
/// Supports greedy facility location, k-Center, and random selection strategies.
/// Works on pre-computed feature embeddings (distance matrix).
/// </para>
/// </remarks>
public class CoresetSelector
{
    private readonly CoresetSelectorOptions _options;
    private readonly Random _random;

    public CoresetSelector(CoresetSelectorOptions? options = null)
    {
        _options = options ?? new CoresetSelectorOptions();
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Selects coreset indices from pre-computed pairwise distances.
    /// </summary>
    /// <param name="distances">Pairwise distance matrix [N x N].</param>
    /// <returns>Indices of selected coreset samples.</returns>
    public List<int> Select(double[,] distances)
    {
        int n = distances.GetLength(0);
        int targetSize = Math.Max(1, (int)(n * _options.SelectionRatio));

        return _options.Strategy switch
        {
            CoresetStrategy.Greedy => GreedyFacilityLocation(distances, n, targetSize),
            CoresetStrategy.KCenter => KCenterSelect(distances, n, targetSize),
            CoresetStrategy.Random => RandomSelect(n, targetSize),
            _ => RandomSelect(n, targetSize)
        };
    }

    /// <summary>
    /// Selects coreset indices from embedding vectors using Euclidean distance.
    /// </summary>
    /// <param name="embeddings">Feature embeddings, one per sample.</param>
    /// <returns>Indices of selected coreset samples.</returns>
    public List<int> Select(double[][] embeddings)
    {
        int n = embeddings.Length;
        var distances = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = EuclideanDistance(embeddings[i], embeddings[j]);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        return Select(distances);
    }

    private List<int> GreedyFacilityLocation(double[,] distances, int n, int targetSize)
    {
        var selected = new List<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, n));

        // Start with the most central point
        int firstIdx = 0;
        double minTotalDist = double.MaxValue;
        for (int i = 0; i < n; i++)
        {
            double totalDist = 0;
            for (int j = 0; j < n; j++)
                totalDist += distances[i, j];
            if (totalDist < minTotalDist)
            {
                minTotalDist = totalDist;
                firstIdx = i;
            }
        }

        selected.Add(firstIdx);
        remaining.Remove(firstIdx);

        // Greedy: pick point that maximizes coverage gain
        var minDistToSelected = new double[n];
        for (int d = 0; d < n; d++)
            minDistToSelected[d] = double.MaxValue;

        for (int i = 0; i < n; i++)
            minDistToSelected[i] = distances[i, firstIdx];

        while (selected.Count < targetSize && remaining.Count > 0)
        {
            int bestIdx = -1;
            double bestGain = double.MinValue;

            foreach (int candidate in remaining)
            {
                double gain = 0;
                foreach (int other in remaining)
                {
                    double reduction = minDistToSelected[other] - distances[other, candidate];
                    if (reduction > 0) gain += reduction;
                }

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestIdx = candidate;
                }
            }

            if (bestIdx < 0) break;

            selected.Add(bestIdx);
            remaining.Remove(bestIdx);

            // Update minimum distances
            for (int i = 0; i < n; i++)
            {
                double d = distances[i, bestIdx];
                if (d < minDistToSelected[i])
                    minDistToSelected[i] = d;
            }
        }

        return selected;
    }

    private List<int> KCenterSelect(double[,] distances, int n, int targetSize)
    {
        var selected = new List<int>();
        var minDistToSelected = new double[n];
        for (int d = 0; d < n; d++)
            minDistToSelected[d] = double.MaxValue;

        // Start with random point
        int first = _random.Next(n);
        selected.Add(first);

        for (int i = 0; i < n; i++)
            minDistToSelected[i] = distances[i, first];

        while (selected.Count < targetSize)
        {
            // Pick the point farthest from any selected point
            int farthest = -1;
            double maxDist = -1;

            for (int i = 0; i < n; i++)
            {
                if (selected.Contains(i)) continue;
                if (minDistToSelected[i] > maxDist)
                {
                    maxDist = minDistToSelected[i];
                    farthest = i;
                }
            }

            if (farthest < 0) break;

            selected.Add(farthest);

            for (int i = 0; i < n; i++)
            {
                double d = distances[i, farthest];
                if (d < minDistToSelected[i])
                    minDistToSelected[i] = d;
            }
        }

        return selected;
    }

    private List<int> RandomSelect(int n, int targetSize)
    {
        var indices = Enumerable.Range(0, n).ToList();
        // Fisher-Yates shuffle
        for (int i = indices.Count - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        return indices.Take(targetSize).ToList();
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        double sum = 0;
        int len = Math.Min(a.Length, b.Length);
        for (int i = 0; i < len; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}
