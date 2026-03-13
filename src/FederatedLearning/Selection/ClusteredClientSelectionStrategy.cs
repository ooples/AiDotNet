using AiDotNet.Models;

namespace AiDotNet.FederatedLearning.Selection;

/// <summary>
/// Cluster-based client selection using simple k-means over per-client embeddings.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> This strategy tries to pick clients from different "types" of behavior by clustering
/// clients into groups and sampling from each cluster.
/// </remarks>
public sealed class ClusteredClientSelectionStrategy : ClientSelectionStrategyBase
{
    private readonly int _clusterCount;
    private readonly int _iterations;

    public ClusteredClientSelectionStrategy(int clusterCount = 3, int iterations = 5)
    {
        if (clusterCount <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(clusterCount), "Cluster count must be positive.");
        }

        if (iterations <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(iterations), "Iterations must be positive.");
        }

        _clusterCount = clusterCount;
        _iterations = iterations;
    }

    public override List<int> SelectClients(ClientSelectionRequest request)
    {
        if (request == null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        var candidates = request.CandidateClientIds ?? Array.Empty<int>();
        int desired = GetDesiredClientCount(candidates, request.FractionToSelect);

        var embeddings = request.ClientEmbeddings;
        if (embeddings == null || embeddings.Count == 0)
        {
            return ShuffleAndTake(candidates, desired, request.Random);
        }

        var usable = candidates.Distinct().Where(id => embeddings.ContainsKey(id)).ToList();
        if (usable.Count == 0)
        {
            return ShuffleAndTake(candidates, desired, request.Random);
        }

        int k = Math.Min(_clusterCount, usable.Count);
        var centroids = InitializeCentroids(usable, embeddings, k, request.Random);
        var assignments = new Dictionary<int, int>();

        for (int iter = 0; iter < _iterations; iter++)
        {
            // Assign.
            foreach (var id in usable)
            {
                assignments[id] = GetNearestCentroidIndex(embeddings[id], centroids);
            }

            // Recompute.
            for (int c = 0; c < k; c++)
            {
                var members = usable.Where(id => assignments[id] == c).ToList();
                if (members.Count == 0)
                {
                    continue;
                }

                centroids[c] = MeanEmbedding(members, embeddings);
            }
        }

        var clusters = new Dictionary<int, List<int>>();
        foreach (var id in usable)
        {
            int c = assignments.TryGetValue(id, out var cc) ? cc : 0;
            if (!clusters.TryGetValue(c, out var list))
            {
                list = new List<int>();
                clusters[c] = list;
            }
            list.Add(id);
        }

        // Sample roughly equally from each cluster.
        int baseCount = Math.Max(1, desired / clusters.Count);
        var selected = new List<int>(desired);

        foreach (var cluster in clusters.OrderBy(kvp => kvp.Key))
        {
            if (selected.Count >= desired)
            {
                break;
            }

            int take = Math.Min(baseCount, desired - selected.Count);
            selected.AddRange(ShuffleAndTake(cluster.Value, Math.Min(take, cluster.Value.Count), request.Random));
        }

        if (selected.Count < desired)
        {
            var remaining = usable.Except(selected).ToList();
            if (remaining.Count > 0)
            {
                selected.AddRange(ShuffleAndTake(remaining, Math.Min(desired - selected.Count, remaining.Count), request.Random));
            }
        }

        selected = selected.Distinct().ToList();
        if (selected.Count > desired)
        {
            selected = selected.Take(desired).ToList();
        }
        selected.Sort();
        return selected;
    }

    public override string GetStrategyName() => "Clustered";

    private static List<double[]> InitializeCentroids(List<int> usable, IReadOnlyDictionary<int, double[]> embeddings, int k, Random random)
    {
        var chosen = usable.OrderBy(_ => random.Next()).Take(k).ToList();
        var centroids = new List<double[]>(k);
        foreach (var id in chosen)
        {
            centroids.Add((double[])embeddings[id].Clone());
        }
        return centroids;
    }

    private static int GetNearestCentroidIndex(double[] point, List<double[]> centroids)
    {
        int best = 0;
        double bestDist = double.PositiveInfinity;

        for (int i = 0; i < centroids.Count; i++)
        {
            double d = SquaredDistance(point, centroids[i]);
            if (d < bestDist)
            {
                bestDist = d;
                best = i;
            }
        }

        return best;
    }

    private static double SquaredDistance(double[] a, double[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        double sum = 0.0;
        for (int i = 0; i < n; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }

        // Penalize dimension mismatch (simple).
        if (a.Length != b.Length)
        {
            sum += Math.Abs(a.Length - b.Length);
        }

        return sum;
    }

    private static double[] MeanEmbedding(List<int> members, IReadOnlyDictionary<int, double[]> embeddings)
    {
        int dim = embeddings[members[0]].Length;
        var mean = new double[dim];
        foreach (var embedding in members.Select(id => embeddings[id]))
        {
            int n = Math.Min(dim, embedding.Length);
            for (int i = 0; i < n; i++)
            {
                mean[i] += embedding[i];
            }
        }

        double inv = 1.0 / members.Count;
        for (int i = 0; i < dim; i++)
        {
            mean[i] *= inv;
        }

        return mean;
    }
}
