using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// Partition MCMC — MCMC sampling over DAG partitions for structure learning.
/// </summary>
/// <remarks>
/// <para>
/// Partition MCMC extends Order MCMC by sampling over partitions of variables instead
/// of total orderings. A partition groups variables into layers; edges can go from earlier
/// layers to later layers. The partition space is between orderings and DAGs in granularity,
/// providing better mixing than Order MCMC while remaining efficient.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize: each variable in its own partition element (= total ordering)</item>
/// <item>Propose moves: split a partition element, merge two adjacent elements, or swap elements</item>
/// <item>For each partition, compute the optimal DAG via BIC-greedy parent selection</item>
/// <item>Accept/reject via Metropolis-Hastings with BIC-based scoring</item>
/// <item>After burn-in, accumulate edge posterior probabilities</item>
/// <item>Return edges with posterior probability > 0.5</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> This method groups variables into "layers" (partitions) where
/// variables in earlier layers can cause variables in later layers but not vice versa.
/// It explores different layer arrangements to find plausible causal structures.
/// </para>
/// <para>
/// Reference: Kuipers and Moffa (2017), "Partition MCMC for Inference on Acyclic
/// Digraphs", JASA.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Partition MCMC for Inference on Acyclic Digraphs", "https://doi.org/10.1080/01621459.2015.1133426", Year = 2017, Authors = "Jack Kuipers, Giusi Moffa")]
public class PartitionMCMCAlgorithm<T> : BayesianCausalBase<T>
{
    private readonly int _maxParents;
    private readonly int _burnIn;

    /// <inheritdoc/>
    public override string Name => "PartitionMCMC";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public PartitionMCMCAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyBayesianOptions(options);
        if (NumSamples < 1)
            throw new ArgumentException("NumSamples must be at least 1.");
        _maxParents = options?.MaxParents ?? 5;
        if (_maxParents < 1)
            throw new ArgumentException("MaxParents must be at least 1.");
        _burnIn = Math.Max(NumSamples / 5, 100);
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);
        var cov = ComputeCovarianceMatrix(data);

        // Pairwise OLS weight of every edge, dropped when |weight| <= 1e-6; it depends only on the data,
        // so it is computed once and each sampled DAG keeps the entries of its selected parent edges.
        var weights = ComputePairwiseOLSWeights(cov);
        var significant = KeepWhereAbove(Engine.TensorAbs(weights), 1e-6, weights).Reshape(new[] { d * d });

        // Initialize: each variable in its own partition element
        var partition = new List<List<int>>();
        var indices = Enumerable.Range(0, d).ToArray();
        for (int i = d - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }
        foreach (int v in indices)
            partition.Add([v]);

        double currentScore = ComputePartitionScore(data, partition, d);
        var edgeCounts = new Tensor<T>(new[] { d, d });
        int sampleCount = 0;

        for (int iter = 0; iter < NumSamples + _burnIn; iter++)
        {
            var proposed = ProposeMove(partition, d, rng);
            double proposedScore = ComputePartitionScore(data, proposed, d);

            double logAccept = proposedScore - currentScore;
            if (logAccept >= 0 || Math.Log(rng.NextDouble()) < logAccept)
            {
                partition = proposed;
                currentScore = proposedScore;
            }

            if (iter >= _burnIn)
            {
                var dag = BuildDAGFromPartition(data, partition, d, significant);
                edgeCounts = AccumulateEdgeCounts(edgeCounts, dag);
                sampleCount++;
            }
        }

        if (sampleCount == 0) return new Matrix<T>(d, d);
        return MajorityEdgeFrequencies(edgeCounts, sampleCount);
    }

    private List<List<int>> ProposeMove(List<List<int>> partition, int d, Random rng)
    {
        var proposed = partition.Select(p => new List<int>(p)).ToList();
        int moveType = rng.Next(3);

        if (moveType == 0 && proposed.Count > 1)
        {
            int pos = rng.Next(proposed.Count - 1);
            proposed[pos].AddRange(proposed[pos + 1]);
            proposed.RemoveAt(pos + 1);
        }
        else if (moveType == 1)
        {
            var candidates = proposed.Where(p => p.Count > 1).ToList();
            if (candidates.Count > 0)
            {
                var target = candidates[rng.Next(candidates.Count)];
                int idx = proposed.IndexOf(target);
                int splitPoint = rng.Next(1, target.Count);
                var part1 = target.Take(splitPoint).ToList();
                var part2 = target.Skip(splitPoint).ToList();
                proposed[idx] = part1;
                proposed.Insert(idx + 1, part2);
            }
        }
        else if (proposed.Count > 1)
        {
            int pos = rng.Next(proposed.Count - 1);
            if (proposed[pos].Count > 0 && proposed[pos + 1].Count > 0)
            {
                int i1 = rng.Next(proposed[pos].Count);
                int i2 = rng.Next(proposed[pos + 1].Count);
                (proposed[pos][i1], proposed[pos + 1][i2]) = (proposed[pos + 1][i2], proposed[pos][i1]);
            }
        }

        return proposed;
    }

    private double ComputePartitionScore(Matrix<T> data, List<List<int>> partition, int d)
    {
        double totalScore = 0;
        var predecessors = new HashSet<int>();

        foreach (var element in partition)
        {
            foreach (int target in element)
            {
                var parents = SelectBestParents(data, target, predecessors);
                totalScore -= ComputeBICScore(data, target, parents);
            }
            foreach (int v in element)
                predecessors.Add(v);
        }

        return totalScore;
    }

    private int[] SelectBestParents(Matrix<T> data, int target, HashSet<int> candidates)
    {
        var parents = new List<int>();
        double bestScore = ComputeBICScore(data, target, []);

        foreach (int candidate in candidates)
        {
            if (parents.Count >= _maxParents) break;
            var trial = parents.Concat([candidate]).ToArray();
            double trialScore = ComputeBICScore(data, target, trial);

            if (trialScore < bestScore)
            {
                parents.Add(candidate);
                bestScore = trialScore;
            }
        }

        return [.. parents];
    }

    private Matrix<T> BuildDAGFromPartition(Matrix<T> data, List<List<int>> partition, int d, Tensor<T> significant)
    {
        var isParent = new bool[d * d];
        var predecessors = new HashSet<int>();

        foreach (var element in partition)
        {
            foreach (int target in element)
            {
                var parents = SelectBestParents(data, target, predecessors);
                foreach (int parent in parents)
                    isParent[parent * d + target] = true;
            }
            foreach (int v in element)
                predecessors.Add(v);
        }

        // significant: the flattened [d * d] pairwise OLS weights with |weight| <= 1e-6 already zeroed.
        return Engine.TensorWhere(isParent, significant, Engine.TensorZerosLike(significant))
            .Reshape(new[] { d, d }).ToMatrix();
    }
}
