using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.ScoreBased;

/// <summary>
/// BOSS (Best Order Score Search) — efficient permutation-based structure learning.
/// </summary>
/// <remarks>
/// <para>
/// BOSS combines permutation search with score-based evaluation. It maintains a variable
/// ordering and iteratively improves it by moving each variable to the position in the
/// ordering that maximizes the total BIC score. This "best position" operation is the
/// key difference from GRaSP's adjacent transpositions.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize with a variable ordering based on marginal variance (ascending)</item>
/// <item>For each variable v in the ordering:</item>
/// <item>  Remove v from its current position</item>
/// <item>  Try inserting v at every possible position (0 to d-1)</item>
/// <item>  Place v at the position that maximizes total BIC score</item>
/// <item>Repeat the full pass until the ordering stabilizes</item>
/// <item>Extract the DAG from the final ordering using greedy parent selection</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> BOSS is a modern, fast algorithm that finds causal structures
/// by efficiently searching through possible variable orderings. For each variable, it
/// finds the single best position in the ordering, making large improvements per step.
/// </para>
/// <para>
/// Reference: Andrews et al. (2022), "Fast Scalable and Accurate Discovery of DAGs
/// Using the Best Order Score Search and Grow-Shrink Trees".
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow-Shrink Trees", "https://arxiv.org/abs/2310.17679", Year = 2023, Authors = "Bryan Andrews, Joseph Ramsey, Ruben Sanchez-Romero, Jazmin Camchong, Erich Kummerfeld")]
public class BOSSAlgorithm<T> : ScoreBasedBase<T>
{
    /// <inheritdoc/>
    public override string Name => "BOSS";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    /// <summary>
    /// Initializes BOSS with optional configuration.
    /// </summary>
    public BOSSAlgorithm(CausalDiscoveryOptions? options = null) { ApplyScoreOptions(options); }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        if (n < 2 || d < 2) return new Matrix<T>(d, d);

        // Initialize ordering using marginal variance (ascending — exogenous variables first)
        var ordering = new List<int>(Enumerable.Range(0, d));
        SortByMarginalVariance(data, ordering, n);

        bool changed = true;
        int iteration = 0;

        while (changed && iteration < MaxIterations)
        {
            changed = false;
            iteration++;

            // Iterate over a snapshot of the ordering to avoid mutating during iteration
            var snapshot = new List<int>(ordering);
            for (int idx = 0; idx < d; idx++)
            {
                int v = snapshot[idx];

                // Remove v from current position
                int currentPos = ordering.IndexOf(v);
                if (currentPos < 0) continue;
                ordering.RemoveAt(currentPos);

                // Find the best position for v
                double bestScore = double.NegativeInfinity;
                int bestPos = idx;

                for (int pos = 0; pos <= ordering.Count; pos++)
                {
                    ordering.Insert(pos, v);
                    double score = ComputeOrderingScore(data, ordering);
                    if (score > bestScore)
                    {
                        bestScore = score;
                        bestPos = pos;
                    }
                    ordering.RemoveAt(pos);
                }

                // Insert at best position
                ordering.Insert(bestPos, v);

                if (bestPos != idx)
                {
                    changed = true;
                }
            }
        }

        // Extract DAG from final ordering
        return ExtractDAG(data, ordering, d, n);
    }

    /// <summary>
    /// Sorts the ordering by marginal variance (ascending — exogenous variables first).
    /// </summary>
    private void SortByMarginalVariance(Matrix<T> data, List<int> ordering, int n)
    {
        // Population variance of every column, then the ordering's columns ascending by variance
        // (equal variances keep their position in the ordering).
        var variances = Engine.ReduceMean(Engine.TensorSquare(CenterColumns(Tensor<T>.FromMatrix(data))), new[] { 0 }, false);
        var keys = Engine.TensorIndexSelect(variances, new Tensor<int>(ordering.ToArray(), new[] { ordering.Count }), 0);
        var indices = CausalTensorHelpers.StableArgsort(keys, descending: false);
        var sorted = indices.Select(i => ordering[i]).ToList();
        ordering.Clear();
        ordering.AddRange(sorted);
    }

    /// <summary>
    /// Computes the total BIC score for a given ordering using greedy parent selection.
    /// </summary>
    private double ComputeOrderingScore(Matrix<T> data, List<int> ordering)
    {
        double totalScore = 0;

        for (int idx = 0; idx < ordering.Count; idx++)
        {
            int target = ordering[idx];
            var bestParents = new HashSet<int>();
            double bestScore = ComputeBIC(data, target, bestParents);

            // Greedily add parents from predecessors
            bool addImproved = true;
            while (addImproved && bestParents.Count < MaxParents)
            {
                addImproved = false;
                int bestCandidate = -1;
                double bestCandidateScore = bestScore;

                for (int predIdx = 0; predIdx < idx; predIdx++)
                {
                    int candidate = ordering[predIdx];
                    if (bestParents.Contains(candidate)) continue;

                    var testParents = new HashSet<int>(bestParents) { candidate };
                    double score = ComputeBIC(data, target, testParents);

                    if (score > bestCandidateScore)
                    {
                        bestCandidateScore = score;
                        bestCandidate = candidate;
                    }
                }

                if (bestCandidate >= 0)
                {
                    bestParents.Add(bestCandidate);
                    bestScore = bestCandidateScore;
                    addImproved = true;
                }
            }

            totalScore += bestScore;
        }

        return totalScore;
    }

    /// <summary>
    /// Extracts the final DAG from the optimal ordering.
    /// </summary>
    private Matrix<T> ExtractDAG(Matrix<T> data, List<int> ordering, int d, int n)
    {
        var parentSets = new HashSet<int>[d];
        for (int i = 0; i < d; i++) parentSets[i] = [];

        for (int idx = 1; idx < d; idx++)
        {
            int target = ordering[idx];
            var bestParents = new HashSet<int>();
            double bestScore = ComputeBIC(data, target, bestParents);

            bool addImproved = true;
            while (addImproved && bestParents.Count < MaxParents)
            {
                addImproved = false;
                int bestCandidate = -1;
                double bestCandidateScore = bestScore;

                for (int predIdx = 0; predIdx < idx; predIdx++)
                {
                    int candidate = ordering[predIdx];
                    if (bestParents.Contains(candidate)) continue;

                    var testParents = new HashSet<int>(bestParents) { candidate };
                    double score = ComputeBIC(data, target, testParents);

                    if (score > bestCandidateScore)
                    {
                        bestCandidateScore = score;
                        bestCandidate = candidate;
                    }
                }

                if (bestCandidate >= 0)
                {
                    bestParents.Add(bestCandidate);
                    bestScore = bestCandidateScore;
                    addImproved = true;
                }
            }

            parentSets[target] = bestParents;
        }

        // Build adjacency matrix with joint multivariate OLS edge weights
        // Centered multivariate OLS of each child on its parents (ridge 1e-10); beta[j] is W[parents[j], child].
        var x = Tensor<T>.FromMatrix(data);
        var positions = new List<int>();
        var coefficients = new List<Tensor<T>>();
        for (int child = 0; child < d; child++)
        {
            var parents = parentSets[child].ToList();
            if (parents.Count == 0) continue;

            var (beta, _) = FitCenteredRegression(x, child, parents.ToArray());
            positions.AddRange(parents.Select(parent => parent * d + child));
            coefficients.Add(beta);
        }

        return CausalTensorHelpers.FromTensorValues(d, d, positions, coefficients);
    }

    // SolveSmallSystem is inherited from CausalDiscoveryBase<T>
}
