using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// Bayesian Network Synthesis generator that learns a DAG structure over features,
/// estimates conditional probability tables (CPTs), and generates synthetic data
/// via ancestral sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This is a classical statistical approach (no neural networks):
/// 1. Discretize continuous features into bins
/// 2. Learn a DAG structure using greedy hill-climbing with BIC scoring
/// 3. Estimate CPTs using maximum likelihood with Laplace smoothing
/// 4. Generate data by sampling from root nodes to leaf nodes in topological order
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as building a probabilistic "family tree" of your features:
///
/// Step 1: Figure out which features depend on which others (the DAG)
/// Step 2: For each feature, learn "if parent features have values X, this feature is Y with probability Z"
/// Step 3: To generate a new row, start with features that have no parents and work downward
///
/// Advantages: Fast, interpretable, no GPU needed.
/// Disadvantage: Less flexible than deep learning for complex distributions.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation URL corrected. arXiv 1401.0939 is "The NINJA-2 project: Detecting and characterizing
// gravitational waveforms modelled using numerical binary black hole simulations" — an unrelated
// gravitational-wave paper. PrivBayes was never on arXiv; it appeared at SIGMOD 2014 with an extended
// TODS 2017 version, so the canonical DOI is used instead of an invented preprint id.
//
// This class previously implemented only a GENERIC Bayesian-network synthesizer (BIC-scored structure
// search, Laplace-SMOOTHED CPTs) with no epsilon and no privacy guarantee at all — Laplace smoothing
// being a prior, not a privacy mechanism. Both of PrivBayes' defining phases are now implemented:
//
//   Phase 1 (GreedyBayes): each (attribute, parent-set) pair is chosen with the EXPONENTIAL MECHANISM
//   over a mutual-information score, calibrated by that score's sensitivity, instead of by arg-max.
//   Parents are drawn only from already-attached attributes, so acyclicity holds by construction.
//
//   Phase 2: Laplace NOISE is injected into each marginal (sensitivity 2/n per marginal, shared across
//   d-k marginals) before conditionals are derived, then negatives are clipped and the distribution
//   renormalized.
//
// The budget splits evenly between the phases per the paper, and epsilon, the split and an opt-out all
// live in BayesianNetworkSynthOptions with the paper's defaults.
[ResearchPaper("PrivBayes: Private Data Release via Bayesian Networks",
    "https://doi.org/10.1145/3134428",
    Year = 2017,
    Authors = "Jun Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, Xiaokui Xiao")]
public class BayesianNetworkSynthGenerator<T> : SyntheticTabularGeneratorBase<T>
{
    private readonly BayesianNetworkSynthOptions<T> _options;

    // DAG structure: parents[j] = list of parent indices for feature j
    private List<int>[] _parents = Array.Empty<List<int>>();

    // Topological ordering of features
    private int[] _topoOrder = Array.Empty<int>();

    // Discretization: bin edges for each feature
    private double[][] _binEdges = Array.Empty<double[]>();

    // Conditional probability tables: _cpts[j] maps (parent_values_key) -> probability distribution over bins
    private Dictionary<string, double[]>[] _cpts = Array.Empty<Dictionary<string, double[]>>();

    // Number of features (original columns)
    private int _numFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="BayesianNetworkSynthGenerator{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for the Bayesian Network Synthesis model.</param>
    public BayesianNetworkSynthGenerator(BayesianNetworkSynthOptions<T> options) : base(options.Seed)
    {
        _options = options;
    }

    /// <inheritdoc />
    protected override void FitInternal(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _numFeatures = data.Columns;

        // Step 1: Discretize all features
        var discretized = DiscretizeData(data);

        // Step 2: Learn DAG structure using greedy hill-climbing
        _parents = LearnStructure(discretized);

        // Step 3: Compute topological ordering
        _topoOrder = TopologicalSort();

        // Step 4: Estimate CPTs
        _cpts = EstimateCPTs(discretized);
    }

    /// <inheritdoc />
    protected override Matrix<T> GenerateInternal(int numSamples, Vector<T>? conditionColumn, Vector<T>? conditionValue)
    {
        var result = new Matrix<T>(numSamples, _numFeatures);

        for (int i = 0; i < numSamples; i++)
        {
            var sample = AncestralSample();
            for (int j = 0; j < _numFeatures; j++)
                result[i, j] = NumOps.FromDouble(sample[j]);
        }

        return result;
    }

    /// <summary>
    /// Discretizes continuous data into bins using equal-width binning.
    /// </summary>
    private int[][] DiscretizeData(Matrix<T> data)
    {
        int numBins = _options.NumBins;
        _binEdges = new double[_numFeatures][];
        var discretized = new int[data.Rows][];

        for (int j = 0; j < _numFeatures; j++)
        {
            double min = double.MaxValue;
            double max = double.MinValue;
            for (int i = 0; i < data.Rows; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                if (val < min) min = val;
                if (val > max) max = val;
            }

            // Create bin edges
            _binEdges[j] = new double[numBins + 1];
            double step = (max - min) / numBins;
            if (step < 1e-10) step = 1.0;
            for (int b = 0; b <= numBins; b++)
                _binEdges[j][b] = min + b * step;
            _binEdges[j][numBins] = max + 1e-10; // Ensure max is included

            // Discretize column
            for (int i = 0; i < data.Rows; i++)
            {
                if (discretized[i] is null) discretized[i] = new int[_numFeatures];
                double val = NumOps.ToDouble(data[i, j]);
                int bin = (int)((val - min) / step);
                if (bin >= numBins) bin = numBins - 1;
                if (bin < 0) bin = 0;
                discretized[i][j] = bin;
            }
        }

        return discretized;
    }

    /// <summary>
    /// Learns the DAG structure using greedy hill-climbing with BIC scoring.
    /// Tries adding edges that improve BIC, respecting MaxParents and acyclicity.
    /// </summary>
    private List<int>[] LearnStructure(int[][] discretized)
    {
        if (_options.EnableDifferentialPrivacy)
        {
            return LearnStructureGreedyBayes(discretized);
        }

        return LearnStructureBIC(discretized);
    }

    /// <summary>
    /// PrivBayes phase 1 (GreedyBayes): builds the k-degree Bayesian network by sampling each
    /// (attribute, parent-set) pair with the EXPONENTIAL MECHANISM instead of taking the arg-max.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Taking the highest-scoring parent set — as plain BIC structure learning does — leaks
    /// information: the winning edge is a deterministic function of the data, so one individual's
    /// record can change which edge appears and the released network reveals that. The exponential
    /// mechanism instead samples candidate <c>c</c> with probability proportional to
    /// <c>exp(eps' * score(c) / (2 * sensitivity))</c>, which makes high-scoring candidates likely
    /// without making any of them certain, and that is what buys the privacy guarantee.
    /// </para>
    /// <para>
    /// The score is mutual information <c>I(X; Pi)</c> between an attribute and a candidate parent
    /// set — the paper's measure of how much a parent set explains an attribute. Its sensitivity over
    /// a dataset of n rows is <c>(1/n)*log2(n) + ((n-1)/n)*log2(n/(n-1))</c>, which is what calibrates
    /// the mechanism.
    /// </para>
    /// <para>
    /// The structure budget is divided evenly across the d-1 selection steps, so the whole phase
    /// consumes exactly <see cref="BayesianNetworkSynthOptions{T}.StructureBudgetFraction"/> of the
    /// total epsilon; phase 2 spends the remainder.
    /// </para>
    /// </remarks>
    private List<int>[] LearnStructureGreedyBayes(int[][] discretized)
    {
        int d = _numFeatures;
        int n = discretized.Length;
        int k = Math.Max(1, _options.MaxParents);

        var parents = new List<int>[d];
        for (int j = 0; j < d; j++) parents[j] = new List<int>();

        if (d <= 1) return parents;

        double structureBudget = _options.PrivacyBudget * _options.StructureBudgetFraction;

        // Budget per selection step: d-1 attributes are attached after the first seed node.
        double stepBudget = structureBudget / (d - 1);
        double sensitivity = MutualInformationSensitivity(n);

        // The first node is chosen uniformly at random and has no parents; a uniform choice consumes
        // no budget because it does not depend on the data at all.
        var attached = new List<int> { Random.Next(d) };
        var remaining = new List<int>();
        for (int j = 0; j < d; j++)
        {
            if (j != attached[0]) remaining.Add(j);
        }

        while (remaining.Count > 0)
        {
            // Candidate set: every unattached attribute paired with every parent subset (size <= k)
            // drawn from the already-attached attributes. Restricting parents to attached nodes is
            // what keeps the result acyclic by construction, so no cycle check is needed.
            var candidates = new List<(int Child, List<int> Parents, double Score)>();
            foreach (int child in remaining)
            {
                foreach (var parentSet in EnumerateParentSets(attached, k))
                {
                    double mi = ComputeMutualInformation(discretized, child, parentSet);
                    candidates.Add((child, parentSet, mi));
                }
            }

            if (candidates.Count == 0) break;

            var chosen = SampleExponentialMechanism(candidates, stepBudget, sensitivity);
            parents[chosen.Child] = chosen.Parents;
            attached.Add(chosen.Child);
            remaining.Remove(chosen.Child);
        }

        return parents;
    }

    /// <summary>
    /// Enumerates every subset of <paramref name="pool"/> with size from 0 to
    /// <paramref name="maxSize"/>, which is the candidate parent-set space for one attribute.
    /// </summary>
    private static IEnumerable<List<int>> EnumerateParentSets(List<int> pool, int maxSize)
    {
        yield return new List<int>();

        int limit = Math.Min(maxSize, pool.Count);
        var current = new List<int>();

        IEnumerable<List<int>> Build(int start, int depth)
        {
            for (int i = start; i < pool.Count; i++)
            {
                current.Add(pool[i]);
                yield return new List<int>(current);
                if (depth + 1 < limit)
                {
                    foreach (var deeper in Build(i + 1, depth + 1)) yield return deeper;
                }

                current.RemoveAt(current.Count - 1);
            }
        }

        if (limit > 0)
        {
            foreach (var set in Build(0, 0)) yield return set;
        }
    }

    /// <summary>
    /// Sensitivity of mutual information for a dataset of <paramref name="n"/> rows:
    /// <c>(1/n)*log2(n) + ((n-1)/n)*log2(n/(n-1))</c>.
    /// </summary>
    /// <remarks>
    /// This is how much the score can change when one individual's record is added or removed, and it
    /// is what the exponential mechanism must be calibrated against. It shrinks as n grows, so larger
    /// datasets need proportionally less distortion for the same guarantee.
    /// </remarks>
    private static double MutualInformationSensitivity(int n)
    {
        if (n <= 1) return 1.0;

        double log2N = Math.Log(n, 2);
        return (1.0 / n) * log2N + ((n - 1.0) / n) * Math.Log(n / (n - 1.0), 2);
    }

    /// <summary>
    /// Mutual information <c>I(X; Pi)</c> in bits between attribute <paramref name="child"/> and the
    /// joint of <paramref name="parentList"/>, estimated from the empirical distribution.
    /// </summary>
    private double ComputeMutualInformation(int[][] data, int child, List<int> parentList)
    {
        int n = data.Length;
        if (n == 0) return 0.0;

        // An empty parent set explains nothing, so the mutual information is zero by definition.
        if (parentList.Count == 0) return 0.0;

        int numBins = _options.NumBins;
        var joint = new Dictionary<string, int[]>();
        var childCounts = new int[numBins];
        var parentTotals = new Dictionary<string, int>();

        for (int i = 0; i < n; i++)
        {
            int bin = data[i][child];
            if (bin < 0 || bin >= numBins) continue;

            string key = GetParentKey(data[i], parentList);
            if (!joint.TryGetValue(key, out var row))
            {
                row = new int[numBins];
                joint[key] = row;
            }

            row[bin]++;
            childCounts[bin]++;
            parentTotals[key] = parentTotals.TryGetValue(key, out int t) ? t + 1 : 1;
        }

        double mi = 0.0;
        foreach (var kvp in joint)
        {
            double pParent = parentTotals[kvp.Key] / (double)n;
            for (int b = 0; b < numBins; b++)
            {
                int c = kvp.Value[b];
                if (c == 0) continue;

                double pJoint = c / (double)n;
                double pChild = childCounts[b] / (double)n;
                if (pChild <= 0 || pParent <= 0) continue;

                mi += pJoint * Math.Log(pJoint / (pParent * pChild), 2);
            }
        }

        return mi < 0 ? 0 : mi;
    }

    /// <summary>
    /// Samples one candidate with probability proportional to
    /// <c>exp(epsilon * score / (2 * sensitivity))</c>.
    /// </summary>
    /// <remarks>
    /// Weights are computed relative to the maximum score before exponentiating. Exponentiating the
    /// raw scores directly overflows to infinity as soon as the exponent is large, which would collapse
    /// the distribution onto a single candidate and silently destroy the privacy guarantee this
    /// mechanism exists to provide.
    /// </remarks>
    private (int Child, List<int> Parents, double Score) SampleExponentialMechanism(
        List<(int Child, List<int> Parents, double Score)> candidates,
        double epsilon,
        double sensitivity)
    {
        if (candidates.Count == 1 || epsilon <= 0 || sensitivity <= 0)
        {
            // A non-positive budget cannot fund a data-dependent choice, so fall back to a uniform
            // draw rather than leaking the arg-max.
            return candidates[Random.Next(candidates.Count)];
        }

        double scale = epsilon / (2.0 * sensitivity);

        double maxScore = double.NegativeInfinity;
        foreach (var c in candidates)
        {
            if (c.Score > maxScore) maxScore = c.Score;
        }

        var weights = new double[candidates.Count];
        double total = 0.0;
        for (int i = 0; i < candidates.Count; i++)
        {
            weights[i] = Math.Exp(scale * (candidates[i].Score - maxScore));
            total += weights[i];
        }

        if (total <= 0 || double.IsNaN(total) || double.IsInfinity(total))
        {
            return candidates[Random.Next(candidates.Count)];
        }

        double u = Random.NextDouble() * total;
        double cum = 0.0;
        for (int i = 0; i < candidates.Count; i++)
        {
            cum += weights[i];
            if (u <= cum) return candidates[i];
        }

        return candidates[candidates.Count - 1];
    }

    /// <summary>
    /// Non-private BIC structure search, used only when
    /// <see cref="BayesianNetworkSynthOptions{T}.EnableDifferentialPrivacy"/> is false.
    /// </summary>
    private List<int>[] LearnStructureBIC(int[][] discretized)
    {
        var parents = new List<int>[_numFeatures];
        for (int j = 0; j < _numFeatures; j++)
            parents[j] = new List<int>();

        int numBins = _options.NumBins;
        int n = discretized.Length;
        double logN = Math.Log(n);

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            bool improved = false;

            for (int child = 0; child < _numFeatures; child++)
            {
                if (parents[child].Count >= _options.MaxParents) continue;

                double bestGain = 0;
                int bestParent = -1;

                for (int parent = 0; parent < _numFeatures; parent++)
                {
                    if (parent == child) continue;
                    if (parents[child].Contains(parent)) continue;

                    // Check if adding this edge would create a cycle
                    if (WouldCreateCycle(parents, parent, child)) continue;

                    // Compute BIC gain of adding parent→child edge
                    double bicWithout = ComputeLocalBIC(discretized, child, parents[child], numBins, n, logN);
                    var withParent = new List<int>(parents[child]) { parent };
                    double bicWith = ComputeLocalBIC(discretized, child, withParent, numBins, n, logN);
                    double gain = bicWith - bicWithout;

                    if (gain > bestGain)
                    {
                        bestGain = gain;
                        bestParent = parent;
                    }
                }

                if (bestParent >= 0)
                {
                    parents[child].Add(bestParent);
                    improved = true;
                }
            }

            if (!improved) break;
        }

        return parents;
    }

    /// <summary>
    /// Checks if adding an edge from parent to child would create a cycle in the DAG.
    /// </summary>
    private bool WouldCreateCycle(List<int>[] parents, int newParent, int child)
    {
        // DFS from child: if we can reach newParent, adding the edge would create a cycle
        var visited = new HashSet<int>();
        var stack = new Stack<int>();
        stack.Push(newParent);

        while (stack.Count > 0)
        {
            int node = stack.Pop();
            if (node == child) return true;
            if (!visited.Add(node)) continue;
            foreach (int p in parents[node])
                stack.Push(p);
        }

        return false;
    }

    /// <summary>
    /// Computes the BIC score for a node given its parent set.
    /// BIC = log-likelihood - (k/2) * log(n), where k is number of parameters.
    /// </summary>
    private static double ComputeLocalBIC(int[][] data, int child, List<int> parentList, int numBins, int n, double logN)
    {
        // Count joint frequencies
        var counts = new Dictionary<string, int[]>();
        for (int i = 0; i < n; i++)
        {
            string key = GetParentKey(data[i], parentList);
            if (!counts.ContainsKey(key))
                counts[key] = new int[numBins];
            int bin = data[i][child];
            if (bin >= 0 && bin < numBins)
                counts[key][bin]++;
        }

        // Compute log-likelihood
        double ll = 0;
        foreach (var kvp in counts)
        {
            int total = 0;
            for (int b = 0; b < numBins; b++) total += kvp.Value[b];
            if (total == 0) continue;
            for (int b = 0; b < numBins; b++)
            {
                if (kvp.Value[b] > 0)
                    ll += kvp.Value[b] * Math.Log((double)kvp.Value[b] / total);
            }
        }

        // Penalty: number of parameters = numParentConfigs * (numBins - 1)
        int numConfigs = Math.Max(counts.Count, 1);
        int k = numConfigs * (numBins - 1);
        return ll - 0.5 * k * logN;
    }

    /// <summary>
    /// Creates a string key from the parent values of a data row.
    /// </summary>
    private static string GetParentKey(int[] row, List<int> parentList)
    {
        if (parentList.Count == 0) return "";
        var parts = new int[parentList.Count];
        for (int p = 0; p < parentList.Count; p++)
            parts[p] = row[parentList[p]];
        return string.Join(",", parts);
    }

    /// <summary>
    /// Computes a topological ordering of the DAG nodes.
    /// </summary>
    private int[] TopologicalSort()
    {
        var order = new List<int>();
        var visited = new HashSet<int>();

        void Visit(int node)
        {
            if (!visited.Add(node)) return;
            foreach (int p in _parents[node])
                Visit(p);
            order.Add(node);
        }

        for (int j = 0; j < _numFeatures; j++)
            Visit(j);

        return order.ToArray();
    }

    /// <summary>
    /// Estimates conditional probability tables (CPTs) for each feature given its parents.
    /// Uses Laplace smoothing to prevent zero probabilities.
    /// </summary>
    private Dictionary<string, double[]>[] EstimateCPTs(int[][] discretized)
    {
        int numBins = _options.NumBins;
        double smooth = _options.LaplaceSmoothing;
        var cpts = new Dictionary<string, double[]>[_numFeatures];

        for (int j = 0; j < _numFeatures; j++)
        {
            cpts[j] = new Dictionary<string, double[]>();
            var counts = new Dictionary<string, int[]>();

            for (int i = 0; i < discretized.Length; i++)
            {
                string key = GetParentKey(discretized[i], _parents[j]);
                if (!counts.ContainsKey(key))
                    counts[key] = new int[numBins];
                int bin = discretized[i][j];
                if (bin >= 0 && bin < numBins)
                    counts[key][bin]++;
            }

            // PrivBayes phase 2: inject Laplace NOISE into the marginal before deriving conditionals.
            //
            // This is the step that makes the released distributions differentially private, and it is
            // distinct from LaplaceSmoothing above (a prior that merely avoids zero probabilities and
            // provides no guarantee). The noise is added to the marginal expressed as FRACTIONS of n:
            // one individual can move a joint cell by 1/n in each of two cells, so the per-marginal
            // sensitivity is 2/n. The (d - k) noisy marginals share the remaining budget, giving a
            // scale of 2*(d-k)/(n*eps2).
            //
            // Noise can push cells negative, which is not a distribution, so the standard
            // post-processing applies: clamp negatives to zero and renormalize. Post-processing a
            // differentially private release cannot weaken its guarantee.
            if (_options.EnableDifferentialPrivacy)
            {
                double eps2 = _options.PrivacyBudget * (1.0 - _options.StructureBudgetFraction);
                int marginalCount = Math.Max(1, _numFeatures - Math.Max(1, _options.MaxParents));
                int n = discretized.Length;
                double noiseScale = n > 0 && eps2 > 0
                    ? 2.0 * marginalCount / (n * eps2)
                    : 0.0;

                foreach (var kvp in counts)
                {
                    var noisy = new double[numBins];
                    for (int b = 0; b < numBins; b++)
                    {
                        double fraction = n > 0 ? kvp.Value[b] / (double)n : 0.0;
                        double perturbed = fraction + SampleLaplace(noiseScale);
                        noisy[b] = perturbed > 0 ? perturbed : 0.0;
                    }

                    double total = 0;
                    for (int b = 0; b < numBins; b++) total += noisy[b];

                    var probs = new double[numBins];
                    if (total > 0)
                    {
                        for (int b = 0; b < numBins; b++) probs[b] = noisy[b] / total;
                    }
                    else
                    {
                        // Every cell was clipped away, which happens when the true marginal is tiny
                        // relative to the noise. Uniform is the maximum-entropy fallback and, being
                        // data-independent, costs no additional budget.
                        double p = 1.0 / numBins;
                        for (int b = 0; b < numBins; b++) probs[b] = p;
                    }

                    cpts[j][kvp.Key] = probs;
                }
            }
            else
            {
                // Convert counts to probabilities with Laplace smoothing (no privacy guarantee).
                foreach (var kvp in counts)
                {
                    double total = 0;
                    for (int b = 0; b < numBins; b++) total += kvp.Value[b] + smooth;
                    var probs = new double[numBins];
                    for (int b = 0; b < numBins; b++)
                        probs[b] = (kvp.Value[b] + smooth) / total;
                    cpts[j][kvp.Key] = probs;
                }
            }

            // Default CPT for unseen parent configurations
            if (!cpts[j].ContainsKey("__default__"))
            {
                var defaultProbs = new double[numBins];
                double p = 1.0 / numBins;
                for (int b = 0; b < numBins; b++)
                    defaultProbs[b] = p;
                cpts[j]["__default__"] = defaultProbs;
            }
        }

        return cpts;
    }

    /// <summary>
    /// Generates a single sample using ancestral sampling (topological order).
    /// </summary>
    private double[] AncestralSample()
    {
        int numBins = _options.NumBins;
        var binValues = new int[_numFeatures];
        var result = new double[_numFeatures];

        // Sample in topological order (parents before children)
        foreach (int j in _topoOrder)
        {
            // Build parent key from already-sampled parent values
            string key = "";
            if (_parents[j].Count > 0)
            {
                var parts = new int[_parents[j].Count];
                for (int p = 0; p < _parents[j].Count; p++)
                    parts[p] = binValues[_parents[j][p]];
                key = string.Join(",", parts);
            }

            // Get CPT for this parent configuration
            double[] probs;
            if (_cpts[j].TryGetValue(key, out var found))
                probs = found;
            else
                probs = _cpts[j]["__default__"];

            // Sample from the distribution
            int sampledBin = SampleFromDistribution(probs);
            binValues[j] = sampledBin;

            // Convert bin back to continuous value (sample uniformly within the bin)
            double lo = _binEdges[j][sampledBin];
            double hi = sampledBin + 1 < _binEdges[j].Length ? _binEdges[j][sampledBin + 1] : lo + 1.0;
            result[j] = lo + Random.NextDouble() * (hi - lo);
        }

        return result;
    }

    /// <summary>
    /// Draws a sample from the zero-mean Laplace distribution with the given scale <c>b</c>.
    /// </summary>
    /// <remarks>
    /// Uses inverse transform sampling on <c>u ~ Uniform(-0.5, 0.5)</c>:
    /// <c>-b * sign(u) * ln(1 - 2|u|)</c>. Drawing from the seeded <see cref="Random"/> keeps
    /// generation reproducible for a given seed, which matters because a synthesizer whose output
    /// cannot be reproduced cannot be tested.
    /// </remarks>
    private double SampleLaplace(double scale)
    {
        if (scale <= 0) return 0.0;

        double u = Random.NextDouble() - 0.5;

        // Guard the logarithm: |u| = 0.5 would give ln(0).
        double magnitude = 1.0 - 2.0 * Math.Abs(u);
        if (magnitude <= 0) magnitude = double.Epsilon;

        return -scale * Math.Sign(u) * Math.Log(magnitude);
    }

    /// <summary>
    /// Samples a bin index from a discrete probability distribution.
    /// </summary>
    private int SampleFromDistribution(double[] probs)
    {
        double u = Random.NextDouble();
        double cumSum = 0;
        for (int b = 0; b < probs.Length; b++)
        {
            cumSum += probs[b];
            if (u <= cumSum) return b;
        }

        return probs.Length - 1;
    }
}
