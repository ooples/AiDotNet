using AiDotNet.Helpers;
using AiDotNet.Models.Options;

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

            // Convert counts to probabilities with Laplace smoothing
            foreach (var kvp in counts)
            {
                double total = 0;
                for (int b = 0; b < numBins; b++) total += kvp.Value[b] + smooth;
                var probs = new double[numBins];
                for (int b = 0; b < numBins; b++)
                    probs[b] = (kvp.Value[b] + smooth) / total;
                cpts[j][kvp.Key] = probs;
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
