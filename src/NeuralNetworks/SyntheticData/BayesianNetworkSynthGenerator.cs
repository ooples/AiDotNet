using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// PrivBayes synthesizer (Zhang et al., SIGMOD 2014 / TODS 2017): learns a k-degree Bayesian network
/// under differential privacy, releases noisy conditional distributions, and generates synthetic data
/// by ancestral sampling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// With <see cref="BayesianNetworkSynthOptions{T}.EnableDifferentialPrivacy"/> on (the default) the two
/// phases follow the paper:
/// </para>
/// <para>
/// 1. <b>GreedyBayes</b> (Algorithm 1 / 4): the first attribute is chosen uniformly at random. Each later
/// attribute X is attached together with a parent set of exactly <c>min(k, |V|)</c> already-attached
/// attributes, chosen by the exponential mechanism over the mutual information I(X; Pi). The sensitivity
/// is Lemma 1's binary form when X or Pi has a binary domain and the general form otherwise.
/// </para>
/// <para>
/// 2. <b>NoisyConditionals</b> (Algorithm 2): for i = k+1..d the full joint distribution of
/// (X_i, Pi_i), zero cells included, receives Laplace noise of scale <c>2(d-k)/(n eps2)</c>; negatives are
/// clipped and the table renormalized. The conditionals of the first k attributes are derived from the
/// noisy joint of (X_{k+1}, Pi_{k+1}), so exactly d-k marginals spend the phase-2 budget.
/// </para>
/// <para>
/// The degree k is the largest value up to <see cref="BayesianNetworkSynthOptions{T}.MaxParents"/> that
/// keeps the network theta-useful (Section 4.4): <c>n eps2 / (2 (d-k) |dom(X, Pi)|) &gt;= theta</c>.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as building a probabilistic "family tree" of your features, then
/// adding carefully calibrated random noise so that no single person's record can be recovered from the
/// released model.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// PrivBayes was never on arXiv; it appeared at SIGMOD 2014 with an extended TODS 2017 version, so the
// canonical DOI is used.
[ResearchPaper("PrivBayes: Private Data Release via Bayesian Networks",
    "https://doi.org/10.1145/3134428",
    Year = 2017,
    Authors = "Jun Zhang, Graham Cormode, Cecilia M. Procopiuc, Divesh Srivastava, Xiaokui Xiao")]
public class BayesianNetworkSynthGenerator<T> : SyntheticTabularGeneratorBase<T>
{
    private const double LogTwo = 0.69314718055994530942;

    private readonly BayesianNetworkSynthOptions<T> _options;

    // parents[j] = parent attributes of j, in the axis order of its conditional table.
    private List<int>[] _parents = Array.Empty<List<int>>();

    // Attributes in sampling order (parents before children).
    private int[] _topoOrder = Array.Empty<int>();

    // Domain size of every attribute after discretization.
    private int[] _cardinality = Array.Empty<int>();

    // Column domains (equal-width bins / category values) and the encoder/decoder for them.
    private DomainDiscretizer<T>? _discretizer;

    // Conditional distribution of every attribute: [|dom(Pi)|, |dom(X)|], rows sum to one.
    private Tensor<T>[] _conditionals = Array.Empty<Tensor<T>>();

    private int _numFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="BayesianNetworkSynthGenerator{T}"/> class.
    /// </summary>
    /// <param name="options">Configuration options for the Bayesian Network Synthesis model.</param>
    public BayesianNetworkSynthGenerator(BayesianNetworkSynthOptions<T> options) : base(options.Seed)
    {
        _options = options;
    }

    /// <summary>The network degree k used by the last fit.</summary>
    internal int Degree { get; private set; }

    /// <summary>How many joint distributions received Laplace noise in the last private fit (d - k).</summary>
    internal int NoisedMarginalCount { get; private set; }

    /// <summary>Parent sets learned by the last fit, in conditional-table axis order.</summary>
    internal IReadOnlyList<IReadOnlyList<int>> Parents => _parents;

    /// <summary>Conditional tables learned by the last fit.</summary>
    internal IReadOnlyList<Tensor<T>> Conditionals => _conditionals;

    /// <inheritdoc />
    protected override void FitInternal(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _numFeatures = data.Columns;
        int n = data.Rows;

        var discretizer = DomainDiscretizer<T>.Fit(data, columns, _options.NumBins,
            _options.DomainLowerBounds, _options.DomainUpperBounds, out var codes);
        _discretizer = discretizer;
        _cardinality = discretizer.Cardinality;

        if (_options.EnableDifferentialPrivacy)
        {
            double eps2 = _options.PrivacyBudget * (1.0 - _options.StructureBudgetFraction);
            Degree = SelectDegree(n, _numFeatures, _cardinality.Max(), eps2,
                _options.UsefulnessThreshold, _options.MaxParents);
            _topoOrder = LearnStructureGreedyBayes(codes, Degree);
            _conditionals = NoisyConditionals(codes, Degree, eps2);
        }
        else
        {
            Degree = Math.Min(Math.Max(0, _options.MaxParents), _numFeatures - 1);
            NoisedMarginalCount = 0;
            _parents = LearnStructureBIC(codes);
            _topoOrder = TopologicalSort();
            _conditionals = SmoothedConditionals(codes);
        }
    }

    /// <inheritdoc />
    protected override Matrix<T> GenerateInternal(int numSamples, Vector<T>? conditionColumn, Vector<T>? conditionValue)
    {
        var engine = AiDotNetEngine.Current;
        var codes = new Tensor<int>[_numFeatures];

        // Ancestral sampling, one attribute at a time for the whole batch: look up each row's
        // conditional distribution from its sampled parents, then draw one category per row.
        foreach (int j in _topoOrder)
        {
            var parentIndex = ParentConfigIndex(codes, _parents[j], numSamples);
            var rows = engine.TensorIndexSelect(_conditionals[j], parentIndex, 0);
            var oneHot = engine.TensorCategoricalSample(rows, 1, Random.Next());
            codes[j] = engine.TensorArgMax(oneHot, 1).Reshape(new[] { numSamples });
        }

        if (_discretizer is null)
        {
            throw new InvalidOperationException("Generator is not fitted.");
        }

        return _discretizer.Decode(codes, numSamples, Random);
    }

    /// <summary>
    /// Picks the network degree k: the largest k up to <paramref name="maxParents"/> (and d - 1) for which
    /// <c>n eps2 / (2 (d - k) m^(k+1)) &gt;= theta</c>, with m the largest attribute domain. This is the
    /// paper's theta-usefulness rule; with binary attributes it reduces to Definition 4.5,
    /// <c>n eps / ((d - k) 2^(k+3)) &gt;= theta</c>. A non-positive theta disables the rule.
    /// </summary>
    internal static int SelectDegree(int n, int d, int maxCardinality, double eps2, double theta, int maxParents)
    {
        int cap = Math.Min(Math.Max(0, maxParents), Math.Max(0, d - 1));
        if (theta <= 0) return cap;

        for (int k = cap; k >= 1; k--)
        {
            double cells = Math.Pow(Math.Max(1, maxCardinality), k + 1);
            if (n * eps2 / (2.0 * (d - k) * cells) >= theta) return k;
        }

        return 0;
    }

    /// <summary>
    /// PrivBayes phase 1 (GreedyBayes): attaches attributes one at a time, each with a parent set of
    /// exactly min(k, |V|) attached attributes chosen by the exponential mechanism. Returns the
    /// attachment order, which is a topological order of the network.
    /// </summary>
    private int[] LearnStructureGreedyBayes(Tensor<int>[] codes, int k)
    {
        int d = _numFeatures;
        int n = codes.Length == 0 ? 0 : codes[0].Length;

        _parents = new List<int>[d];
        for (int j = 0; j < d; j++) _parents[j] = new List<int>();

        // The first node is chosen uniformly at random; a data-independent choice spends no budget.
        var attached = new List<int> { Random.Next(d) };
        var remaining = Enumerable.Range(0, d).Where(j => j != attached[0]).ToList();
        if (d <= 1) return attached.ToArray();

        double stepBudget = _options.PrivacyBudget * _options.StructureBudgetFraction / (d - 1);

        while (remaining.Count > 0)
        {
            int parentCount = Math.Min(k, attached.Count);
            var candidates = new List<(int Child, List<int> Parents)>();
            var scores = new List<Tensor<T>>();
            double sensitivity = 0.0;

            foreach (int child in remaining)
            {
                foreach (var parentSet in EnumerateParentSets(attached, parentCount))
                {
                    candidates.Add((child, parentSet));
                    scores.Add(MutualInformation(codes, child, parentSet, n));

                    int parentDomain = parentSet.Aggregate(1, (acc, p) => acc * _cardinality[p]);
                    bool binary = _cardinality[child] == 2 || parentDomain == 2;
                    sensitivity = Math.Max(sensitivity, MutualInformationSensitivity(n, binary));
                }
            }

            int chosen = SampleExponentialMechanism(scores, stepBudget, sensitivity);
            _parents[candidates[chosen].Child] = candidates[chosen].Parents;
            attached.Add(candidates[chosen].Child);
            remaining.Remove(candidates[chosen].Child);
        }

        return attached.ToArray();
    }

    /// <summary>
    /// Every subset of <paramref name="pool"/> with exactly <paramref name="size"/> members, in
    /// lexicographic order of pool position (members keep the pool's order).
    /// </summary>
    internal static IEnumerable<List<int>> EnumerateParentSets(IReadOnlyList<int> pool, int size)
    {
        if (size <= 0)
        {
            yield return new List<int>();
            yield break;
        }

        if (size > pool.Count) yield break;

        var picks = Enumerable.Range(0, size).ToArray();
        while (true)
        {
            yield return picks.Select(i => pool[i]).ToList();

            int pos = size - 1;
            while (pos >= 0 && picks[pos] == pool.Count - size + pos) pos--;
            if (pos < 0) yield break;

            picks[pos]++;
            for (int r = pos + 1; r < size; r++) picks[r] = picks[r - 1] + 1;
        }
    }

    /// <summary>
    /// Sensitivity of mutual information over n rows (PrivBayes Lemma 1). Binary form, when X or Pi has
    /// a binary domain: <c>(1/n) log2 n + ((n-1)/n) log2(n/(n-1))</c>. General form:
    /// <c>(2/n) log2((n+1)/2) + ((n-1)/n) log2((n+1)/(n-1))</c>.
    /// </summary>
    internal static double MutualInformationSensitivity(int n, bool binary)
    {
        if (n <= 1) return 1.0;

        double nm1 = n - 1.0;
        return binary
            ? (1.0 / n) * Math.Log(n, 2) + (nm1 / n) * Math.Log(n / nm1, 2)
            : (2.0 / n) * Math.Log((n + 1.0) / 2.0, 2) + (nm1 / n) * Math.Log((n + 1.0) / nm1, 2);
    }

    /// <summary>
    /// Mutual information I(X; Pi) in bits from the empirical joint distribution, as a one-element
    /// tensor: H(X) + H(Pi) - H(X, Pi).
    /// </summary>
    private Tensor<T> MutualInformation(Tensor<int>[] codes, int child, IReadOnlyList<int> parents, int n)
        => MutualInformation(JointCounts(codes, child, parents), n);

    /// <summary>
    /// Mutual information in bits of a [|dom(Pi)|, |dom(X)|] count table over n rows, as a one-element
    /// tensor.
    /// </summary>
    internal static Tensor<T> MutualInformation(Tensor<T> counts, int n)
    {
        var engine = AiDotNetEngine.Current;
        var joint = engine.TensorDivideScalar(counts, NumOps.FromDouble(Math.Max(1, n)));
        var mi = engine.TensorSubtract(
            engine.TensorAdd(Entropy(engine.ReduceSum(joint, new[] { 0 }, false)),
                Entropy(engine.ReduceSum(joint, new[] { 1 }, false))),
            Entropy(joint));
        return engine.TensorClampMin(engine.TensorDivideScalar(mi, NumOps.FromDouble(LogTwo)), NumOps.Zero);
    }

    /// <summary>Shannon entropy in nats of a distribution tensor, as a one-element tensor (0 log 0 = 0).</summary>
    private static Tensor<T> Entropy(Tensor<T> p)
    {
        var engine = AiDotNetEngine.Current;
        var flat = p.Reshape(new[] { p.Length });
        var safe = engine.TensorWhere(engine.TensorGreaterThan(flat, NumOps.Zero), flat, Filled(new[] { p.Length }, NumOps.One));
        return engine.TensorNegate(engine.ReduceSum(engine.TensorMultiply(flat, engine.TensorLog(safe)), new[] { 0 }, true));
    }

    /// <summary>
    /// Counts of every (parent configuration, child value) cell, zero cells included, shaped
    /// [|dom(Pi)|, |dom(X)|]. Parent configurations are mixed-radix in parent-list order.
    /// </summary>
    private Tensor<T> JointCounts(Tensor<int>[] codes, int child, IReadOnlyList<int> parents)
        => JointCounts(codes[child], _cardinality[child], ParentConfigIndex(codes, parents, codes[child].Length),
            parents.Aggregate(1, (acc, p) => acc * _cardinality[p]));

    /// <summary>Counts of every (parent configuration, child value) cell as a [parentDomain, childDomain] table.</summary>
    internal static Tensor<T> JointCounts(Tensor<int> childCodes, int childDomain, Tensor<int> parentIndex, int parentDomain)
    {
        var engine = AiDotNetEngine.Current;
        var cell = engine.TensorAdd(engine.TensorMultiplyScalar(parentIndex, childDomain), childCodes);
        return engine.TensorBinCount(cell, parentDomain * childDomain).Cast<T>()
            .Reshape(new[] { parentDomain, childDomain });
    }

    /// <summary>Mixed-radix index of each row's parent configuration (zero when there are no parents).</summary>
    private Tensor<int> ParentConfigIndex(Tensor<int>[] codes, IReadOnlyList<int> parents, int rows)
    {
        var engine = AiDotNetEngine.Current;
        var index = new Tensor<int>(new[] { rows });
        foreach (int p in parents)
        {
            index = engine.TensorAdd(engine.TensorMultiplyScalar(index, _cardinality[p]), codes[p]);
        }

        return index;
    }

    /// <summary>
    /// Exponential mechanism: draws candidate c with probability proportional to
    /// <c>exp(epsilon * score(c) / (2 * sensitivity))</c>. Scores are shifted by their maximum first so
    /// the exponent cannot overflow. Returns the chosen index.
    /// </summary>
    private int SampleExponentialMechanism(List<Tensor<T>> scores, double epsilon, double sensitivity)
    {
        if (scores.Count == 1 || epsilon <= 0 || sensitivity <= 0)
        {
            // A non-positive budget cannot fund a data-dependent choice, so draw uniformly.
            return Random.Next(scores.Count);
        }

        var engine = AiDotNetEngine.Current;
        var all = engine.TensorConcatenate(scores.ToArray(), 0);
        var shifted = engine.TensorSubtractScalar(all, engine.TensorMaxValue(all));
        var weights = engine.TensorExp(engine.TensorMultiplyScalar(shifted, NumOps.FromDouble(epsilon / (2.0 * sensitivity))));
        var oneHot = engine.TensorCategoricalSample(weights, 0, Random.Next());
        return engine.TensorArgMax(oneHot, 0).GetFlat(0);
    }

    /// <summary>
    /// PrivBayes phase 2 (NoisyConditionals). The joints of attributes k+1..d get Laplace noise of scale
    /// 2(d-k)/(n eps2); the first k conditionals are marginalized out of the (k+1)-th noisy joint, whose
    /// parent set is exactly the first k attributes.
    /// </summary>
    private Tensor<T>[] NoisyConditionals(Tensor<int>[] codes, int k, double eps2)
    {
        var engine = AiDotNetEngine.Current;
        int d = _numFeatures;
        int n = codes[0].Length;
        NoisedMarginalCount = d - k;
        double scale = n > 0 && eps2 > 0 ? 2.0 * NoisedMarginalCount / (n * eps2) : 0.0;

        var conditionals = new Tensor<T>[d];
        Tensor<T>? anchor = null;
        for (int i = k; i < d; i++)
        {
            int x = _topoOrder[i];
            var noisy = NoisyJoint(JointCounts(codes, x, _parents[x]), n, scale);
            conditionals[x] = Conditional(noisy);
            if (i == k) anchor = noisy;
        }

        if (anchor is not null && k > 0)
        {
            // Axis order of the anchor: the first k attached attributes, then X_{k+1}.
            var full = anchor.Reshape(_topoOrder.Take(k + 1).Select(a => _cardinality[a]).ToArray());
            for (int i = 0; i < k; i++)
            {
                var axes = Enumerable.Range(i + 1, k - i).ToArray();
                var marginal = engine.ReduceSum(full, axes, false);
                int childDomain = _cardinality[_topoOrder[i]];
                conditionals[_topoOrder[i]] = Conditional(
                    marginal.Reshape(new[] { marginal.Length / childDomain, childDomain }));
            }
        }

        return conditionals;
    }

    /// <summary>
    /// counts / n plus Laplace(scale) noise in every cell, negatives clipped, renormalized; uniform when
    /// every cell clipped away (data-independent, so it costs no budget).
    /// </summary>
    private Tensor<T> NoisyJoint(Tensor<T> counts, int n, double scale)
    {
        var engine = AiDotNetEngine.Current;
        var fractions = engine.TensorDivideScalar(counts, NumOps.FromDouble(Math.Max(1, n)));
        var noisy = engine.TensorClampMin(engine.TensorAdd(fractions, LaplaceNoise(counts.Shape.ToArray(), scale)), NumOps.Zero);
        T total = engine.TensorSum(noisy);
        if (NumOps.GreaterThan(total, NumOps.Zero))
        {
            return engine.TensorDivideScalar(noisy, total);
        }

        return Filled(counts.Shape.ToArray(), NumOps.FromDouble(1.0 / counts.Length));
    }

    /// <summary>
    /// Zero-mean Laplace noise of scale b by inverse transform: <c>-b sign(u) ln(1 - 2|u|)</c>,
    /// u ~ U[-0.5, 0.5), drawn by the engine with one seed from the generator's RNG.
    /// </summary>
    private Tensor<T> LaplaceNoise(int[] shape, double scale)
    {
        var engine = AiDotNetEngine.Current;
        if (scale <= 0) return new Tensor<T>(shape);

        var u = engine.TensorRandomUniformRange(shape, NumOps.FromDouble(-0.5), NumOps.FromDouble(0.5), Random.Next());
        var magnitude = engine.TensorClampMin(
            engine.ScalarMinusTensor(NumOps.One, engine.TensorMultiplyScalar(engine.TensorAbs(u), NumOps.FromDouble(2.0))),
            NumOps.FromDouble(1e-30));
        return engine.TensorMultiplyScalar(engine.TensorMultiply(engine.TensorSign(u), engine.TensorLog(magnitude)),
            NumOps.FromDouble(-scale));
    }

    /// <summary>Normalizes each row of a [q, r] joint to a conditional; empty rows become uniform.</summary>
    private static Tensor<T> Conditional(Tensor<T> joint)
    {
        var engine = AiDotNetEngine.Current;
        int r = joint.Shape[joint.Rank - 1];
        int q = joint.Length / r;
        var shape = new[] { q, r };
        var table = joint.Reshape(shape);
        var rowTotals = engine.TensorBroadcastTo(engine.ReduceSum(table, new[] { 1 }, true), shape).Contiguous();
        var hasMass = engine.TensorGreaterThan(rowTotals, NumOps.Zero);
        var safe = engine.TensorWhere(hasMass, rowTotals, Filled(shape, NumOps.One));
        return engine.TensorWhere(hasMass, engine.TensorDivide(table, safe), Filled(shape, NumOps.FromDouble(1.0 / r)));
    }

    /// <summary>
    /// Non-private BIC structure search, used only when
    /// <see cref="BayesianNetworkSynthOptions{T}.EnableDifferentialPrivacy"/> is false. Each pass adds,
    /// per child, the acyclic parent with the largest positive BIC gain.
    /// </summary>
    private List<int>[] LearnStructureBIC(Tensor<int>[] codes)
    {
        var engine = AiDotNetEngine.Current;
        var parents = new List<int>[_numFeatures];
        for (int j = 0; j < _numFeatures; j++) parents[j] = new List<int>();

        int n = codes[0].Length;
        T halfLogN = NumOps.FromDouble(0.5 * Math.Log(Math.Max(1, n)));
        int maxParents = Math.Max(0, _options.MaxParents);

        for (int iter = 0; iter < _options.MaxIterations; iter++)
        {
            bool improved = false;

            for (int child = 0; child < _numFeatures; child++)
            {
                if (parents[child].Count >= maxParents) continue;

                var options = Enumerable.Range(0, _numFeatures)
                    .Where(p => p != child && !parents[child].Contains(p) && !WouldCreateCycle(parents, p, child))
                    .ToList();
                if (options.Count == 0) continue;

                var current = LocalBIC(codes, child, parents[child], halfLogN);
                var gains = engine.TensorSubtract(
                    engine.TensorConcatenate(options
                        .Select(p => LocalBIC(codes, child, new List<int>(parents[child]) { p }, halfLogN))
                        .ToArray(), 0),
                    engine.TensorBroadcastTo(current, new[] { options.Count }).Contiguous());

                int best = engine.TensorArgMax(gains, 0).GetFlat(0);
                if (IsPositive(gains.GetFlat(best)))
                {
                    parents[child].Add(options[best]);
                    improved = true;
                }
            }

            if (!improved) break;
        }

        return parents;
    }

    private static bool IsPositive(T value) => NumOps.GreaterThan(value, NumOps.Zero);

    /// <summary>
    /// Checks if adding an edge from parent to child would create a cycle in the DAG.
    /// </summary>
    private static bool WouldCreateCycle(List<int>[] parents, int newParent, int child)
    {
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
    /// Local BIC score as a one-element tensor: log-likelihood minus (q (r - 1) / 2) log n, where q is
    /// the number of parent configurations and r the child's domain size (Schwarz 1978).
    /// </summary>
    private Tensor<T> LocalBIC(Tensor<int>[] codes, int child, IReadOnlyList<int> parents, T halfLogN)
    {
        var engine = AiDotNetEngine.Current;
        var counts = JointCounts(codes, child, parents);
        int q = counts.Shape[0];
        int r = counts.Shape[1];

        // sum c log(c / R) = sum c log c - sum R log R, with 0 log 0 = 0.
        var logLikelihood = engine.TensorSubtract(
            engine.TensorNegate(Entropy(counts)),
            engine.TensorNegate(Entropy(engine.ReduceSum(counts, new[] { 1 }, false))));
        return engine.TensorSubtractScalar(logLikelihood,
            NumOps.Multiply(NumOps.FromDouble(q * (r - 1.0)), halfLogN));
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
    /// Non-private conditionals with Laplace (additive) smoothing: (count + alpha) / (row + r alpha).
    /// </summary>
    private Tensor<T>[] SmoothedConditionals(Tensor<int>[] codes)
    {
        var engine = AiDotNetEngine.Current;
        T alpha = NumOps.FromDouble(Math.Max(0.0, _options.LaplaceSmoothing));
        var conditionals = new Tensor<T>[_numFeatures];
        for (int j = 0; j < _numFeatures; j++)
        {
            conditionals[j] = Conditional(engine.TensorAddScalar(JointCounts(codes, j, _parents[j]), alpha));
        }

        return conditionals;
    }

    private static Tensor<T> Filled(int[] shape, T value) => DomainDiscretizer<T>.Filled(shape, value);
}
