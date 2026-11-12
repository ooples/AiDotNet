using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// AIM (Adaptive and Iterative Mechanism) generator for differentially private synthetic data
/// (McKenna, Mullins, Sheldon, Miklau 2022).
/// </summary>
/// <remarks>
/// <para>
/// The algorithm follows the paper's Algorithm 1 and the reference implementation in the private-pgm
/// (mbi) library:
/// </para>
/// <para>
/// 1. The (epsilon, delta) budget is converted to a zCDP budget rho (Canonne, Kamath and Steinke 2020).
/// 2. Every one-way marginal is measured first with Gaussian noise of scale
/// <c>sigma_0 = sqrt(T / (2 alpha rho))</c>, T = 16 d rounds, alpha = 0.9.
/// 3. Each round selects a marginal from the downward closure of the workload with the exponential
/// mechanism (epsilon_t, starting at <c>sqrt(8 (1 - alpha) rho / T)</c>), using the quality
/// <c>w_r (||M_r(D) - M_r(p_hat)||_1 - sqrt(2/pi) sigma_t n_r)</c> and sensitivity max w_r. Only
/// candidates that keep the junction tree within <c>MaxModelSize * rho_used / rho</c> are eligible.
/// 4. The chosen marginal is measured with Gaussian noise of scale sigma_t, and p_hat is re-estimated by
/// Private-PGM: entropic mirror descent on the log-potentials of a graphical model whose cliques are the
/// measured marginals, minimizing <c>sum_i ||mu_i - y_i||^2 / (2 sigma_i^2)</c> with Armijo
/// backtracking.
/// 5. If the new estimate moved the chosen marginal by no more than <c>sqrt(2/pi) sigma_t n_r</c> in
/// L1, the noise is halved and the selection budget doubled (sigma/2, 2 epsilon).
/// 6. When the remaining budget cannot fund two more rounds, the last round spends all of it.
/// </para>
/// <para>
/// Synthetic rows are drawn from the estimated graphical model by sampling attributes in reverse
/// elimination order, each conditioned on its earlier neighbours in the triangulated model graph.
/// </para>
/// <para>
/// <b>For Beginners:</b> AIM works like a privacy-preserving census: it privately counts how many
/// people fall into combinations of categories, choosing adaptively which combinations to count next,
/// and then fits a probabilistic model that agrees with all those noisy counts.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.SyntheticDataGenerator)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic Data",
    "https://arxiv.org/abs/2201.12677",
    Year = 2022,
    Authors = "Ryan McKenna, Brett Mullins, Daniel Sheldon, Gerome Miklau")]
public class AIMGenerator<T> : SyntheticTabularGeneratorBase<T>
{
    private const double SqrtTwoOverPi = 0.79788456080286535588;
    private const double BytesPerCell = 8.0;
    private const double BytesPerMegabyte = 1048576.0;

    private readonly AIMOptions<T> _options;

    private DomainDiscretizer<T>? _discretizer;
    private int[] _cardinality = Array.Empty<int>();
    private int _numCols;

    // Private-PGM log-potentials, one per measured clique (keyed by the sorted attribute list).
    private readonly Dictionary<string, Factor> _potentials = new(StringComparer.Ordinal);
    private readonly List<Measurement> _measurements = new();
    private T _total = NumOps.One;

    // Sampling plan built from the final model.
    private int[] _sampleOrder = Array.Empty<int>();
    private int[][] _sampleParents = Array.Empty<int[]>();
    private Tensor<T>[] _sampleConditionals = Array.Empty<Tensor<T>>();

    /// <summary>
    /// Initializes a new AIM generator.
    /// </summary>
    /// <param name="options">Configuration options for AIM.</param>
    public AIMGenerator(AIMOptions<T> options)
        : base(options.Seed)
    {
        _options = options;
    }

    /// <summary>zCDP budget rho of the last fit.</summary>
    internal double Rho { get; private set; }

    /// <summary>zCDP budget spent by the last fit; equals <see cref="Rho"/> when the fit completes.</summary>
    internal double RhoUsed { get; private set; }

    /// <summary>Every measurement of the last fit in order: the clique and its noise scale.</summary>
    internal IReadOnlyList<(int[] Clique, double Sigma)> MeasurementLog
        => _measurements.Select(m => (m.Clique, m.Sigma)).ToList();

    /// <inheritdoc />
    protected override void FitInternal(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        ValidateOptions();
        _numCols = columns.Count;
        var discretizer = DomainDiscretizer<T>.Fit(data, columns, _options.NumBins,
            _options.DomainLowerBounds, _options.DomainUpperBounds, out var codes);
        _discretizer = discretizer;
        _cardinality = discretizer.Cardinality;
        _potentials.Clear();
        _measurements.Clear();

        var trueCounts = new Dictionary<string, Tensor<T>>(StringComparer.Ordinal);
        Tensor<T> Truth(int[] clique)
        {
            string key = Key(clique);
            if (!trueCounts.TryGetValue(key, out var counts))
            {
                counts = discretizer.Counts(codes, clique);
                trueCounts[key] = counts;
            }

            return counts;
        }

        Rho = CdpRho(_options.Epsilon, _options.Delta);
        int rounds = _options.Rounds ?? 16 * _numCols;
        double alpha = _options.MeasurementBudgetFraction;
        var (sigma, epsilon) = InitialScales(Rho, rounds, alpha);

        var candidates = CompileWorkload(_cardinality, _options.MaxMarginalOrder, _options.MaxCells);

        // Every one-way marginal is measured up front (Algorithm 1, line 4).
        for (int j = 0; j < _numCols; j++)
        {
            Measure(new[] { j }, Truth(new[] { j }), sigma);
        }

        RhoUsed = _numCols * 0.5 / (sigma * sigma);
        Estimate();

        bool terminate = false;
        while (!terminate)
        {
            double remaining = Rho - RhoUsed;
            if (remaining <= 0) break;

            if (remaining < 2.0 * (0.5 / (sigma * sigma) + epsilon * epsilon / 8.0))
            {
                // Final round: spend everything that is left.
                sigma = Math.Sqrt(1.0 / (2.0 * alpha * remaining));
                epsilon = Math.Sqrt(8.0 * (1.0 - alpha) * remaining);
                terminate = true;
            }

            RhoUsed += epsilon * epsilon / 8.0 + 0.5 / (sigma * sigma);
            double sizeLimit = _options.MaxModelSizeMb * RhoUsed / Rho;

            var modelCliques = _potentials.Values.Select(f => f.Attributes).ToList();
            var eligible = candidates
                .Where(c => ModelSizeMb(_cardinality, modelCliques.Concat(new[] { c.Clique })) <= sizeLimit)
                .ToList();
            if (eligible.Count == 0)
            {
                // One-way cliques are already in the model, so they never grow it.
                eligible = candidates.Where(c => c.Clique.Length == 1).ToList();
            }

            var chosen = WorstApproximated(eligible, Truth, epsilon, sigma);
            var before = ModelMarginal(chosen);
            Measure(chosen, Truth(chosen), sigma);
            Estimate();
            var after = ModelMarginal(chosen);

            if (MovedWithinNoise(before, after, sigma))
            {
                sigma /= 2.0;
                epsilon *= 2.0;
            }
        }

        BuildSamplingPlan();
    }

    /// <inheritdoc />
    protected override Matrix<T> GenerateInternal(int numSamples, Vector<T>? conditionColumn, Vector<T>? conditionValue)
    {
        if (_discretizer is null)
        {
            throw new InvalidOperationException("Generator is not fitted.");
        }

        var engine = AiDotNetEngine.Current;
        var codes = new Tensor<int>[_numCols];
        foreach (int v in _sampleOrder)
        {
            var parentIndex = _discretizer.ConfigIndex(codes, _sampleParents[v], numSamples);
            var rows = engine.TensorIndexSelect(_sampleConditionals[v], parentIndex, 0);
            var oneHot = engine.TensorCategoricalSample(rows, 1, Random.Next());
            codes[v] = engine.TensorArgMax(oneHot, 1).Reshape(new[] { numSamples });
        }

        return _discretizer.Decode(codes, numSamples, Random);
    }

    private void ValidateOptions()
    {
        if (!(_options.Epsilon > 0) || double.IsInfinity(_options.Epsilon))
            throw new ArgumentOutOfRangeException(nameof(_options.Epsilon), "Epsilon must be positive and finite.");
        if (!(_options.Delta > 0) || _options.Delta >= 1)
            throw new ArgumentOutOfRangeException(nameof(_options.Delta), "Delta must be in (0, 1).");
        if (!(_options.MeasurementBudgetFraction > 0) || _options.MeasurementBudgetFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(_options.MeasurementBudgetFraction),
                "MeasurementBudgetFraction must be in (0, 1).");
        if (_options.Rounds is int r && r < 1)
            throw new ArgumentOutOfRangeException(nameof(_options.Rounds), "Rounds must be at least 1.");
    }

    #region Privacy accounting

    /// <summary>
    /// The zCDP budget rho that gives (epsilon, delta)-DP, by bisection on the conversion of Canonne,
    /// Kamath and Steinke (2020), as in private-pgm's <c>cdp_rho</c>.
    /// </summary>
    internal static double CdpRho(double epsilon, double delta)
    {
        if (delta >= 1) return 0.0;

        double lo = 0.0;
        double hi = epsilon + 1.0;
        for (int i = 0; i < 1000; i++)
        {
            double rho = (lo + hi) / 2.0;
            if (CdpDelta(rho, epsilon) <= delta) lo = rho;
            else hi = rho;
        }

        return lo;
    }

    /// <summary>The smallest delta for which rho-zCDP implies (epsilon, delta)-DP.</summary>
    internal static double CdpDelta(double rho, double epsilon)
    {
        if (rho == 0) return 0.0;

        double aMin = 1.01;
        double aMax = (epsilon + 1.0) / (2.0 * rho) + 2.0;
        double a = aMin;
        for (int i = 0; i < 1000; i++)
        {
            a = (aMin + aMax) / 2.0;
            double derivative = (2.0 * a - 1.0) * rho - epsilon + Log1p(-1.0 / a);
            if (derivative < 0) aMin = a;
            else aMax = a;
        }

        double delta = Math.Exp((a - 1.0) * (a * rho - epsilon) + a * Log1p(-1.0 / a)) / (a - 1.0);
        return Math.Min(delta, 1.0);
    }

    private static double Log1p(double x) => Math.Abs(x) < 1e-4 ? x - x * x / 2.0 + x * x * x / 3.0 : Math.Log(1.0 + x);

    /// <summary>Initial noise scale and selection budget: sigma_0 = sqrt(T / (2 alpha rho)), epsilon_0 = sqrt(8 (1 - alpha) rho / T).</summary>
    internal static (double Sigma, double Epsilon) InitialScales(double rho, int rounds, double alpha)
        => (Math.Sqrt(rounds / (2.0 * alpha * rho)), Math.Sqrt(8.0 * (1.0 - alpha) * rho / rounds));

    #endregion

    #region Workload and selection

    /// <summary>
    /// The downward closure of the workload (every r-way marginal, r = <paramref name="order"/>, with at
    /// most <paramref name="maxCells"/> cells) with weights <c>w_r = sum over workload s of |r ∩ s|</c>.
    /// </summary>
    internal static List<(int[] Clique, double Weight)> CompileWorkload(int[] cardinality, int order, int maxCells)
    {
        int d = cardinality.Length;
        int size = Math.Max(1, Math.Min(order, d));
        var workload = Combinations(Enumerable.Range(0, d).ToArray(), size)
            .Where(c => c.Aggregate(1.0, (acc, a) => acc * cardinality[a]) <= maxCells)
            .ToList();

        var closure = new Dictionary<string, int[]>(StringComparer.Ordinal);
        foreach (var w in workload)
        {
            for (int k = 1; k <= w.Length; k++)
            {
                foreach (var sub in Combinations(w, k)) closure[Key(sub)] = sub;
            }
        }

        return closure.Values
            .OrderBy(c => c.Length).ThenBy(Key, StringComparer.Ordinal)
            .Select(c => (c, (double)workload.Sum(w => c.Intersect(w).Count())))
            .ToList();
    }

    private static IEnumerable<int[]> Combinations(int[] pool, int size)
    {
        if (size <= 0 || size > pool.Length) yield break;

        var picks = Enumerable.Range(0, size).ToArray();
        while (true)
        {
            yield return picks.Select(i => pool[i]).ToArray();

            int pos = size - 1;
            while (pos >= 0 && picks[pos] == pool.Length - size + pos) pos--;
            if (pos < 0) yield break;

            picks[pos]++;
            for (int r = pos + 1; r < size; r++) picks[r] = picks[r - 1] + 1;
        }
    }

    /// <summary>
    /// Selects the worst-approximated candidate with the exponential mechanism: probability proportional
    /// to <c>exp(epsilon / (2 Delta) * q_r)</c>, Delta = max w_r.
    /// </summary>
    private int[] WorstApproximated(List<(int[] Clique, double Weight)> candidates, Func<int[], Tensor<T>> truth,
        double epsilon, double sigma)
    {
        var engine = AiDotNetEngine.Current;
        var qualities = candidates
            .Select(c => Quality(truth(c.Clique), ModelMarginal(c.Clique), c.Weight, sigma))
            .ToArray();
        double sensitivity = candidates.Max(c => Math.Abs(c.Weight));
        if (candidates.Count == 1 || !(sensitivity > 0)) return candidates[Random.Next(candidates.Count)].Clique;

        var q = engine.TensorConcatenate(qualities, 0);
        var shifted = engine.TensorSubtractScalar(q, engine.TensorMaxValue(q));
        var weights = engine.TensorExp(engine.TensorMultiplyScalar(shifted, NumOps.FromDouble(0.5 * epsilon / sensitivity)));
        var oneHot = engine.TensorCategoricalSample(weights, 0, Random.Next());
        return candidates[engine.TensorArgMax(oneHot, 0).GetFlat(0)].Clique;
    }

    /// <summary>
    /// AIM's quality score <c>w (||x - x_hat||_1 - sqrt(2/pi) sigma n_r)</c> as a one-element tensor:
    /// the L1 error minus the error the measurement noise alone would cause.
    /// </summary>
    internal static Tensor<T> Quality(Tensor<T> truth, Tensor<T> model, double weight, double sigma)
    {
        var engine = AiDotNetEngine.Current;
        var l1 = engine.ReduceSum(engine.TensorAbs(engine.TensorSubtract(truth, model)).Reshape(new[] { truth.Length }),
            new[] { 0 }, true);
        return engine.TensorMultiplyScalar(
            engine.TensorSubtractScalar(l1, NumOps.FromDouble(SqrtTwoOverPi * sigma * truth.Length)),
            NumOps.FromDouble(weight));
    }

    /// <summary>Annealing test: did re-estimation move the marginal by at most sqrt(2/pi) sigma n_r in L1?</summary>
    internal static bool MovedWithinNoise(Tensor<T> before, Tensor<T> after, double sigma)
    {
        var engine = AiDotNetEngine.Current;
        T moved = engine.TensorSum(engine.TensorAbs(engine.TensorSubtract(after, before)));
        return NumOps.LessThanOrEquals(moved, NumOps.FromDouble(SqrtTwoOverPi * sigma * before.Length));
    }

    /// <summary>
    /// Size in megabytes (8 bytes per cell) of the junction tree of a model over
    /// <paramref name="cliques"/>, from a greedy minimum-weight triangulation.
    /// </summary>
    internal static double ModelSizeMb(int[] cardinality, IEnumerable<int[]> cliques)
        => Triangulate(cardinality, cliques).Cells * BytesPerCell / BytesPerMegabyte;

    /// <summary>
    /// Greedy minimum-weight elimination over the model graph. Returns the elimination order, each
    /// attribute's neighbours at its elimination (a clique of later-eliminated attributes), and the total
    /// cell count of the maximal elimination cliques (the junction tree).
    /// </summary>
    private static (List<int> Order, int[][] Neighbors, double Cells) Triangulate(int[] cardinality, IEnumerable<int[]> cliques)
    {
        int d = cardinality.Length;
        var adjacency = Enumerable.Range(0, d).Select(_ => new HashSet<int>()).ToArray();
        foreach (var clique in cliques)
        {
            foreach (int a in clique)
                foreach (int b in clique)
                    if (a != b) adjacency[a].Add(b);
        }

        var remaining = new HashSet<int>(Enumerable.Range(0, d));
        var order = new List<int>(d);
        var neighbors = new int[d][];
        var eliminationCliques = new List<int[]>(d);
        while (remaining.Count > 0)
        {
            int v = remaining
                .OrderBy(a => adjacency[a].Where(remaining.Contains).Aggregate((double)cardinality[a], (acc, b) => acc * cardinality[b]))
                .ThenBy(a => a)
                .First();
            var nbrs = adjacency[v].Where(remaining.Contains).OrderBy(a => a).ToArray();
            foreach (int a in nbrs)
                foreach (int b in nbrs)
                    if (a != b) adjacency[a].Add(b);

            neighbors[v] = nbrs;
            order.Add(v);
            eliminationCliques.Add(nbrs.Concat(new[] { v }).OrderBy(a => a).ToArray());
            remaining.Remove(v);
        }

        double cells = eliminationCliques
            .Where(c => !eliminationCliques.Any(o => o.Length > c.Length && !c.Except(o).Any()))
            .GroupBy(Key, StringComparer.Ordinal)
            .Sum(g => g.First().Aggregate(1.0, (acc, a) => acc * cardinality[a]));
        return (order, neighbors, cells);
    }

    #endregion

    #region Measurement and Private-PGM estimation

    private sealed class Factor
    {
        public Factor(int[] attributes, Tensor<T> values)
        {
            Attributes = attributes;
            Values = values;
        }

        public int[] Attributes { get; }

        public Tensor<T> Values { get; set; }
    }

    private sealed class Measurement
    {
        public Measurement(int[] clique, Tensor<T> noisy, double sigma, T inverseVariance, Tensor<T> totalWeight)
        {
            Clique = clique;
            Noisy = noisy;
            Sigma = sigma;
            InverseVariance = inverseVariance;
            TotalWeight = totalWeight;
        }

        public int[] Clique { get; }

        public Tensor<T> Noisy { get; }

        public double Sigma { get; }

        public T InverseVariance { get; }

        // 1 / (sigma^2 n_r): the inverse variance of this measurement's estimate of the total.
        public Tensor<T> TotalWeight { get; }
    }

    private static string Key(int[] clique) => string.Join(",", clique);

    /// <summary>Gaussian mechanism: y = M_r(D) + N(0, sigma^2 I), drawn by the engine with one seed.</summary>
    private void Measure(int[] clique, Tensor<T> truth, double sigma)
    {
        var engine = AiDotNetEngine.Current;
        var shape = truth.Shape.ToArray();
        var noise = engine.TensorNormal(new Tensor<T>(shape),
            DomainDiscretizer<T>.Filled(shape, NumOps.FromDouble(sigma)), Random.Next());
        var noisy = engine.TensorAdd(truth, noise);
        _measurements.Add(new Measurement(clique, noisy, sigma, NumOps.FromDouble(1.0 / (sigma * sigma)),
            DomainDiscretizer<T>.Filled(new[] { 1 }, NumOps.FromDouble(1.0 / (sigma * sigma * truth.Length)))));

        string key = Key(clique);
        if (!_potentials.ContainsKey(key))
        {
            _potentials[key] = new Factor(clique, new Tensor<T>(shape));
        }
    }

    /// <summary>
    /// Private-PGM estimate: mirror descent on the log-potentials with Armijo backtracking (private-pgm's
    /// FactoredInference), warm-started from the current potentials.
    /// </summary>
    private void Estimate()
    {
        _total = EstimateTotal();
        double alpha = 1.0 / Lipschitz();
        var mu = Marginals();
        var (loss, grad) = LossAndGradient(mu);

        for (int iteration = 0; iteration < _options.NumIterations; iteration++)
        {
            var omega = _potentials.ToDictionary(kv => kv.Key, kv => kv.Value.Values, StringComparer.Ordinal);
            var nu = mu;
            var currentLoss = loss;
            var direction = grad;
            alpha *= 2.0;

            for (int search = 0; search < 25; search++)
            {
                ApplyStep(omega, direction, alpha);
                mu = Marginals();
                (loss, grad) = LossAndGradient(mu);
                if (SufficientDecrease(currentLoss, loss, alpha, Dot(direction, nu, mu))) break;
                alpha *= 0.5;
            }
        }
    }

    private double Lipschitz()
        => _measurements.GroupBy(m => Key(m.Clique), StringComparer.Ordinal)
            .Max(g => g.Sum(m => 1.0 / (m.Sigma * m.Sigma)));

    /// <summary>
    /// The total row count, estimated as the inverse-variance-weighted mean of the noisy measurement sums
    /// (sum of y_i has variance sigma_i^2 n_i), floored at one.
    /// </summary>
    private T EstimateTotal()
    {
        var engine = AiDotNetEngine.Current;
        var sums = engine.TensorConcatenate(_measurements
            .Select(m => engine.ReduceSum(m.Noisy.Reshape(new[] { m.Noisy.Length }), new[] { 0 }, true)).ToArray(), 0);
        var weights = engine.TensorConcatenate(_measurements.Select(m => m.TotalWeight).ToArray(), 0);
        T total = NumOps.Divide(engine.TensorSum(engine.TensorMultiply(sums, weights)), engine.TensorSum(weights));
        return NumOps.GreaterThan(total, NumOps.One) ? total : NumOps.One;
    }

    /// <summary>theta = omega - alpha * gradient, for every clique potential.</summary>
    private void ApplyStep(Dictionary<string, Tensor<T>> omega, Dictionary<string, Tensor<T>> gradient, double alpha)
    {
        var engine = AiDotNetEngine.Current;
        T step = NumOps.FromDouble(alpha);
        foreach (var kv in _potentials)
        {
            kv.Value.Values = gradient.TryGetValue(kv.Key, out var g)
                ? engine.TensorSubtract(omega[kv.Key], engine.TensorMultiplyScalar(g, step))
                : omega[kv.Key];
        }
    }

    private static bool SufficientDecrease(T currentLoss, T loss, double alpha, T dot)
        => NumOps.GreaterThanOrEquals(NumOps.Subtract(currentLoss, loss),
            NumOps.Multiply(NumOps.FromDouble(0.5 * alpha), dot));

    /// <summary>sum over cliques of &lt;direction_C, nu_C - mu_C&gt;.</summary>
    private T Dot(Dictionary<string, Tensor<T>> direction, Dictionary<string, Tensor<T>> nu, Dictionary<string, Tensor<T>> mu)
    {
        var engine = AiDotNetEngine.Current;
        var terms = direction
            .Select(kv => engine.TensorMultiply(kv.Value, engine.TensorSubtract(nu[kv.Key], mu[kv.Key])).Reshape(new[] { kv.Value.Length }))
            .ToArray();
        return engine.TensorSum(engine.TensorConcatenate(terms, 0));
    }

    /// <summary>Loss sum_i ||mu_i - y_i||^2 / (2 sigma_i^2) and its gradient with respect to each clique marginal.</summary>
    private (T Loss, Dictionary<string, Tensor<T>> Gradient) LossAndGradient(Dictionary<string, Tensor<T>> mu)
    {
        var engine = AiDotNetEngine.Current;
        var gradient = new Dictionary<string, Tensor<T>>(StringComparer.Ordinal);
        var terms = new List<Tensor<T>>(_measurements.Count);
        foreach (var m in _measurements)
        {
            string key = Key(m.Clique);
            var diff = engine.TensorSubtract(mu[key], m.Noisy);
            var scaled = engine.TensorMultiplyScalar(diff, m.InverseVariance);
            terms.Add(engine.TensorMultiply(scaled, diff).Reshape(new[] { diff.Length }));
            gradient[key] = gradient.TryGetValue(key, out var existing) ? engine.TensorAdd(existing, scaled) : scaled;
        }

        return (HalfSum(engine.TensorConcatenate(terms.ToArray(), 0)), gradient);
    }

    private static T HalfSum(Tensor<T> values)
        => NumOps.Multiply(NumOps.FromDouble(0.5), AiDotNetEngine.Current.TensorSum(values));

    /// <summary>Model marginals (as counts scaled to the estimated total) of every potential's clique.</summary>
    private Dictionary<string, Tensor<T>> Marginals()
        => _potentials.ToDictionary(kv => kv.Key, kv => ModelMarginal(kv.Value.Attributes), StringComparer.Ordinal);

    /// <summary>The model's marginal over <paramref name="clique"/>, as counts scaled to the estimated total.</summary>
    internal Tensor<T> ModelMarginal(int[] clique)
    {
        var engine = AiDotNetEngine.Current;
        var log = LogMarginal(clique);
        var normalized = engine.TensorExp(engine.TensorSubtractScalar(log, engine.TensorLogSumExp(log).GetFlat(0)));
        return engine.TensorMultiplyScalar(normalized, _total);
    }

    /// <summary>
    /// Unnormalized log-marginal of the model over sorted <paramref name="target"/> attributes by variable
    /// elimination (sum-product in log space) over the clique potentials.
    /// </summary>
    private Tensor<T> LogMarginal(int[] target)
    {
        var engine = AiDotNetEngine.Current;
        var factors = _potentials.Values.Select(f => (f.Attributes, f.Values)).ToList();
        var keep = new HashSet<int>(target);
        var eliminate = factors.SelectMany(f => f.Attributes).Distinct().Where(a => !keep.Contains(a)).ToList();

        while (eliminate.Count > 0)
        {
            int v = eliminate
                .OrderBy(a => factors.Where(f => f.Attributes.Contains(a)).SelectMany(f => f.Attributes).Distinct()
                    .Aggregate(1.0, (acc, b) => acc * _cardinality[b]))
                .ThenBy(a => a)
                .First();
            eliminate.Remove(v);

            var group = factors.Where(f => f.Attributes.Contains(v)).ToList();
            factors = factors.Where(f => !f.Attributes.Contains(v)).ToList();
            var union = group.SelectMany(f => f.Attributes).Distinct().OrderBy(a => a).ToArray();
            var product = Product(group, union);
            var reduced = engine.TensorLogSumExp(product, Array.IndexOf(union, v), false);
            var rest = union.Where(a => a != v).ToArray();
            factors.Add((rest, reduced.Reshape(rest.Length == 0 ? new[] { 1 } : rest.Select(a => _cardinality[a]).ToArray())));
        }

        return Product(factors, target);
    }

    /// <summary>Sum of log-factors, each broadcast to the sorted <paramref name="target"/> attributes.</summary>
    private Tensor<T> Product(List<(int[] Attributes, Tensor<T> Values)> factors, int[] target)
    {
        var engine = AiDotNetEngine.Current;
        var shape = target.Select(a => _cardinality[a]).ToArray();
        var sum = new Tensor<T>(shape);
        foreach (var (attributes, values) in factors)
        {
            var aligned = target.Select(a => attributes.Contains(a) ? _cardinality[a] : 1).ToArray();
            sum = engine.TensorAdd(sum, engine.TensorBroadcastTo(values.Reshape(aligned), shape).Contiguous());
        }

        return sum;
    }

    /// <summary>
    /// Conditional tables for ancestral sampling: attributes are drawn in reverse elimination order, each
    /// conditioned on its neighbours at elimination, which are exactly the attributes it depends on
    /// among those already drawn (a clique of the triangulated graph).
    /// </summary>
    private void BuildSamplingPlan()
    {
        var engine = AiDotNetEngine.Current;
        var (order, neighbors, _) = Triangulate(_cardinality, _potentials.Values.Select(f => f.Attributes));
        _sampleOrder = Enumerable.Reverse(order).ToArray();
        _sampleParents = neighbors;
        _sampleConditionals = new Tensor<T>[_numCols];
        foreach (int v in _sampleOrder)
        {
            var attributes = neighbors[v].Concat(new[] { v }).OrderBy(a => a).ToArray();
            var log = LogMarginal(attributes);
            var childLast = engine.TensorMoveDim(log, Array.IndexOf(attributes, v), attributes.Length - 1).Contiguous();
            int parentConfigs = neighbors[v].Aggregate(1, (acc, a) => acc * _cardinality[a]);
            _sampleConditionals[v] = engine.Softmax(childLast.Reshape(new[] { parentConfigs, _cardinality[v] }), 1);
        }
    }

    #endregion
}
