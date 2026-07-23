using System.Linq;
using AiDotNet.Enums;

namespace AiDotNet.CausalDiscovery.DeepLearning;

/// <summary>
/// Base class for deep learning-based causal discovery algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Deep learning methods learn causal structure by training neural networks that
/// parameterize the structural equation model. The DAG constraint is typically
/// enforced through continuous relaxation (e.g., NOTEARS-style) during training.
/// </para>
/// <para>
/// <b>For Beginners:</b> These methods use neural networks to discover causal relationships.
/// They can capture complex nonlinear effects but require more data and computation than
/// traditional methods. Think of them as "letting the neural network figure out which
/// variables cause which" by training it on the data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class DeepCausalBase<T> : CausalDiscoveryBase<T>
{
    /// <inheritdoc/>
    public override CausalDiscoveryCategory Category => CausalDiscoveryCategory.DeepLearning;

    /// <summary>
    /// Number of hidden units in neural network layers.
    /// </summary>
    protected int HiddenUnits { get; set; } = 64;

    /// <summary>
    /// Learning rate for gradient-based optimization.
    /// </summary>
    protected double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Maximum training epochs.
    /// </summary>
    protected int MaxEpochs { get; set; } = 100;

    /// <summary>
    /// Edge weight threshold for post-training pruning.
    /// </summary>
    protected double EdgeThreshold { get; set; } = 0.1;

    /// <summary>
    /// Initial log-variance for variational parameters.
    /// </summary>
    protected double InitialLogVariance { get; set; } = -4.0;

    /// <summary>
    /// Default KL divergence weight for variational regularization.
    /// </summary>
    protected double DefaultKlWeight { get; set; } = 0.01;

    /// <summary>
    /// Maximum KL divergence weight after warm-up schedule.
    /// </summary>
    protected double MaxKlWeight { get; set; } = 0.25;

    /// <summary>
    /// Whether to use KL weight warm-up schedule to prevent posterior collapse.
    /// </summary>
    protected bool UseKlWarmUp { get; set; } = true;

    /// <summary>
    /// Maximum penalty parameter (rho_max) for augmented Lagrangian methods.
    /// </summary>
    protected double MaxPenaltyValue { get; set; } = 1e+16;

    /// <summary>
    /// Applies deep learning options.
    /// </summary>
    protected void ApplyDeepOptions(Models.Options.CausalDiscoveryOptions? options)
    {
        if (options == null) return;
        if (options.MaxIterations.HasValue) MaxEpochs = options.MaxIterations.Value;
        if (options.MaxEpochs.HasValue) MaxEpochs = options.MaxEpochs.Value;
        if (options.HiddenUnits.HasValue)
        {
            if (options.HiddenUnits.Value <= 0)
                throw new ArgumentException("HiddenUnits must be greater than 0.");
            HiddenUnits = options.HiddenUnits.Value;
        }
        if (options.EdgeThreshold.HasValue)
        {
            if (double.IsNaN(options.EdgeThreshold.Value) || double.IsInfinity(options.EdgeThreshold.Value) || options.EdgeThreshold.Value < 0)
                throw new ArgumentException("EdgeThreshold must be a non-negative finite value.");
            EdgeThreshold = options.EdgeThreshold.Value;
        }
        if (options.LearningRate.HasValue)
        {
            if (double.IsNaN(options.LearningRate.Value) || double.IsInfinity(options.LearningRate.Value) || options.LearningRate.Value <= 0)
                throw new ArgumentException("LearningRate must be a positive finite value.");
            LearningRate = options.LearningRate.Value;
        }
        if (options.InitialLogVariance.HasValue)
        {
            if (double.IsNaN(options.InitialLogVariance.Value) || double.IsInfinity(options.InitialLogVariance.Value))
                throw new ArgumentException("InitialLogVariance must be a finite value.");
            InitialLogVariance = options.InitialLogVariance.Value;
        }
        if (options.DefaultKlWeight.HasValue)
        {
            if (double.IsNaN(options.DefaultKlWeight.Value) || double.IsInfinity(options.DefaultKlWeight.Value) || options.DefaultKlWeight.Value < 0)
                throw new ArgumentException("DefaultKlWeight must be a non-negative finite value.");
            DefaultKlWeight = options.DefaultKlWeight.Value;
        }
        if (options.MaxKlWeight.HasValue)
        {
            if (double.IsNaN(options.MaxKlWeight.Value) || double.IsInfinity(options.MaxKlWeight.Value) || options.MaxKlWeight.Value < 0)
                throw new ArgumentException("MaxKlWeight must be a non-negative finite value.");
            MaxKlWeight = options.MaxKlWeight.Value;
        }
        if (options.UseKlWarmUp.HasValue) UseKlWarmUp = options.UseKlWarmUp.Value;
        if (options.MaxPenalty.HasValue)
        {
            if (double.IsNaN(options.MaxPenalty.Value) || double.IsInfinity(options.MaxPenalty.Value) || options.MaxPenalty.Value <= 0)
                throw new ArgumentException("MaxPenalty must be a positive finite value.");
            MaxPenaltyValue = options.MaxPenalty.Value;
        }
    }

    /// <summary>
    /// Z-scores each column (zero mean, unit variance) so causal discovery is invariant to per-variable
    /// scaling — multiplying any column by a constant leaves the standardized data (and therefore the
    /// discovered structure) unchanged — and so the optimizer sees a consistent unit-variance signal
    /// instead of one dominated by whichever variable happens to have the largest raw magnitude. Columns
    /// with ~zero variance are centered with a unit divisor to avoid division by zero.
    /// </summary>
    protected Matrix<T> StandardizeColumns(Matrix<T> data)
    {
        int n = data.Rows, d = data.Columns;
        var result = new Matrix<T>(n, d);
        for (int j = 0; j < d; j++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++) mean += NumOps.ToDouble(data[i, j]);
            mean /= Math.Max(1, n);
            double variance = 0;
            for (int i = 0; i < n; i++) { double c = NumOps.ToDouble(data[i, j]) - mean; variance += c * c; }
            variance /= Math.Max(1, n - 1);
            double std = Math.Sqrt(variance);
            double inv = std > 1e-10 ? 1.0 / std : 1.0;
            for (int i = 0; i < n; i++)
                result[i, j] = NumOps.FromDouble((NumOps.ToDouble(data[i, j]) - mean) * inv);
        }
        return result;
    }

    /// <summary>
    /// Projects a learned (possibly cyclic) edge-probability matrix onto a DAG by zeroing every edge that
    /// disagrees with a node ordering. Nodes are ordered by net out-flow (Σ_j P[i,j] − P[j,i]) — the most
    /// "source-like" first — and an edge i→j is kept only when i precedes j in that order, which guarantees
    /// the retained edges admit a topological order (i.e. are acyclic). Retained probabilities are unchanged.
    /// Algorithms that already produce a symmetric probability matrix get acyclicity for free from
    /// <see cref="BuildFinalAdjacency"/>'s direction tie-break and do not need this; it is for the
    /// asymmetric-probability learners (e.g. DAG-GNN) where longer directed cycles can otherwise survive.
    /// </summary>
    protected static double[,] ProjectToDag(double[,] p, int d)
    {
        // Default source score: net out-flow Σ_j P[i,j] − P[j,i] (most edges pointing OUT ⇒ source-like).
        var score = new double[d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (i != j) score[i] += p[i, j] - p[j, i];
        return ProjectToDag(p, d, score);
    }

    /// <summary>
    /// DAG projection with an explicit per-node source score (higher ⇒ earlier in the topological order, i.e.
    /// more cause-like). Lets a learner that cannot identify edge orientation from a symmetric signal supply a
    /// scale-invariant orientation cue — e.g. raw-data variance, which ranks an exogenous root above its
    /// attenuated descendants and is preserved (in ratio) under uniform data scaling.
    /// </summary>
    protected static double[,] ProjectToDag(double[,] p, int d, double[] sourceScore)
    {
        // Stable order: highest score first; ties broken by index for determinism.
        var order = Enumerable.Range(0, d).OrderByDescending(i => sourceScore[i]).ThenBy(i => i).ToArray();
        var position = new int[d];
        for (int r = 0; r < d; r++) position[order[r]] = r;

        var result = new double[d, d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (i != j && position[i] < position[j]) result[i, j] = p[i, j];
        return result;
    }

    /// <summary>
    /// Builds the final weighted adjacency matrix from learned edge probabilities and covariance.
    /// Uses learned P for directionality when training converged, falls back to asymmetric
    /// covariance ratio for directionality otherwise.
    /// </summary>
    /// <param name="learnedP">Learned edge probability matrix [d x d] (values in [0,1]).</param>
    /// <param name="cov">Covariance matrix [d x d].</param>
    /// <param name="d">Number of variables.</param>
    /// <returns>Weighted adjacency matrix.</returns>
    protected Matrix<T> BuildFinalAdjacency(double[,] learnedP, Matrix<T> cov, int d)
    {
        var result = new Matrix<T>(d, d);

        // Threshold scales with d only when uniform attention (= 1/d) crowds
        // a fixed 0.3 floor. For d ≤ 3 keep the historical 0.3 floor —
        // 1/d > 0.3 already, so the discrimination floor is "above uniform"
        // by margin regardless. For d ≥ 4 require uniform + 0.15 margin so
        // an algorithm that just spreads attention near uniform doesn't
        // cross the bar.
        double scaleAwareThreshold = d <= 3 ? 0.3 : Math.Max(1.0 / d + 0.15, 0.3);

        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;

                // Only add edge if learned probability exceeds threshold
                if (learnedP[i, j] < scaleAwareThreshold) continue;

                // Direction: edge i→j only if P[i,j] > P[j,i]
                // For ties, only process the (i,j) pair where i < j to avoid 2-cycles.
                if (learnedP[i, j] < learnedP[j, i]) continue;
                if (Math.Abs(learnedP[i, j] - learnedP[j, i]) < 1e-10 && i > j) continue;

                // Use covariance for edge weight
                double varI = NumOps.ToDouble(cov[i, i]);
                if (varI < 1e-10) continue;
                double covIJ = NumOps.ToDouble(cov[i, j]);
                double weight = covIJ / varI;
                if (Math.Abs(weight) < EdgeThreshold) continue;

                result[i, j] = NumOps.FromDouble(weight);
            }
        return result;
    }

}
