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
    /// Projects an already weighted directed graph onto a DAG by retaining the strongest edges
    /// whose insertion does not close a directed cycle.
    /// </summary>
    /// <remarks>
    /// Pairwise direction learners can orient every pair consistently in isolation while the
    /// combined orientations still form a longer cycle. Sorting by absolute learned strength and
    /// rejecting only cycle-closing edges preserves the strongest evidence and supplies the DAG
    /// guarantee advertised by the common causal-graph contract.
    /// </remarks>
    protected Matrix<T> ProjectWeightedGraphToDag(Matrix<T> weights)
    {
        // Candidates are the finite, non-zero off-diagonal weights; they are inserted strongest
        // first (ties in (from, to) order), skipping any edge that would close a directed cycle.
        var w = Tensor<T>.FromMatrix(weights);
        var finite = Engine.TensorWhere(Engine.TensorIsFinite(w), w, Engine.TensorZerosLike(w));
        return ProjectToAcyclic(finite.ToMatrix());
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
        if (d <= 0)
            throw new ArgumentOutOfRangeException(nameof(d), d, "The variable count must be positive.");

        // Require a fixed margin above uniform influence for every graph size.
        // This matters most for d = 2 and d = 3, where 1/d already exceeds the
        // historical 0.3 floor and an uninformative uniform predictor would pass it.
        double scaleAwareThreshold = Math.Max(1.0 / d + 0.15, 0.3);
        return BuildFinalAdjacency(learnedP, cov, d, scaleAwareThreshold);
    }

    /// <summary>
    /// Converts learned directed influence strengths into a weighted adjacency matrix
    /// using an algorithm-defined absolute influence threshold.
    /// </summary>
    /// <param name="learnedP">Learned directed influence strengths [d x d].</param>
    /// <param name="cov">Covariance matrix [d x d].</param>
    /// <param name="d">Number of variables.</param>
    /// <param name="learnedThreshold">Minimum learned influence required for an edge.</param>
    /// <returns>Weighted adjacency matrix.</returns>
    protected Matrix<T> BuildFinalAdjacency(double[,] learnedP, Matrix<T> cov, int d, double learnedThreshold)
    {
        // The learned influences arrive as double; the direction tests run on them unchanged.
        var flatP = new double[d * d];
        Buffer.BlockCopy(learnedP, 0, flatP, 0, d * d * sizeof(double));
        var p = new Tensor<double>(flatP, new[] { d, d });
        var pT = Engine.TensorTranspose(p);
        var lowerTriangle = Engine.TensorTranspose(Engine.TensorTriu(Engine.TensorOnesLike(p), 1)); // i > j

        // Only add edge if learned influence reaches the threshold.
        // Direction: edge i→j only if P[i,j] >= P[j,i]; for ties (within 1e-10) only the (i,j)
        // pair with i < j is kept, to avoid 2-cycles.
        var notBelowThreshold = Engine.ScalarMinusTensor(1.0, Engine.TensorLessThan(p, learnedThreshold));
        var notWeaker = Engine.ScalarMinusTensor(1.0, Engine.TensorLessThan(p, pT));
        var tie = Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorSubtract(p, pT)), 1e-10);
        var notLowerTie = Engine.ScalarMinusTensor(1.0, Engine.TensorMultiply(tie, lowerTriangle));
        var directed = Engine.TensorMultiply(Engine.TensorMultiply(notBelowThreshold, notWeaker), notLowerTie).Cast<T>();

        // Use covariance for edge weight cov[i,j] / cov[i,i], skipped when cov[i,i] < 1e-10 or
        // |weight| < EdgeThreshold.
        var c = Tensor<T>.FromMatrix(cov);
        var variance = Engine.TensorBroadcastTo(Engine.TensorDiagonal(c).Reshape(new[] { d, 1 }), new[] { d, d });
        var smallVariance = Engine.TensorLessThan(variance, NumOps.FromDouble(1e-10));
        var weight = Engine.TensorDivide(c, Engine.TensorWhere(smallVariance, Engine.TensorOnesLike(c), variance));
        var keep = Engine.TensorMultiply(Engine.TensorMultiply(directed, OffDiagonalMask(d)),
            Engine.TensorMultiply(Engine.ScalarMinusTensor(NumOps.One, smallVariance),
                Engine.ScalarMinusTensor(NumOps.One,
                    Engine.TensorLessThan(Engine.TensorAbs(weight), NumOps.FromDouble(EdgeThreshold)))));
        return Engine.TensorWhere(keep, weight, Engine.TensorZerosLike(weight)).ToMatrix();
    }

    /// <summary>
    /// Copies a [d x d] tensor into the double grid that the DAG projection and adjacency helpers take.
    /// </summary>
    private protected static double[,] ToDoubleGrid(Tensor<T> values, int d)
    {
        var flat = values.Cast<double>().ToArray();
        var grid = new double[d, d];
        Buffer.BlockCopy(flat, 0, grid, 0, d * d * sizeof(double));
        return grid;
    }
}
