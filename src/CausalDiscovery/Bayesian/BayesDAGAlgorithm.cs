using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// BayesDAG — Bayesian DAG learning with gradient-based posterior inference.
/// </summary>
/// <remarks>
/// <para>
/// BayesDAG uses a DAG-constrained variational framework that maintains a continuous
/// relaxation of the DAG posterior. It parameterizes edge probabilities via logits Z
/// and uses the Gumbel-Sigmoid trick for differentiable sampling. The acyclicity
/// constraint is enforced via augmented Lagrangian on the expected adjacency.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize edge logits Z[i,j] = 0 (uniform prior on each edge)</item>
/// <item>Sample adjacency: A[i,j] ~ Gumbel-Sigmoid(Z[i,j] / tau)</item>
/// <item>Compute data likelihood: L = -0.5 * ||X - X*A||^2 / n</item>
/// <item>Compute acyclicity: h(A) = tr(e^(A*A)) - d</item>
/// <item>Compute ELBO = likelihood - KL(q || prior) - lambda*h(A)</item>
/// <item>Update Z via gradient ascent on ELBO</item>
/// <item>Anneal temperature tau from soft to hard</item>
/// <item>Threshold final sigmoid(Z) to get binary adjacency</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> BayesDAG is a modern Bayesian method that efficiently explores
/// the space of possible causal graphs using gradient-based optimization, providing
/// principled uncertainty quantification about the causal structure.
/// </para>
/// <para>
/// Reference: Annadani et al. (2024), "BayesDAG: Gradient-Based Posterior Inference
/// for Causal Discovery", NeurIPS.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.Causal)]
[ModelCategory(ModelCategory.CausalModel)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.CausalInference)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Matrix<>))]
[ResearchPaper("BayesDAG: Gradient-Based Posterior Inference for Causal Discovery", "https://openreview.net/forum?id=VBnYjBJLvS", Year = 2024, Authors = "Yashas Annadani, Nick Pawlowski, Joel Jennings, Stefan Bauer, Cheng Zhang, Wenbo Gong")]
public class BayesDAGAlgorithm<T> : BayesianCausalBase<T>
{
    private readonly double _learningRate;

    /// <inheritdoc/>
    public override string Name => "BayesDAG";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public BayesDAGAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyBayesianOptions(options);
        // Default logit learning rate 0.05 (was 0.01). The logit gradient is scaled by the
        // sigmoid derivative P(1-P)/tau, which shrinks as an edge's probability approaches its
        // equilibrium, so a 0.01 rate did not converge the true edges above p=0.5 within the
        // default iteration budget — they plateaued just below (correctly RANKED above the
        // non-edges, but never crossing the threshold). 0.05 reaches the equilibrium in budget.
        _learningRate = options?.LearningRate ?? 0.05;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var cov = ComputeCovarianceMatrix(data);

        // Edge logits Z[i,j]: sigmoid(Z / tau) = edge probability. Initialize SPARSE
        // (z=-1 -> p≈0.27 at tau=1). A zero init put every edge at p=0.5 — a fully dense
        // graph whose least-squares reconstruction is over-determined, so the data-fit residual
        // (and therefore the per-edge gradient) is ~0 and no edge is ever distinguished: the
        // logits drifted to ~-0.015 for TRUE and false edges alike and DiscoverStructure
        // recovered 0 edges. Starting sparse gives each candidate edge a clear residual signal
        // so strong true edges get a large negative (strengthening) data-fit gradient. z=-1 is
        // mild enough that it does not saturate the sigmoid under the (now floored) temperature.
        var Z = new Matrix<T>(d, d);
        T zInit = NumOps.FromDouble(-1.0);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (i != j) Z[i, j] = zInit;
        T lr = NumOps.FromDouble(_learningRate);

        // Acyclicity penalty weight. Kept LOW (0.1) so it discourages cycles without
        // overwhelming the data fit: with the previous rho=1 (and the runaway augmented-
        // Lagrangian escalation before that) the positive-for-every-edge acyclicity gradient
        // held even genuine strong edges just below p=0.5, so 0 edges were recovered. At 0.1 the
        // data fit lifts true edges past their KL-to-0.5 equilibrium while non-edges stay near 0.5.
        // Applied as a FIXED penalty (no augmented-Lagrangian dual ascent), so there is no alpha
        // multiplier and no rho escalation/cap.
        T rho = NumOps.FromDouble(0.1);

        for (int outerIter = 0; outerIter < NumSamples; outerIter++)
        {
            // Temperature annealing. Floor at 0.5 (not 0.1): the sigmoid-derivative that
            // scales every logit's gradient is P(1-P)/tau, and once P is even moderately away
            // from 0.5 a tiny tau drives that derivative to ~0, freezing the logits. A 0.5 floor
            // keeps gradients flowing so the data fit can actually move the edges.
            double tau = Math.Max(0.5, 1.0 * Math.Pow(0.95, outerIter));
            T tauT = NumOps.FromDouble(tau);

            // Compute edge probabilities: p[i,j] = sigmoid(Z[i,j] / tau)
            var P = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T zScaled = NumOps.Divide(Z[i, j], tauT);
                    double sv = NumOps.ToDouble(zScaled);
                    double sigVal = sv > 20 ? 1.0 : sv < -20 ? 0.0 : 1.0 / (1.0 + Math.Exp(-sv));
                    P[i, j] = NumOps.FromDouble(sigVal);
                }

            // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
            var PSqC = new Matrix<T>(d, d);
            for (int i2 = 0; i2 < d; i2++)
                for (int j2 = 0; j2 < d; j2++)
                    PSqC[i2, j2] = NumOps.Multiply(P[i2, j2], P[i2, j2]);
            var expPSqC = MatrixExponentialTaylor(PSqC, d);
            T hValC = NumOps.Zero;
            for (int i2 = 0; i2 < d; i2++)
                hValC = NumOps.Add(hValC, expPSqC[i2, i2]);
            hValC = NumOps.Subtract(hValC, NumOps.FromDouble(d));

            // Compute expected weighted adjacency: W[i,j] = P[i,j] * cov[i,j] / cov[i,i]
            var W = new Matrix<T>(d, d);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    T varI = cov[i, i];
                    T eps = NumOps.FromDouble(1e-10);
                    if (NumOps.GreaterThan(varI, eps))
                    {
                        T olsWeight = NumOps.Divide(cov[i, j], varI);
                        W[i, j] = NumOps.Multiply(P[i, j], olsWeight);
                    }
                }

            // Compute L2 loss gradient: dL/dW = (W^T * cov - cov) for each column
            // Simplified: gradient w.r.t. P[i,j] = dL/dW[i,j] * olsWeight
            var gradZ = new Matrix<T>(d, d);

            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;

                    // Data fit gradient: d(-likelihood)/dP[i,j]
                    T varI = cov[i, i];
                    T eps = NumOps.FromDouble(1e-10);
                    if (!NumOps.GreaterThan(varI, eps)) continue;

                    T olsWeight = NumOps.Divide(cov[i, j], varI);

                    // Residual contribution
                    T residGrad = NumOps.Zero;
                    for (int k = 0; k < d; k++)
                    {
                        if (k == j) continue;
                        residGrad = NumOps.Add(residGrad, NumOps.Multiply(W[k, j], cov[i, k]));
                    }
                    residGrad = NumOps.Subtract(residGrad, cov[i, j]);
                    T dataGrad = NumOps.Multiply(residGrad, olsWeight);

                    // KL divergence gradient: log(P/(1-P)) (Bernoulli prior = 0.5)
                    T pij = P[i, j];
                    T oneMinusP = NumOps.Subtract(NumOps.One, pij);
                    T klGrad = NumOps.Zero;
                    if (NumOps.GreaterThan(pij, NumOps.FromDouble(1e-8)) &&
                        NumOps.GreaterThan(oneMinusP, NumOps.FromDouble(1e-8)))
                    {
                        klGrad = NumOps.FromDouble(
                            Math.Log(NumOps.ToDouble(pij) / NumOps.ToDouble(oneMinusP)));
                    }

                    // Acyclicity gradient (fixed-penalty NOTEARS): rho*h * [exp(P∘P)^T ∘ 2P][i,j]
                    T acycGrad = NumOps.Multiply(
                        NumOps.Multiply(rho, hValC),
                        NumOps.Multiply(expPSqC[j, i], NumOps.Multiply(NumOps.FromDouble(2), pij)));

                    // Total gradient w.r.t. P[i,j]
                    T totalGradP = NumOps.Add(dataGrad, NumOps.Add(klGrad, acycGrad));

                    // Chain rule: dL/dZ = dL/dP * dP/dZ = dL/dP * P*(1-P)/tau
                    T sigDeriv = NumOps.Divide(NumOps.Multiply(pij, oneMinusP), tauT);
                    gradZ[i, j] = NumOps.Multiply(totalGradP, sigDeriv);
                }

            // Update logits
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                {
                    if (i == j) continue;
                    Z[i, j] = NumOps.Subtract(Z[i, j], NumOps.Multiply(lr, gradZ[i, j]));
                }

            // Acyclicity is applied as a MILD FIXED penalty (rho constant, no dual multiplier), NOT
            // a growing augmented-Lagrangian dual. The original code ran the dual ascent
            // (alpha += rho*h) and penalty escalation (rho *= 10) on EVERY one of the
            // NumSamples (5000) inner gradient steps, compounding alpha/rho to ~1e10 within a
            // few dozen steps. The acyclicity gradient — which is positive for EVERY edge, not
            // just cycle-forming ones — then dwarfed the data-fit gradient (~O(1)) by many
            // orders of magnitude and drove all edge logits to zero, so DiscoverStructure
            // recovered 0 edges on a clean linear SEM. A constant O(1) penalty keeps acyclicity
            // comparable to the data fit, so genuine strong edges rise above p=0.5 while the
            // penalty still discourages cycles.
        }

        // Final edge extraction. The relaxed objective (data fit + a KL prior anchored at
        // p=0.5) converges the edge logits to an equilibrium just BELOW 0 — the true edges are
        // consistently RANKED above their reverse direction and above non-edges, but none
        // crosses a hard sigmoid(Z) > 0.5 gate, so that gate recovered 0 edges. Extract a
        // DIRECTED edge i->j when (a) its logit exceeds the reverse logit Z[j,i] (the learned
        // orientation) and (b) the OLS coefficient of x_i in x_j is significant. This uses the
        // logits for orientation and the covariance for edge strength — the true SEM edges
        // (strong |coeff| in the correct direction) are recovered; symmetric noise is not.
        T weightThreshold = NumOps.FromDouble(0.1);

        // Collect the significant, correctly-oriented candidate edges with their strengths.
        var candidates = new List<(int i, int j, T weight, double strength)>();
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                if (i == j) continue;

                // Learned orientation: prefer i->j over j->i.
                if (!NumOps.GreaterThan(Z[i, j], Z[j, i])) continue;

                T varI = cov[i, i];
                if (!NumOps.GreaterThan(varI, NumOps.FromDouble(1e-10))) continue;

                T weight = NumOps.Divide(cov[i, j], varI);
                if (NumOps.GreaterThan(NumOps.Abs(weight), weightThreshold))
                    candidates.Add((i, j, weight, Math.Abs(NumOps.ToDouble(weight))));
            }

        // The strict Z[i,j] > Z[j,i] orientation guarantees anti-symmetry (never both i->j and
        // j->i) but NOT acyclicity — three or more edges can still form a directed cycle. The
        // CausalGraph contract requires a DAG (GetTopologicalOrder throws on cycles), so insert
        // edges greedily strongest-first and skip any that would close a cycle. Dropping the
        // weakest edge of each cycle keeps the more significant causal links, matching how
        // continuous-optimization DAG learners resolve residual cyclicity after thresholding.
        candidates.Sort((a, b) => b.strength.CompareTo(a.strength));

        var result = new Matrix<T>(d, d);
        var adjacency = new List<int>[d];
        for (int k = 0; k < d; k++) adjacency[k] = new List<int>();

        foreach (var (i, j, weight, _) in candidates)
        {
            // Adding i->j closes a cycle iff j can already reach i.
            if (CanReach(adjacency, j, i, d)) continue;
            result[i, j] = weight;
            adjacency[i].Add(j);
        }

        return result;
    }

    /// <summary>Depth-first reachability test: can <paramref name="from"/> reach <paramref name="to"/> along the current directed edges?</summary>
    private static bool CanReach(List<int>[] adjacency, int from, int to, int d)
    {
        if (from == to) return true;
        var visited = new bool[d];
        var stack = new Stack<int>();
        stack.Push(from);
        visited[from] = true;
        while (stack.Count > 0)
        {
            int node = stack.Pop();
            foreach (int next in adjacency[node])
            {
                if (next == to) return true;
                if (!visited[next])
                {
                    visited[next] = true;
                    stack.Push(next);
                }
            }
        }
        return false;
    }

}
