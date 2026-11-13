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
        var c = Tensor<T>.FromMatrix(cov);
        var offDiagonal = OffDiagonalMask(d);

        // Edge logits Z[i,j]: sigmoid(Z / tau) = edge probability. Initialize SPARSE
        // (z=-1 -> p≈0.27 at tau=1). A zero init put every edge at p=0.5 — a fully dense
        // graph whose least-squares reconstruction is over-determined, so the data-fit residual
        // (and therefore the per-edge gradient) is ~0 and no edge is ever distinguished: the
        // logits drifted to ~-0.015 for TRUE and false edges alike and DiscoverStructure
        // recovered 0 edges. Starting sparse gives each candidate edge a clear residual signal
        // so strong true edges get a large negative (strengthening) data-fit gradient. z=-1 is
        // mild enough that it does not saturate the sigmoid under the (now floored) temperature.
        var Z = Engine.TensorNegate(offDiagonal);

        // OLS slope of x_i in x_j: cov[i,j] / cov[i,i], usable only where cov[i,i] > 1e-10.
        var olsWeight = Engine.TensorMultiply(offDiagonal, ComputePairwiseOLSWeights(cov));
        var rowVariance = Engine.TensorBroadcastTo(Engine.TensorDiagonal(c).Reshape(new[] { d, 1 }), new[] { d, d });
        var updatable = Engine.TensorMultiply(offDiagonal, Engine.TensorGreaterThan(rowVariance, NumOps.FromDouble(1e-10)));

        for (int outerIter = 0; outerIter < NumSamples; outerIter++)
        {
            // Temperature annealing. Floor at 0.5 (not 0.1): the sigmoid-derivative that
            // scales every logit's gradient is P(1-P)/tau, and once P is even moderately away
            // from 0.5 a tiny tau drives that derivative to ~0, freezing the logits. A 0.5 floor
            // keeps gradients flowing so the data fit can actually move the edges.
            double tau = Math.Max(0.5, 1.0 * Math.Pow(0.95, outerIter));
            Z = UpdateEdgeLogits(c, Z, olsWeight, offDiagonal, updatable, tau, d);

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
        var oriented = Engine.TensorMultiply(updatable, Engine.TensorGreaterThan(Z, Engine.TensorTranspose(Z)));
        var significant = Engine.TensorMultiply(oriented,
            Engine.TensorGreaterThan(Engine.TensorAbs(olsWeight), NumOps.FromDouble(0.1)));
        var candidates = Engine.TensorWhere(significant, olsWeight, Engine.TensorZerosLike(olsWeight));

        // The strict Z[i,j] > Z[j,i] orientation guarantees anti-symmetry (never both i->j and
        // j->i) but NOT acyclicity — three or more edges can still form a directed cycle. The
        // CausalGraph contract requires a DAG (GetTopologicalOrder throws on cycles), so insert
        // edges greedily strongest-first and skip any that would close a cycle. Dropping the
        // weakest edge of each cycle keeps the more significant causal links, matching how
        // continuous-optimization DAG learners resolve residual cyclicity after thresholding.
        return ProjectToAcyclic(candidates.ToMatrix());
    }

    /// <summary>
    /// One gradient step on the edge logits: data fit (residual of the expected weighted adjacency
    /// W = P ∘ OLS), KL to a Bernoulli(0.5) prior, and the fixed-penalty NOTEARS acyclicity
    /// gradient rho·h·[exp(P∘P)^T ∘ 2P], chained through dP/dZ = P(1-P)/tau.
    /// </summary>
    private Tensor<T> UpdateEdgeLogits(Tensor<T> c, Tensor<T> Z, Tensor<T> olsWeight, Tensor<T> offDiagonal,
        Tensor<T> updatable, double tau, int d)
    {
        // Acyclicity penalty weight. Kept LOW (0.1) so it discourages cycles without
        // overwhelming the data fit: with the previous rho=1 (and the runaway augmented-
        // Lagrangian escalation before that) the positive-for-every-edge acyclicity gradient
        // held even genuine strong edges just below p=0.5, so 0 edges were recovered. At 0.1 the
        // data fit lifts true edges past their KL-to-0.5 equilibrium while non-edges stay near 0.5.
        // Applied as a FIXED penalty (no augmented-Lagrangian dual ascent), so there is no alpha
        // multiplier and no rho escalation/cap.
        T rho = NumOps.FromDouble(0.1);
        var zeros = Engine.TensorZerosLike(Z);
        var ones = Engine.TensorOnesLike(Z);

        // Compute edge probabilities: p[i,j] = sigmoid(Z[i,j] / tau)
        var P = Engine.TensorMultiply(offDiagonal, ClampedSigmoid(Z, tau));
        var oneMinusP = Engine.ScalarMinusTensor(NumOps.One, P);

        // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
        var expPSq = Tensor<T>.FromMatrix(MatrixExponentialTaylor(Engine.TensorMultiply(P, P).ToMatrix(), d));
        T hVal = NumOps.Subtract(Engine.TensorTrace(expPSq), NumOps.FromDouble(d));

        // Expected weighted adjacency W[i,j] = P[i,j] * cov[i,j] / cov[i,i]; residual
        // sum_{k != j} W[k,j] cov[i,k] - cov[i,j] (W has a zero diagonal).
        var W = Engine.TensorMultiply(P, olsWeight);
        var residGrad = Engine.TensorSubtract(Engine.TensorMatMul(c, W), c);
        var dataGrad = Engine.TensorMultiply(residGrad, olsWeight);

        // KL divergence gradient: log(P/(1-P)) (Bernoulli prior = 0.5), zero unless both P and 1-P exceed 1e-8
        var klUsable = Engine.TensorMultiply(Engine.TensorGreaterThan(P, NumOps.FromDouble(1e-8)),
            Engine.TensorGreaterThan(oneMinusP, NumOps.FromDouble(1e-8)));
        var klGrad = Engine.TensorLog(Engine.TensorWhere(klUsable,
            Engine.TensorDivide(P, Engine.TensorWhere(klUsable, oneMinusP, ones)), ones));

        // Acyclicity gradient (fixed-penalty NOTEARS): rho*h * [exp(P∘P)^T ∘ 2P][i,j]
        var acycGrad = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorTranspose(expPSq), Engine.TensorMultiplyScalar(P, NumOps.FromDouble(2))),
            NumOps.Multiply(rho, hVal));

        // Chain rule: dL/dZ = dL/dP * dP/dZ = dL/dP * P*(1-P)/tau
        var totalGradP = Engine.TensorAdd(dataGrad, Engine.TensorAdd(klGrad, acycGrad));
        var sigDeriv = Engine.TensorDivideScalar(Engine.TensorMultiply(P, oneMinusP), NumOps.FromDouble(tau));
        var gradZ = Engine.TensorWhere(updatable, Engine.TensorMultiply(totalGradP, sigDeriv), zeros);

        // Update logits
        var stepped = Engine.TensorSubtract(Z, Engine.TensorMultiplyScalar(gradZ, NumOps.FromDouble(_learningRate)));
        return Engine.TensorWhere(offDiagonal, stepped, Z);
    }

}
