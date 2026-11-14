using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.CausalDiscovery.Bayesian;

/// <summary>
/// BCD-Nets — Bayesian Causal Discovery Networks.
/// </summary>
/// <remarks>
/// <para>
/// BCD-Nets use variational inference to approximate the joint posterior over DAG structures
/// and parameters. The graph structure is parameterized via Gumbel-Softmax continuous
/// relaxation of binary edge variables, and parameters (edge weights) are modeled via
/// a factorized Gaussian variational posterior. Both are optimized jointly via gradient
/// ascent on the ELBO.
/// </para>
/// <para>
/// <b>Algorithm:</b>
/// <list type="number">
/// <item>Initialize edge logits Z[i,j] and weight means mu[i,j], log-variances logvar[i,j]</item>
/// <item>Sample edges: E[i,j] ~ GumbelSigmoid(Z[i,j]/tau)</item>
/// <item>Sample weights: W[i,j] ~ N(mu[i,j], exp(logvar[i,j]))</item>
/// <item>Effective adjacency: A = E * W (element-wise)</item>
/// <item>Compute ELBO = E_q[log p(X|A)] - KL(q(Z)||p(Z)) - KL(q(W)||p(W)) - lambda*h(E)</item>
/// <item>Update Z, mu, logvar via gradient ascent</item>
/// <item>Anneal temperature tau and update augmented Lagrangian</item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> BCD-Nets learn both the graph structure AND the strength of each
/// connection simultaneously, using modern deep learning optimization techniques. They
/// provide uncertainty estimates for both.
/// </para>
/// <para>
/// Reference: Cundy et al. (2021), "BCD Nets: Scalable Variational Approaches for
/// Bayesian Causal Discovery", NeurIPS.
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
[ResearchPaper("BCD Nets: Scalable Variational Approaches for Bayesian Causal Discovery", "https://proceedings.neurips.cc/paper/2021/hash/5a378f8490c8d6af8647a753812f6e31-Abstract.html", Year = 2021, Authors = "Chris Cundy, Aditya Grover, Stefano Ermon")]
public class BCDNetsAlgorithm<T> : BayesianCausalBase<T>
{
    private readonly double _learningRate;
    private const double RhoMax = 1e+16;
    private const double RhoMultiplier = 10.0;
    private const double RhoIncreaseThreshold = 0.25;

    /// <inheritdoc/>
    public override string Name => "BCD-Nets";

    /// <inheritdoc/>
    public override bool SupportsNonlinear => false;

    public BCDNetsAlgorithm(CausalDiscoveryOptions? options = null)
    {
        ApplyBayesianOptions(options);
        _learningRate = options?.LearningRate ?? 0.01;
    }

    /// <inheritdoc/>
    protected override Matrix<T> DiscoverStructureCore(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;
        if (n < 3 || d < 2) return new Matrix<T>(d, d);

        var cov = ComputeCovarianceMatrix(data);
        var rng = Tensors.Helpers.RandomHelper.CreateSeededRandom(Seed);

        // Variational parameters: edge logits Z, weight means mu, weight log-variances logvar
        var offDiagonal = OffDiagonalMask(d);
        var c = Tensor<T>.FromMatrix(cov);

        // Initialize mu from OLS; initialize Z DIRECTION-AWARE per pair.
        //
        // The previous init turned BOTH directions of every correlated pair ON
        // (any |OLS slope| > 0.2 → logit +1), so the optimization started from a
        // dense graph of symmetric 2-cycles. The acyclicity penalty then pruned
        // directions essentially arbitrarily and the posterior routinely settled
        // on a REVERSED or spurious DAG (Markov-equivalent fits the covariance,
        // so the reconstruction term cannot recover the orientation once the
        // wrong basin is entered).
        //
        // For the linear-Gaussian equal-variance SEMs this variational sketch
        // models, the causal direction is identifiable by the smaller
        // conditional residual variance (Peters & Bühlmann 2014, the
        // identifiability result the NOTEARS family builds on):
        //   resVar(j|i) = cov[j,j] − cov[i,j]²/cov[i,i]   for i → j.
        // Initialize only the lower-residual direction of each pair ON so the
        // optimizer starts inside the true-DAG basin; the reverse logit starts
        // OFF and can still be revived by the reconstruction gradient if the
        // data disagrees.
        var (Z, mu, logvar) = InitializeVariationalParameters(c, offDiagonal, d);

        // Row-major flat positions of the off-diagonal entries, the order the weight noise is drawn in.
        var offDiagonalPositions = new int[d * (d - 1)];
        for (int i = 0, slot = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                if (i != j) offDiagonalPositions[slot++] = i * d + j;
        var noisePositions = new Tensor<int>(offDiagonalPositions, new[] { offDiagonalPositions.Length });

        // Augmented Lagrangian for acyclicity. Per the NOTEARS scheme BCD-Nets
        // builds on (Zheng et al. 2018 §3.2; Cundy et al. 2021 §4), the duals are
        // updated on an OUTER loop after an inner optimization with FIXED
        // (alpha, rho): alpha <- alpha + rho*h once per round, and rho is
        // escalated only when h fails to shrink by the relative factor 0.25
        // versus the previous round (h_new > 0.25 * h_prev). The previous code
        // performed the dual ascent after EVERY gradient step against an
        // ABSOLUTE h > 0.25 test, so rho exploded 10x per step to 1e16 within
        // ~16 steps and the acyclicity term symmetrically annihilated every
        // edge logit before the reconstruction term could orient the graph —
        // the discovered structure was always empty.
        var duals = new BcdDuals(NumOps.Zero, NumOps.One, double.PositiveInfinity);
        int innerSteps = Math.Max(1, Math.Min(25, NumSamples));

        for (int iter = 0; iter < NumSamples; iter++)
        {
            double tau = Math.Max(0.1, 1.0 * Math.Pow(0.95, iter));

            // Approximate N(0,1) reparameterization noise in [-1, 1), one draw per off-diagonal entry
            var uniform = Tensor<T>.CreateRandom(rng, offDiagonalPositions.Length);
            var noise = Engine.TensorScatter(new Tensor<T>(new[] { d * d }), noisePositions, uniform, 0).Reshape(new[] { d, d });

            T hValCurrent;
            (Z, mu, logvar, hValCurrent) = VariationalStep(c, Z, mu, logvar, noise, offDiagonal, duals, tau, d);

            // Dual ascent once per OUTER round (every innerSteps gradient steps),
            // per the NOTEARS / BCD-Nets augmented-Lagrangian schedule:
            //   alpha <- alpha + rho * h
            //   rho   <- 10 * rho   only if h did not shrink to <= 0.25 * h_prev
            if ((iter + 1) % innerSteps == 0)
                duals = UpdateDuals(duals, hValCurrent);
        }

        // Final output: threshold edge probabilities and use posterior mean weights
        var probability = ClampedSigmoid(Z, 0.1);
        var keep = Engine.TensorMultiply(offDiagonal, Engine.TensorMultiply(
            Engine.TensorGreaterThan(probability, NumOps.FromDouble(0.5)),
            Engine.TensorGreaterThan(Engine.TensorAbs(mu), NumOps.FromDouble(0.1))));
        return Engine.TensorWhere(keep, mu, Engine.TensorZerosLike(mu)).ToMatrix();
    }

    /// <summary>Augmented Lagrangian multipliers and the constraint value of the previous round.</summary>
    private readonly struct BcdDuals
    {
        public BcdDuals(T alpha, T rho, double hPrev)
        {
            Alpha = alpha;
            Rho = rho;
            HPrev = hPrev;
        }

        public T Alpha { get; }
        public T Rho { get; }
        public double HPrev { get; }
    }

    /// <summary>
    /// Initial logits, OLS weight means and log-variances. Off the diagonal: mu = cov[i,j]/cov[i,i]
    /// (when cov[i,i] &gt; 1e-10), logvar = -4, Z = -1 except Z = +1 on the lower-residual
    /// direction of each pair with an OLS slope above 0.2 in either direction.
    /// </summary>
    private (Tensor<T> Z, Tensor<T> Mu, Tensor<T> Logvar) InitializeVariationalParameters(
        Tensor<T> c, Tensor<T> offDiagonal, int d)
    {
        var zeros = Engine.TensorZerosLike(c);
        var ones = Engine.TensorOnesLike(c);
        var shape = new[] { d, d };
        var diagonal = Engine.TensorDiagonal(c);
        var rowVariance = Engine.TensorBroadcastTo(diagonal.Reshape(new[] { d, 1 }), shape); // cov[i, i]
        var columnVariance = Engine.TensorBroadcastTo(diagonal.Reshape(new[] { 1, d }), shape); // cov[j, j]
        T eps = NumOps.FromDouble(1e-10);

        var rowUsable = Engine.TensorGreaterThan(rowVariance, eps);
        var mu = Engine.TensorWhere(Engine.TensorMultiply(offDiagonal, rowUsable),
            Engine.TensorDivide(c, Engine.TensorWhere(rowUsable, rowVariance, ones)), zeros);
        var logvar = Engine.TensorMultiplyScalar(offDiagonal, NumOps.FromDouble(-4));
        var z = Engine.TensorNegate(offDiagonal);

        // Pair (i, j), i < j, is skipped when either variance is below 1e-10 or both |OLS| slopes are <= 0.2.
        var olsIJ = Engine.TensorAbs(Engine.TensorDivide(c, rowVariance));
        var olsJI = Engine.TensorAbs(Engine.TensorDivide(c, columnVariance));
        var weak = Engine.TensorMultiply(Engine.TensorLessEqual(olsIJ, Engine.TensorFullLike(c, NumOps.FromDouble(0.2))),
            Engine.TensorLessEqual(olsJI, Engine.TensorFullLike(c, NumOps.FromDouble(0.2))));
        var active = Engine.TensorMultiply(
            Engine.TensorMultiply(Engine.ScalarMinusTensor(NumOps.One, Engine.TensorLessThan(rowVariance, eps)),
                Engine.ScalarMinusTensor(NumOps.One, Engine.TensorLessThan(columnVariance, eps))),
            Engine.ScalarMinusTensor(NumOps.One, weak));

        // resJgivenI = vj - cij²/vi (i → j), resIgivenJ = vi - cij²/vj (j → i)
        var cSquared = Engine.TensorMultiply(c, c);
        var resJgivenI = Engine.TensorSubtract(columnVariance, Engine.TensorDivide(cSquared, rowVariance));
        var resIgivenJ = Engine.TensorSubtract(rowVariance, Engine.TensorDivide(cSquared, columnVariance));
        var forward = Engine.TensorLessEqual(resJgivenI, resIgivenJ);

        var forwardOn = Engine.TensorTriu(Engine.TensorMultiply(active, forward), 1);
        var reverseOn = Engine.TensorTranspose(Engine.TensorTriu(
            Engine.TensorMultiply(active, Engine.ScalarMinusTensor(NumOps.One, forward)), 1));
        z = Engine.TensorWhere(Engine.TensorAdd(forwardOn, reverseOn), ones, z);

        return (z, mu, logvar);
    }

    /// <summary>
    /// One variational gradient step on (Z, mu, logvar) with the given reparameterization noise
    /// (uniform draws in [0, 1), off-diagonal), returning the updated parameters and the NOTEARS
    /// acyclicity h(P) = tr(exp(P∘P)) - d of the edge probabilities used for the step.
    /// </summary>
    private (Tensor<T> Z, Tensor<T> Mu, Tensor<T> Logvar, T H) VariationalStep(
        Tensor<T> c, Tensor<T> Z, Tensor<T> mu, Tensor<T> logvar, Tensor<T> uniform, Tensor<T> offDiagonal,
        BcdDuals duals, double tau, int d)
    {
        var zeros = Engine.TensorZerosLike(Z);
        T tauT = NumOps.FromDouble(tau);

        // Edge probability via sigmoid; reparameterized weight sample w = mu + exp(logvar/2) * epsilon
        var P = Engine.TensorMultiply(offDiagonal, ClampedSigmoid(Z, tau));
        var stddev = Engine.TensorExp(Engine.TensorMultiplyScalar(logvar, NumOps.FromDouble(0.5)));
        var epsilon = Engine.TensorAddScalar(Engine.TensorMultiplyScalar(uniform, NumOps.FromDouble(2)), NumOps.FromDouble(-1));
        var wSampleDrawn = Engine.TensorAdd(mu, Engine.TensorMultiply(stddev, epsilon));
        var W = Engine.TensorWhere(offDiagonal, Engine.TensorMultiply(P, wSampleDrawn), zeros);

        // NOTEARS acyclicity: h(P) = tr(exp(P∘P)) - d
        var expPSq = Tensor<T>.FromMatrix(MatrixExponentialTaylor(Engine.TensorMultiply(P, P).ToMatrix(), d));
        T hValCurrent = NumOps.Subtract(Engine.TensorTrace(expPSq), NumOps.FromDouble(d));

        // Reconstruction gradient: -cov[i,j] + sum_{k != j} W[k,j] cov[i,k] (W has a zero diagonal)
        var reconGrad = Engine.TensorSubtract(Engine.TensorMatMul(c, W), c);

        // Gradient w.r.t. mu: P[i,j] * reconGrad + KL term 0.01 * mu
        var gradMu = Engine.TensorAdd(Engine.TensorMultiply(P, reconGrad),
            Engine.TensorMultiplyScalar(mu, NumOps.FromDouble(0.01)));

        // Gradient w.r.t. logvar through the KL term: 0.5 * (exp(logvar) - 1)
        var gradLogvar = Engine.TensorMultiplyScalar(
            Engine.TensorAddScalar(Engine.TensorExp(logvar), NumOps.FromDouble(-1)), NumOps.FromDouble(0.5));

        // Gradient w.r.t. Z through the edge probability
        var pUsable = Engine.TensorGreaterThan(P, NumOps.FromDouble(1e-10));
        var wSample = Engine.TensorWhere(pUsable,
            Engine.TensorDivide(W, Engine.TensorWhere(pUsable, P, Engine.TensorOnesLike(P))), zeros);
        var edgeGrad = Engine.TensorMultiply(reconGrad, wSample);

        // Acyclicity gradient: (alpha + rho*h) * [exp(P∘P)^T ∘ 2P][i,j]
        T augFactor = NumOps.Add(duals.Alpha, NumOps.Multiply(duals.Rho, hValCurrent));
        var acycGrad = Engine.TensorMultiplyScalar(
            Engine.TensorMultiply(Engine.TensorTranspose(expPSq), Engine.TensorMultiplyScalar(P, NumOps.FromDouble(2))),
            augFactor);

        var sigDeriv = Engine.TensorDivideScalar(Engine.TensorMultiply(P, Engine.ScalarMinusTensor(NumOps.One, P)),
            NumOps.Add(tauT, NumOps.FromDouble(1e-10)));
        var gradZ = Engine.TensorMultiply(Engine.TensorAdd(edgeGrad, acycGrad), sigDeriv);

        // Apply gradients off the diagonal. Z is clamped to [-4, 4]: during the early phase the
        // acyclicity penalty pushes EVERY logit down roughly symmetrically
        // (all 2-cycles violate the constraint), and an unclamped Z saturates
        // so deep that sigmoid'(Z/tau) ≈ 0 freezes ALL edge gradients — the
        // reconstruction term can then never pull the true edges back and the
        // posterior collapses to the empty graph. Clamping keeps the gate
        // responsive (standard logit clamping for Gumbel-sigmoid relaxations)
        // while sigmoid(±4) ≈ 0.018/0.982 still saturates the edge decision.
        T lr = NumOps.FromDouble(_learningRate);
        var zStepped = Engine.TensorSubtract(Z, Engine.TensorMultiplyScalar(gradZ, lr));
        var four = Engine.TensorFullLike(Z, NumOps.FromDouble(4.0));
        var minusFour = Engine.TensorNegate(four);
        zStepped = Engine.TensorWhere(Engine.TensorGreaterThan(zStepped, four), four, zStepped);
        zStepped = Engine.TensorWhere(Engine.TensorLessThan(zStepped, minusFour), minusFour, zStepped);

        var newZ = Engine.TensorWhere(offDiagonal, zStepped, Z);
        var newMu = Engine.TensorWhere(offDiagonal,
            Engine.TensorSubtract(mu, Engine.TensorMultiplyScalar(gradMu, lr)), mu);
        var newLogvar = Engine.TensorWhere(offDiagonal,
            Engine.TensorSubtract(logvar, Engine.TensorMultiplyScalar(gradLogvar, NumOps.FromDouble(_learningRate * 0.1))),
            logvar);

        return (newZ, newMu, newLogvar, hValCurrent);
    }

    /// <summary>
    /// Dual ascent for one outer round: only while the acyclicity constraint is genuinely
    /// violated (h &gt; 1e-8), alpha += rho·h and rho is multiplied by 10 (capped at RhoMax)
    /// when h did not shrink to at most 0.25 of the previous round's value.
    /// </summary>
    private BcdDuals UpdateDuals(BcdDuals duals, T hValCurrent)
    {
        double hNow = NumOps.ToDouble(hValCurrent);
        T alpha = duals.Alpha;
        T rho = duals.Rho;

        // Once h ≈ 0 (the edge probabilities form a DAG) the multiplier must stop
        // growing — alpha += rho·h is ~0 anyway, but skipping the rho-stall check
        // prevents a needless rho escalation from numerical jitter around zero.
        if (hNow > 1e-8)
        {
            alpha = NumOps.Add(alpha, NumOps.Multiply(rho, hValCurrent));

            T rhoMaxT = NumOps.FromDouble(RhoMax);
            if (!double.IsPositiveInfinity(duals.HPrev)
                && hNow > RhoIncreaseThreshold * duals.HPrev)
            {
                T newRho = NumOps.Multiply(rho, NumOps.FromDouble(RhoMultiplier));
                rho = NumOps.GreaterThan(newRho, rhoMaxT) ? rhoMaxT : newRho;
            }
        }

        return new BcdDuals(alpha, rho, hNow);
    }

}
